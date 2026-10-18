import os
import glob
import csv
import random
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    DATASET_DIR, LABELS_DIR, SPLITS_DIR,
    CLASSES, CLASS_TO_IDX, NUM_CLASSES,
    NUM_FRAMES, IMG_SIZE, NORM_MEAN, NORM_STD,
    RANDOM_SEED, NUM_WORKERS
)

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_parent_video_id(clip_name):
    """
    Extracts the parent source recording ID from a clip name.
    Example: 'Fighting002_x264_3' -> 'Fighting002_x264'
    """
    clean_name = clip_name.strip()
    # Correct known DCSASS typo where leading 'R' was truncated
    if clean_name.startswith("oadAccidents"):
        clean_name = "R" + clean_name
    parts = clean_name.rsplit('_', 1)
    return parts[0]

def build_dataset_metadata(max_normal_per_category=200):
    """
    Scans the DCSASS dataset and label CSVs, maps each clip to its path on disk,
    assigns the 7 target activity classes, and balances the Normal background class.
    """
    set_seed(RANDOM_SEED)
    categories = ["Fighting", "Robbery", "Shoplifting", "Burglary", "RoadAccidents", "Vandalism"]
    
    records = []
    normal_records_by_cat = {cat: [] for cat in categories}
    
    for cat in categories:
        csv_path = os.path.join(LABELS_DIR, f"{cat}.csv")
        cat_dir = os.path.join(DATASET_DIR, cat)
        if not os.path.exists(csv_path):
            continue
            
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                clip_raw = row[0].strip()
                cat_label = row[1].strip()
                flag = row[2].strip()
                
                # Normalize filename
                clip_clean = clip_raw
                if clip_clean.startswith("oadAccidents"):
                    clip_clean = "R" + clip_clean
                    
                parent_id = parse_parent_video_id(clip_clean)
                parent_folder = f"{parent_id}.mp4"
                clip_filename = f"{clip_clean}.mp4"
                video_path = os.path.join(cat_dir, parent_folder, clip_filename)
                
                # Verify file existence on disk
                if not os.path.exists(video_path):
                    continue
                    
                if flag == '1':
                    # Suspicious activity clip
                    records.append({
                        'clip_name': clip_clean,
                        'parent_video_id': parent_id,
                        'video_path': video_path,
                        'category': cat,
                        'label': CLASS_TO_IDX[cat]
                    })
                elif flag == '0':
                    # Normal background clip
                    normal_records_by_cat[cat].append({
                        'clip_name': clip_clean,
                        'parent_video_id': parent_id,
                        'video_path': video_path,
                        'category': 'Normal',
                        'label': CLASS_TO_IDX['Normal']
                    })
                    
    # Subsample Normal clips evenly to prevent extreme class imbalance
    for cat, norm_list in normal_records_by_cat.items():
        if len(norm_list) > max_normal_per_category:
            sampled_norm = random.sample(norm_list, max_normal_per_category)
        else:
            sampled_norm = norm_list
        records.extend(sampled_norm)
        
    df = pd.DataFrame(records)
    print(f"Total verified dataset clips: {len(df)}")
    print("Class distribution:\n", df['category'].value_counts())
    return df

def generate_splits(df, test_size=0.15, val_size=0.15):
    """
    Splits the dataset into train, validation, and test subsets strictly by
    parent_video_id (source recording) to prevent scene/camera data leakage,
    while stratifying parent videos by primary category to ensure all classes
    are represented across train, val, and test splits.
    """
    set_seed(RANDOM_SEED)
    
    # Determine the primary category for each parent_video_id
    # (If a parent has suspicious clips, its primary category is that anomaly)
    parent_cat_map = {}
    for parent_id, group in df.groupby('parent_video_id'):
        suspicious_cats = [c for c in group['category'].unique() if c != 'Normal']
        if suspicious_cats:
            parent_cat_map[parent_id] = suspicious_cats[0]
        else:
            parent_cat_map[parent_id] = 'Normal'
            
    # Group parents by category
    cats_to_parents = {}
    for parent_id, cat in parent_cat_map.items():
        cats_to_parents.setdefault(cat, []).append(parent_id)
        
    train_parents = set()
    val_parents = set()
    test_parents = set()
    
    for cat, parents in cats_to_parents.items():
        random.shuffle(parents)
        n = len(parents)
        if n == 1:
            train_parents.add(parents[0])
        elif n == 2:
            train_parents.add(parents[0])
            test_parents.add(parents[1])
        else:
            n_test = max(1, int(round(n * test_size)))
            n_val = max(1, int(round(n * val_size)))
            # Ensure at least 1 in train
            if n - n_test - n_val < 1:
                n_test = 1
                n_val = 1
            n_train = n - n_test - n_val
            
            test_p = parents[:n_test]
            val_p = parents[n_test:n_test + n_val]
            train_p = parents[n_test + n_val:]
            
            test_parents.update(test_p)
            val_parents.update(val_p)
            train_parents.update(train_p)
            
    train_df = df[df['parent_video_id'].isin(train_parents)].copy().reset_index(drop=True)
    val_df = df[df['parent_video_id'].isin(val_parents)].copy().reset_index(drop=True)
    test_df = df[df['parent_video_id'].isin(test_parents)].copy().reset_index(drop=True)
    
    # Zero-Leakage Assertion Check
    assert len(train_parents.intersection(val_parents)) == 0, "Data Leakage detected between Train and Val!"
    assert len(train_parents.intersection(test_parents)) == 0, "Data Leakage detected between Train and Test!"
    assert len(val_parents.intersection(test_parents)) == 0, "Data Leakage detected between Val and Test!"
    
    print("\n--- Category-Stratified Zero-Leakage Split Summary ---")
    print(f"Train : {len(train_df)} clips across {len(train_parents)} source recordings")
    print(f"Val   : {len(val_df)} clips across {len(val_parents)} source recordings")
    print(f"Test  : {len(test_df)} clips across {len(test_parents)} source recordings")
    
    # Save split metadata to CSV
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path = os.path.join(SPLITS_DIR, "val.csv")
    test_path = os.path.join(SPLITS_DIR, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return train_df, val_df, test_df

def compute_class_weights(train_df):
    """
    Computes inverse-frequency class weights for Cross-Entropy Loss
    to handle class imbalances.
    """
    counts = train_df['label'].value_counts().to_dict()
    total_samples = len(train_df)
    weights = []
    for c in range(NUM_CLASSES):
        count = counts.get(c, 1)
        w = total_samples / (NUM_CLASSES * count)
        weights.append(w)
    # Normalize weights so mean is 1.0
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

class VideoDataset(Dataset):
    """
    PyTorch Video Dataset with on-the-fly frame sampling and data augmentation.
    Produces tensors of shape (Channels=3, Time=NUM_FRAMES, Height, Width).
    """
    def __init__(self, df, num_frames=NUM_FRAMES, img_size=IMG_SIZE, is_training=False):
        self.df = df
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_training = is_training
        self.mean = np.array(NORM_MEAN, dtype=np.float32)
        self.std = np.array(NORM_STD, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0 or not cap.isOpened():
            cap.release()
            # Return empty black clip if file failed to open
            return np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.float32)
            
        # Select uniform frame indices across clip
        if total_frames <= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
        frames = []
        frame_idx = 0
        read_success = True
        
        # Read frames efficiently
        curr_target_idx = 0
        while cap.isOpened() and curr_target_idx < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx == indices[curr_target_idx]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frames.append(frame)
                curr_target_idx += 1
                # If subsequent indices target the same frame (in very short clips)
                while curr_target_idx < self.num_frames and indices[curr_target_idx] == frame_idx:
                    frames.append(frame.copy())
                    curr_target_idx += 1
            frame_idx += 1
            
        cap.release()
        
        # Pad with last frame if needed
        while len(frames) < self.num_frames:
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
                
        return np.array(frames, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']
        label = int(row['label'])
        
        frames = self._load_frames(video_path)
        
        # Augmentations for training
        if self.is_training:
            # Random horizontal flip
            if random.random() > 0.5:
                frames = np.flip(frames, axis=2).copy()
            # Random brightness adjustment (+- 15%)
            brightness_factor = random.uniform(0.85, 1.15)
            frames = np.clip(frames * brightness_factor, 0, 255)
            
        # Normalize: [0, 255] -> [0, 1] then subtract mean, divide std
        frames = frames / 255.0
        frames = (frames - self.mean) / self.std
        
        # Transpose from (T, H, W, C) to (C, T, H, W) for PyTorch 3D CNNs
        frames = np.transpose(frames, (3, 0, 1, 2))
        
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_dataloaders(batch_size=4):
    """
    Helper function to load train, validation, and test DataLoaders.
    """
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path = os.path.join(SPLITS_DIR, "val.csv")
    test_path = os.path.join(SPLITS_DIR, "test.csv")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        df = build_dataset_metadata()
        train_df, val_df, test_df = generate_splits(df)
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
    train_ds = VideoDataset(train_df, is_training=True)
    val_ds = VideoDataset(val_df, is_training=False)
    test_ds = VideoDataset(test_df, is_training=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_df
