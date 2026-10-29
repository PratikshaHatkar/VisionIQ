import os
import argparse
import json
import numpy as np
import cv2
import torch
from torch.amp import autocast

from src.config import (
    CHECKPOINTS_DIR, CLASSES, CLASS_TO_IDX,
    NUM_CLASSES, NUM_FRAMES, IMG_SIZE,
    NORM_MEAN, NORM_STD, DEVICE, USE_AMP
)
from src.model import build_model
from src.api.classifier import get_event_classifier

def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 100)
    return f"{mins:02d}:{secs:02d}.{millis:02d}"

def extract_clip_frames(frames, target_num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    """
    Uniformly samples target_num_frames from a list of BGR video frames.
    """
    total = len(frames)
    if total == 0:
        return None
    indices = np.linspace(0, total - 1, target_num_frames, dtype=int)
    sampled = []
    for idx in indices:
        f = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (img_size, img_size))
        sampled.append(f)
        
    sampled = np.array(sampled, dtype=np.float32)
    # Normalize
    mean = np.array(NORM_MEAN, dtype=np.float32)
    std = np.array(NORM_STD, dtype=np.float32)
    sampled = sampled / 255.0
    sampled = (sampled - mean) / std
    # (T, H, W, C) -> (C, T, H, W)
    tensor = np.transpose(sampled, (3, 0, 1, 2))
    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0) # (1, C, T, H, W)

def detect_suspicious_events(video_path, model, threshold=0.55, window_sec=2.0, stride_sec=1.0):
    """
    Performs sliding-window inference over a long CCTV video and identifies suspicious events.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Duration: {format_timestamp(total_duration)} ({total_frames} frames @ {fps:.1f} fps)")
    print(f"Sliding Window: {window_sec}s (stride {stride_sec}s), Alert Threshold: {threshold*100:.0f}%\n")
    
    window_frames_len = int(window_sec * fps)
    stride_frames_len = max(1, int(stride_sec * fps))
    
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    
    window_predictions = []
    
    for start_frame in range(0, len(all_frames) - window_frames_len + 1, stride_frames_len):
        end_frame = start_frame + window_frames_len
        window_raw_frames = all_frames[start_frame:end_frame]
        
        clip_tensor = extract_clip_frames(window_raw_frames)
        if clip_tensor is None:
            continue
            
        clip_tensor = clip_tensor.to(DEVICE)
        with torch.no_grad():
            with autocast('cuda', enabled=USE_AMP and DEVICE.type == 'cuda'):
                logits = model(clip_tensor)
                raw_logits_np = logits.squeeze(0).cpu().float().numpy()
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                
        classifier = get_event_classifier()
        class_result = classifier.classify(
            frames=window_raw_frames,
            logits=raw_logits_np,
            probs=probs,
            threshold=threshold
        )
        pred_class = class_result["predicted_class"]
        confidence = class_result["confidence"]
        
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        window_predictions.append({
            'start_time': start_time,
            'end_time': end_time,
            'predicted_class': pred_class,
            'confidence': confidence,
            'is_suspicious': (pred_class != "Normal" and confidence >= threshold)
        })
        
    # Aggregate consecutive/overlapping suspicious windows into discrete alert events
    candidate_events = []
    current_event = None
    
    for win in window_predictions:
        if win['is_suspicious']:
            if current_event is None:
                current_event = {
                    'class': win['predicted_class'],
                    'start_time': win['start_time'],
                    'end_time': win['end_time'],
                    'confidences': [win['confidence']]
                }
            elif current_event['class'] == win['predicted_class']:
                current_event['end_time'] = win['end_time']
                current_event['confidences'].append(win['confidence'])
            else:
                candidate_events.append({
                    'class': current_event['class'],
                    'start_time': current_event['start_time'],
                    'end_time': current_event['end_time'],
                    'avg_confidence': float(np.mean(current_event['confidences'])),
                    'max_confidence': float(np.max(current_event['confidences']))
                })
                current_event = {
                    'class': win['predicted_class'],
                    'start_time': win['start_time'],
                    'end_time': win['end_time'],
                    'confidences': [win['confidence']]
                }
        else:
            if current_event is not None:
                candidate_events.append({
                    'class': current_event['class'],
                    'start_time': current_event['start_time'],
                    'end_time': current_event['end_time'],
                    'avg_confidence': float(np.mean(current_event['confidences'])),
                    'max_confidence': float(np.max(current_event['confidences']))
                })
                current_event = None
                
    if current_event is not None:
        candidate_events.append({
            'class': current_event['class'],
            'start_time': current_event['start_time'],
            'end_time': current_event['end_time'],
            'avg_confidence': float(np.mean(current_event['confidences'])),
            'max_confidence': float(np.max(current_event['confidences']))
        })
        
    return window_predictions, candidate_events

def main():
    parser = argparse.ArgumentParser(description="Inference on Unseen CCTV Video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(CHECKPOINTS_DIR, "best_model.pth"))
    parser.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold for anomaly alert")
    parser.add_argument("--window_sec", type=float, default=2.0)
    parser.add_argument("--stride_sec", type=float, default=1.0)
    parser.add_argument("--save_json", type=str, default=None, help="Save alerts to JSON file")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}. Please train the model first.")
        
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    model_name = checkpoint.get('model_name', 'r2plus1d_18')
    
    model = build_model(model_name=model_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    _, events = detect_suspicious_events(
        args.video, model,
        threshold=args.threshold,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec
    )
    
    print("=" * 60)
    print("             DETECTED SUSPICIOUS EVENTS              ")
    print("=" * 60)
    if not events:
        print(">> No suspicious activities detected (Video classified as Normal).")
    else:
        for idx, ev in enumerate(events, 1):
            s_str = format_timestamp(ev['start_time'])
            e_str = format_timestamp(ev['end_time'])
            print(f"[{idx:02d}] [{s_str} -> {e_str}] ALERT: {ev['class'].upper()} "
                  f"(Peak Conf: {ev['max_confidence']*100:.1f}%, Avg Conf: {ev['avg_confidence']*100:.1f}%)")
    print("=" * 60)
    
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump({'video': args.video, 'events': events}, f, indent=4)
        print(f"Saved event log to {args.save_json}")

if __name__ == "__main__":
    main()
