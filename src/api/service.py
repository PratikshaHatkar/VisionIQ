import os
import uuid
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch
from torch.amp import autocast

from src.config import (
    CHECKPOINTS_DIR, CLASSES, CLASS_TO_IDX, NUM_CLASSES,
    NUM_FRAMES, IMG_SIZE, NORM_MEAN, NORM_STD,
    DEVICE, USE_AMP, EVIDENCE_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_WINDOW_SEC, DEFAULT_STRIDE_SEC
)
from src.model import build_model
from src.api.database import SessionLocal
from src.api.models import Video, Event
from src.api.storage import get_storage

def format_hhmmss(seconds: float) -> str:
    """
    Formats a timestamp in seconds to standard HH:MM:SS format.
    """
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

class ModelManager:
    """
    Thread-safe Singleton manager for loading and serving the trained video action model.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.model_name = "r2plus1d_18"
        self.is_loaded = False
        self.checkpoint_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
        self.device = DEVICE
        self.inference_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def load_model(self):
        with self.inference_lock:
            if self.is_loaded and self.model is not None:
                return self.model

            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(
                    f"Model checkpoint not found at: {self.checkpoint_path}. "
                    f"Please verify that the trained checkpoint exists."
                )

            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model_name = checkpoint.get("model_name", "r2plus1d_18")
            
            # Build architecture and load weights
            self.model = build_model(
                model_name=self.model_name,
                num_classes=NUM_CLASSES,
                pretrained=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"[ModelManager] Model '{self.model_name}' successfully loaded on {self.device}.")
            return self.model

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.model_name,
            "loaded": self.is_loaded,
            "device": str(self.device),
            "classes": CLASSES,
            "checkpoint": self.checkpoint_path
        }

def probe_video_file(file_path: str) -> Dict[str, Any]:
    """
    Extracts essential video metadata using OpenCV.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total_frames / fps if total_frames > 0 and fps > 0 else 0.0

    return {
        "duration": round(duration, 2),
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}" if width and height else None
    }

def extract_clip_tensor(frames: List[np.ndarray], target_num_frames: int = NUM_FRAMES, img_size: int = IMG_SIZE) -> Optional[torch.Tensor]:
    """
    Uniformly samples target_num_frames from raw BGR video frames and returns normalized tensor (1, 3, T, H, W).
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
    mean = np.array(NORM_MEAN, dtype=np.float32)
    std = np.array(NORM_STD, dtype=np.float32)
    sampled = sampled / 255.0
    sampled = (sampled - mean) / std
    tensor = np.transpose(sampled, (3, 0, 1, 2))  # (T, H, W, C) -> (C, T, H, W)
    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)

def analyze_video(
    video_path: str,
    video_id: str,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC
) -> List[Dict[str, Any]]:
    """
    Performs sliding-window inference on a video and extracts genuine timestamped events and evidence snapshots.
    """
    model_mgr = ModelManager.get_instance()
    model = model_mgr.load_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot read video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    total_frames = len(all_frames)
    if total_frames == 0:
        raise ValueError("Video contains no valid readable frames.")

    window_frames_len = max(1, int(window_sec * fps))
    stride_frames_len = max(1, int(stride_sec * fps))

    # Generate sliding window ranges (supporting short videos gracefully)
    windows = []
    if total_frames <= window_frames_len:
        windows.append((0, total_frames))
    else:
        for start in range(0, total_frames - window_frames_len + 1, stride_frames_len):
            windows.append((start, start + window_frames_len))

    window_predictions = []

    with model_mgr.inference_lock:
        for start_f, end_f in windows:
            window_raw_frames = all_frames[start_f:end_f]
            clip_tensor = extract_clip_tensor(window_raw_frames)
            if clip_tensor is None:
                continue

            clip_tensor = clip_tensor.to(model_mgr.device)
            with torch.no_grad():
                with autocast('cuda', enabled=USE_AMP and model_mgr.device.type == "cuda"):
                    logits = model(clip_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            pred_idx = int(np.argmax(probs))
            pred_class = CLASSES[pred_idx]
            confidence = float(probs[pred_idx])
            start_time = start_f / fps
            end_time = end_f / fps

            window_predictions.append({
                "start_time": start_time,
                "end_time": end_time,
                "start_frame": start_f,
                "end_frame": end_f,
                "predicted_class": pred_class,
                "confidence": confidence,
                "is_suspicious": (pred_class != "Normal" and confidence >= threshold)
            })

    # Merge consecutive/overlapping windows of the same activity
    merged_events = []
    current_event = None

    for win in window_predictions:
        if win["is_suspicious"]:
            if current_event is None:
                current_event = {
                    "class": win["predicted_class"],
                    "start_time": win["start_time"],
                    "end_time": win["end_time"],
                    "confidences": [win["confidence"]],
                    "best_confidence": win["confidence"],
                    "best_frame_idx": (win["start_frame"] + win["end_frame"]) // 2
                }
            elif current_event["class"] == win["predicted_class"]:
                current_event["end_time"] = win["end_time"]
                current_event["confidences"].append(win["confidence"])
                if win["confidence"] > current_event["best_confidence"]:
                    current_event["best_confidence"] = win["confidence"]
                    current_event["best_frame_idx"] = (win["start_frame"] + win["end_frame"]) // 2
            else:
                merged_events.append(current_event)
                current_event = {
                    "class": win["predicted_class"],
                    "start_time": win["start_time"],
                    "end_time": win["end_time"],
                    "confidences": [win["confidence"]],
                    "best_confidence": win["confidence"],
                    "best_frame_idx": (win["start_frame"] + win["end_frame"]) // 2
                }
        else:
            if current_event is not None:
                merged_events.append(current_event)
                current_event = None

    if current_event is not None:
        merged_events.append(current_event)

    # Build finalized event list and save genuine evidence snapshot images
    results = []
    for ev in merged_events:
        event_id = str(uuid.uuid4())
        best_frame_idx = min(max(0, ev["best_frame_idx"]), total_frames - 1)
        evidence_frame = all_frames[best_frame_idx]

        # Save snapshot frame to disk
        snapshot_filename = f"{event_id}.jpg"
        snapshot_path = os.path.join(EVIDENCE_DIR, snapshot_filename)
        cv2.imwrite(snapshot_path, evidence_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        start_sec = round(ev["start_time"], 2)
        end_sec = round(ev["end_time"], 2)

        results.append({
            "event_id": event_id,
            "video_id": video_id,
            "activity_label": ev["class"],
            "confidence": round(float(ev["best_confidence"]), 4),
            "start_timestamp": start_sec,
            "end_timestamp": end_sec,
            "start_time_formatted": format_hhmmss(start_sec),
            "end_time_formatted": format_hhmmss(end_sec),
            "evidence_snapshot": f"/api/evidence/{snapshot_filename}",
            "evidence_clip": None,
            "processing_status": "completed",
            "created_at": datetime.now(timezone.utc)
        })

    return results

def process_video_background(
    video_id: str,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC
):
    """
    Background worker function executed asynchronously without blocking FastAPI HTTP threads.
    Persists results directly into MongoDB with automatic SQLite synchronization and deduplication.
    """
    repo = get_storage()
    try:
        video = repo.get_video(video_id)
        if not video:
            print(f"[Worker] Video {video_id} not found in storage.")
            return

        repo.update_video_status(video_id, status="processing")

        # Run model analysis
        events_data = analyze_video(
            video_path=video["file_path"],
            video_id=video_id,
            threshold=threshold,
            window_sec=window_sec,
            stride_sec=stride_sec
        )

        # Idempotent save: clears any previous events for this video and persists new ones
        repo.save_events(video_id, events_data)

        repo.update_video_status(video_id, status="completed")
        print(f"[Worker] Video {video_id} processed successfully: {len(events_data)} events saved.")

    except Exception as e:
        repo.update_video_status(video_id, status="failed", error_message=str(e))
        print(f"[Worker Error] Failed processing video {video_id}: {e}")
