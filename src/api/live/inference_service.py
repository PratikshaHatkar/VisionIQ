import os
import time
import uuid
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import cv2
import torch

from src.config import EVIDENCE_DIR, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, DEVICE
from src.api.service import ModelManager, extract_clip_tensor
from src.api.live.storage import get_live_storage
from src.api.live.stream_manager import get_stream_manager
from src.api.live.notifications import get_notification_broker

SUSPICIOUS_ACTIVITIES = {
    "Fighting",
    "Robbery",
    "Shoplifting",
    "Burglary",
    "RoadAccidents",
    "Vandalism"
}

def utc_now():
    return datetime.now(timezone.utc)

class LiveInferenceService:
    """
    Independent background service that continuously evaluates frame windows from
    active camera streams, executes 3D CNN inference using the shared ModelManager,
    handles temporal event deduplication, saves evidence frames, and triggers notifications.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, check_interval_sec: float = 1.5, ongoing_timeout_sec: float = 6.0):
        self.check_interval_sec = check_interval_sec
        self.ongoing_timeout_sec = ongoing_timeout_sec
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Tracks ongoing incidents: camera_id -> dict
        self.ongoing_incidents: Dict[str, Dict[str, Any]] = {}
        self.state_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start(self):
        with self._lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._inference_loop, name="LiveInferenceLoop", daemon=True)
            self.thread.start()
            print("[LiveInferenceService] Service started.")

    def stop(self):
        with self._lock:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)
            self.ongoing_incidents.clear()
            print("[LiveInferenceService] Service stopped.")

    def _save_evidence_snapshot(self, frame, incident_id: str) -> Optional[str]:
        try:
            filename = f"live_{incident_id}_{int(time.time())}.jpg"
            disk_path = os.path.join(EVIDENCE_DIR, filename)
            cv2.imwrite(disk_path, frame)
            return f"/api/evidence/{filename}"
        except Exception as e:
            print(f"[LiveInferenceService] Failed to save evidence snapshot: {e}")
            return None

    def _inference_loop(self):
        model_mgr = ModelManager.get_instance()
        stream_mgr = get_stream_manager()
        storage = get_live_storage()
        notif_broker = get_notification_broker()

        while self.running:
            time.sleep(self.check_interval_sec)

            # Get copy of active workers
            with stream_mgr.workers_lock:
                active_workers = list(stream_mgr.workers.values())

            if not active_workers:
                continue

            for worker in active_workers:
                if not worker.running or worker.status != "online":
                    continue

                cam_id = worker.camera_id
                frames = worker.get_frame_window(num_frames=16)
                if not frames or len(frames) < 16:
                    continue

                # Run inference with existing trained model
                tensor = extract_clip_tensor(frames)
                if tensor is None:
                    continue

                pred_class = "Normal"
                confidence = 0.0

                try:
                    # Thread-safe forward pass sharing the model with recorded-video analysis
                    with model_mgr.inference_lock:
                        if not model_mgr.is_loaded or model_mgr.model is None:
                            model_mgr.load_model()

                        device = model_mgr.device
                        tensor = tensor.to(device)
                        with torch.no_grad():
                            outputs = model_mgr.model(tensor)
                            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

                    top_idx = int(np.argmax(probs))
                    pred_class = IDX_TO_CLASS.get(top_idx, "Normal")
                    confidence = float(probs[top_idx])
                except Exception as infer_err:
                    print(f"[LiveInferenceService] Inference error on cam {cam_id}: {infer_err}")
                    continue

                threshold = worker.confidence_threshold
                is_suspicious = (pred_class in SUSPICIOUS_ACTIVITIES) and (confidence >= threshold)

                now = utc_now()
                now_epoch = time.time()

                with self.state_lock:
                    ongoing = self.ongoing_incidents.get(cam_id)

                    if is_suspicious:
                        if ongoing and ongoing["activity_label"] == pred_class:
                            # 1. GROUPING & DEDUPLICATION: Extend existing ongoing incident
                            ongoing["last_seen_epoch"] = now_epoch
                            ongoing["end_time"] = now
                            start_time = ongoing["start_time"]
                            duration = max(1.0, (now - start_time).total_seconds())
                            ongoing["duration_seconds"] = duration
                            if confidence > ongoing["confidence"]:
                                ongoing["confidence"] = confidence

                            # Update database record without creating duplicate alerts
                            try:
                                storage.update_incident(ongoing["incident_id"], {
                                    "end_time": now,
                                    "duration_seconds": duration,
                                    "confidence": ongoing["confidence"]
                                })
                            except Exception as db_err:
                                print(f"[LiveInferenceService] Incident update error: {db_err}")
                        else:
                            # 2. NEW INCIDENT DETECTED
                            incident_id = str(uuid.uuid4())
                            # Save snapshot from the middle frame of the window
                            mid_frame = frames[len(frames) // 2]
                            snapshot_url = self._save_evidence_snapshot(mid_frame, incident_id)

                            incident_data = {
                                "id": incident_id,
                                "incident_id": incident_id,
                                "camera_id": cam_id,
                                "camera_name": worker.name,
                                "location": worker.location,
                                "owner_id": worker.owner_id,
                                "activity_label": pred_class,
                                "confidence": confidence,
                                "start_time": now,
                                "end_time": now,
                                "duration_seconds": 1.5,
                                "evidence_snapshot": snapshot_url,
                                "is_acknowledged": False,
                                "created_at": now
                            }

                            try:
                                storage.create_incident(incident_data)
                            except Exception as db_err:
                                print(f"[LiveInferenceService] Incident create error: {db_err}")

                            # Dispatch notification to camera owner
                            notif_data = {
                                "camera_id": cam_id,
                                "incident_id": incident_id,
                                "owner_id": worker.owner_id,
                                "activity_label": pred_class,
                                "camera_name": worker.name,
                                "location": worker.location,
                                "confidence": confidence,
                                "evidence_snapshot": snapshot_url,
                                "created_at": now
                            }
                            notif_broker.broadcast_incident_sync(notif_data)

                            # Track ongoing incident state
                            self.ongoing_incidents[cam_id] = {
                                "incident_id": incident_id,
                                "activity_label": pred_class,
                                "confidence": confidence,
                                "start_time": now,
                                "last_seen_epoch": now_epoch,
                                "evidence_snapshot": snapshot_url
                            }

                    else:
                        # Non-suspicious (Normal or low confidence)
                        if ongoing:
                            # Check if ongoing incident has elapsed past timeout
                            if now_epoch - ongoing["last_seen_epoch"] > self.ongoing_timeout_sec:
                                # Close ongoing incident
                                del self.ongoing_incidents[cam_id]

def get_live_inference_service() -> LiveInferenceService:
    return LiveInferenceService.get_instance()
