import os
import threading
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import torch

from src.config import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, DEVICE

# Empirical class distribution from splits/train.csv:
# Robbery: 1330, Normal: 848, Burglary: 669, RoadAccidents: 335, Vandalism: 249, Shoplifting: 105, Fighting: 56
TRAIN_CLASS_COUNTS = {
    "Normal": 848,
    "Fighting": 56,
    "Robbery": 1330,
    "Shoplifting": 105,
    "Burglary": 669,
    "RoadAccidents": 335,
    "Vandalism": 249
}

TOTAL_TRAIN_SAMPLES = sum(TRAIN_CLASS_COUNTS.values())
CLASS_PRIORS = np.array([TRAIN_CLASS_COUNTS.get(c, 100) / TOTAL_TRAIN_SAMPLES for c in CLASSES], dtype=np.float32)
LOG_PRIORS = np.log(np.maximum(CLASS_PRIORS, 1e-6))


class CalibratedEventClassifier:
    """
    Expert Multimodal Event Classifier for VisionIQ.

    Combines:
    1. 3D CNN Spatio-Temporal Action Logits
    2. Bayesian Prior Logit Calibration (Counteracting 24:1 class imbalance)
    3. Spatial-Temporal Kinematics & Bounding Box Overlap (YOLOv8 + Optical Flow)
    4. Deterministic Multi-Person vs Single-Intruder Disambiguation Rules
    """

    def __init__(self, tau: float = 0.5):
        self.tau = tau
        self.log_priors = LOG_PRIORS
        self._yolo = None
        self._yolo_failed = False
        self._lock = threading.Lock()

    def _get_yolo(self):
        """Lazy thread-safe loader for YOLOv8n detector."""
        if self._yolo is None and not self._yolo_failed:
            with self._lock:
                if self._yolo is None and not self._yolo_failed:
                    try:
                        from ultralytics import YOLO
                        # Load yolov8n weights (cached locally)
                        model = YOLO("yolov8n.pt")
                        # Place on CUDA if available
                        if torch.cuda.is_available():
                            model.to("cuda")
                        self._yolo = model
                    except Exception as e:
                        print(f"[CalibratedEventClassifier] YOLO load failed, using logit calibration only: {e}")
                        self._yolo_failed = True
        return self._yolo

    def extract_visual_kinematics(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Samples keyframes across the clip and calculates person count,
        inter-person proximity, bounding box overlap (IoU), and motion energy.
        """
        stats = {
            "max_persons": 0,
            "avg_persons": 0.0,
            "min_dist_ratio": 999.0,
            "max_iou": 0.0,
            "avg_motion": 0.0,
            "vehicle_count": 0
        }

        if not frames:
            return stats

        yolo = self._get_yolo()
        total_f = len(frames)
        # Sample 4-6 frames uniformly
        num_samples = min(5, total_f)
        if num_samples <= 0:
            return stats

        sample_indices = np.linspace(0, total_f - 1, num_samples, dtype=int)

        person_boxes_per_frame = []
        total_vehicles = 0
        prev_gray = None
        motion_mags = []

        for idx in sample_indices:
            frame = frames[idx]
            if frame is None or frame.size == 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optical flow motion estimation
            if prev_gray is not None:
                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion_mags.append(float(np.mean(mag)))
                except Exception:
                    pass
            prev_gray = gray

            # Detect persons (class 0) & vehicles (classes 1, 2, 3, 5, 7 in COCO)
            p_boxes = []
            if yolo is not None:
                try:
                    results = yolo(frame, classes=[0, 1, 2, 3, 5, 7], verbose=False)[0]
                    boxes = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    cls_ids = results.boxes.cls.cpu().numpy()

                    for b, conf, cls_id in zip(boxes, confs, cls_ids):
                        if conf >= 0.30:
                            if int(cls_id) == 0:
                                p_boxes.append(b)
                            else:
                                total_vehicles += 1
                except Exception:
                    pass

            person_boxes_per_frame.append(p_boxes)

        # Calculate person proximity and IoU
        max_persons = max((len(pb) for pb in person_boxes_per_frame), default=0)
        avg_persons = float(np.mean([len(pb) for pb in person_boxes_per_frame])) if person_boxes_per_frame else 0.0
        avg_motion = float(np.mean(motion_mags)) if motion_mags else 0.0

        min_dist_ratio = 999.0
        max_iou = 0.0

        for p_boxes in person_boxes_per_frame:
            if len(p_boxes) >= 2:
                for i in range(len(p_boxes)):
                    for j in range(i + 1, len(p_boxes)):
                        b1, b2 = p_boxes[i], p_boxes[j]
                        # Compute intersection over union (IoU)
                        xA = max(b1[0], b2[0])
                        yA = max(b1[1], b2[1])
                        xB = min(b1[2], b2[2])
                        yB = min(b1[3], b2[3])
                        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
                        boxAArea = (b1[2] - b1[0]) * (b1[3] - b1[1])
                        boxBArea = (b2[2] - b2[0]) * (b2[3] - b2[1])
                        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
                        if iou > max_iou:
                            max_iou = float(iou)

                        # Center distance normalized by average person width
                        c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
                        c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
                        avg_w = ((b1[2] - b1[0]) + (b2[2] - b2[0])) / 2.0
                        dist = float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) / (avg_w + 1e-6))
                        if dist < min_dist_ratio:
                            min_dist_ratio = dist

        stats["max_persons"] = max_persons
        stats["avg_persons"] = avg_persons
        stats["min_dist_ratio"] = min_dist_ratio
        stats["max_iou"] = max_iou
        stats["avg_motion"] = avg_motion
        stats["vehicle_count"] = total_vehicles
        return stats

    def classify(
        self,
        frames: Optional[List[np.ndarray]] = None,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
        threshold: float = 0.50,
        kinematics_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Disambiguates and classifies an event window using combined multimodal evidence.

        Args:
            frames: Raw BGR frames of the video clip
            logits: Unnormalized logits (shape: [7]) from 3D CNN
            probs: Softmax probabilities (shape: [7]) if logits not available
            threshold: Confidence threshold for suspicious activity alert

        Returns:
            Dict containing predicted_class, confidence, is_suspicious, and diagnostic cues.
        """
        if logits is not None:
            raw_logits = np.asarray(logits, dtype=np.float32)
            raw_probs = np.exp(raw_logits - np.max(raw_logits))
            raw_probs = raw_probs / np.sum(raw_probs)
        elif probs is not None:
            raw_probs = np.asarray(probs, dtype=np.float32)
            raw_logits = np.log(np.maximum(raw_probs, 1e-7))
        else:
            raise ValueError("Either logits or probs must be provided to classify.")

        raw_idx = int(np.argmax(raw_probs))
        raw_pred = CLASSES[raw_idx]
        raw_conf = float(raw_probs[raw_idx])

        # 1. Balanced Logit Calibration (Logit Adjustment: logit - tau * log(P(c)))
        cal_logits = raw_logits - self.tau * self.log_priors
        cal_probs = np.exp(cal_logits - np.max(cal_logits))
        cal_probs = cal_probs / np.sum(cal_probs)
        cal_idx = int(np.argmax(cal_probs))
        cal_pred = CLASSES[cal_idx]
        cal_conf = float(cal_probs[cal_idx])

        # 2. Extract Visual Kinematics (Person Proximity, Overlap, Optical Flow)
        if kinematics_override is not None:
            kinematics = kinematics_override
        else:
            kinematics = self.extract_visual_kinematics(frames) if frames else {
                "max_persons": 0, "avg_persons": 0.0, "min_dist_ratio": 999.0,
                "max_iou": 0.0, "avg_motion": 0.0, "vehicle_count": 0
            }

        max_persons = kinematics["max_persons"]
        avg_persons = kinematics["avg_persons"]
        min_dist = kinematics["min_dist_ratio"]
        max_iou = kinematics["max_iou"]
        motion = kinematics["avg_motion"]
        vehicles = kinematics["vehicle_count"]

        final_pred = raw_pred
        final_conf = raw_conf

        # 3. Expert Disambiguation Logic
        # Visual signature of Fighting: 2+ persons in close proximity or overlap with dynamic combat motion
        has_fighting_kinematics = (
            (max_persons >= 2 or avg_persons >= 1.3) and
            (min_dist < 1.85 or max_iou > 0.04) and
            (motion > 0.75 or raw_pred in ("Robbery", "Burglary"))
        )

        # Rule A: Multi-person violent wrestling misclassified as Robbery or Burglary
        if raw_pred in ("Robbery", "Burglary") and has_fighting_kinematics:
            final_pred = "Fighting"
            # Ground confidence based on physical evidence
            final_conf = max(raw_conf, 0.78)

        # Rule B: High combat collision misclassified as Normal
        elif raw_pred == "Normal" and (max_iou > 0.12 or min_dist < 1.1) and motion > 1.2 and max_persons >= 2:
            final_pred = "Fighting"
            final_conf = 0.74

        # Rule C: Calibrated model preferred Fighting and kinematics confirm close multi-person interaction
        elif cal_pred == "Fighting" and (max_persons >= 2 or min_dist < 2.0):
            final_pred = "Fighting"
            final_conf = max(cal_conf, 0.72)

        # Rule D: Burglary protection (Burglary almost never involves multi-person wrestling)
        elif final_pred == "Fighting" and max_persons <= 1 and motion < 0.8:
            # Single intruder or quiet scene wrongly flagged as fighting
            final_pred = "Burglary" if raw_pred == "Burglary" else "Normal"
            final_conf = raw_conf

        # Rule E: Road Accident verification
        elif final_pred == "RoadAccidents" and vehicles == 0 and has_fighting_kinematics:
            final_pred = "Fighting"
            final_conf = 0.75

        is_suspicious = (final_pred != "Normal" and final_conf >= threshold)

        return {
            "predicted_class": final_pred,
            "confidence": round(float(final_conf), 4),
            "is_suspicious": bool(is_suspicious),
            "raw_class": raw_pred,
            "raw_confidence": round(float(raw_conf), 4),
            "calibrated_class": cal_pred,
            "calibrated_confidence": round(float(cal_conf), 4),
            "visual_cues": kinematics
        }


# Singleton accessor
_classifier_instance: Optional[CalibratedEventClassifier] = None
_instance_lock = threading.Lock()


def get_event_classifier(tau: float = 0.5) -> CalibratedEventClassifier:
    """Returns the singleton instance of CalibratedEventClassifier."""
    global _classifier_instance
    if _classifier_instance is None:
        with _instance_lock:
            if _classifier_instance is None:
                _classifier_instance = CalibratedEventClassifier(tau=tau)
    return _classifier_instance
