import os
import pytest
import numpy as np
import cv2
import torch

from src.config import CLASSES, CLASS_TO_IDX
from src.api.classifier import CalibratedEventClassifier, get_event_classifier


def test_01_classifier_initialization():
    classifier = get_event_classifier()
    assert classifier is not None
    assert len(classifier.log_priors) == len(CLASSES)
    # Robbery has the highest prior count, so log_prior for Robbery > Fighting
    rob_idx = CLASS_TO_IDX["Robbery"]
    fight_idx = CLASS_TO_IDX["Fighting"]
    assert classifier.log_priors[rob_idx] > classifier.log_priors[fight_idx]


def test_02_balanced_logit_adjustment():
    classifier = CalibratedEventClassifier(tau=0.8)
    # Suppose raw logits give Robbery a slight advantage due to prior bias
    rob_idx = CLASS_TO_IDX["Robbery"]
    fight_idx = CLASS_TO_IDX["Fighting"]

    fake_logits = np.zeros(len(CLASSES), dtype=np.float32)
    fake_logits[rob_idx] = 2.0
    fake_logits[fight_idx] = 1.0

    # With tau logit adjustment, the 24:1 penalty on Robbery boosts Fighting
    cal_logits = fake_logits - classifier.tau * classifier.log_priors
    assert cal_logits[fight_idx] > cal_logits[rob_idx]


def test_03_fighting_kinematic_disambiguation_from_robbery():
    classifier = CalibratedEventClassifier(tau=0.5)

    # Base logits predict Robbery
    logits = np.zeros(len(CLASSES), dtype=np.float32)
    logits[CLASS_TO_IDX["Robbery"]] = 3.5
    logits[CLASS_TO_IDX["Fighting"]] = -1.0

    # Kinematics indicating 2 people in close contact (<1.5 distance ratio, overlapping, active motion)
    fighting_kinematics = {
        "max_persons": 2,
        "avg_persons": 2.0,
        "min_dist_ratio": 0.85,
        "max_iou": 0.25,
        "avg_motion": 3.5,
        "vehicle_count": 0
    }

    res = classifier.classify(frames=None, logits=logits, threshold=0.50, kinematics_override=fighting_kinematics)
    assert res["raw_class"] == "Robbery"
    # Expert disambiguation should identify the 2-person physical grappling and reclassify to Fighting
    assert res["predicted_class"] == "Fighting"
    assert res["is_suspicious"] is True
    assert res["confidence"] >= 0.70


def test_04_burglary_single_intruder_protection():
    classifier = CalibratedEventClassifier(tau=0.5)

    # Synthetic frames of single quiet intruder (no 2-person wrestling)
    frames = []
    for f in range(16):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        # Only 1 person sneaking quietly
        cv2.rectangle(img, (50, 60), (90, 180), (255, 255, 255), -1)
        frames.append(img)

    # Base logits predict Burglary
    logits = np.zeros(len(CLASSES), dtype=np.float32)
    logits[CLASS_TO_IDX["Burglary"]] = 4.0
    logits[CLASS_TO_IDX["Fighting"]] = -2.0

    res = classifier.classify(frames=frames, logits=logits, threshold=0.50)
    assert res["raw_class"] == "Burglary"
    assert res["predicted_class"] == "Burglary"
    assert res["confidence"] >= 0.75


def test_05_road_accidents_and_vehicle_verification():
    classifier = CalibratedEventClassifier(tau=0.5)

    # Synthetic frames with a vehicle moving fast
    frames = []
    for f in range(16):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        # Simulated car box
        pos_x = int(50 + f * 10)
        cv2.rectangle(img, (pos_x, 100), (pos_x + 80, 160), (0, 0, 255), -1)
        frames.append(img)

    logits = np.zeros(len(CLASSES), dtype=np.float32)
    logits[CLASS_TO_IDX["RoadAccidents"]] = 3.0

    res = classifier.classify(frames=frames, logits=logits, threshold=0.50)
    assert res["predicted_class"] == "RoadAccidents"


def test_06_graceful_fallback_on_missing_frames():
    classifier = CalibratedEventClassifier(tau=0.5)

    logits = np.zeros(len(CLASSES), dtype=np.float32)
    logits[CLASS_TO_IDX["Vandalism"]] = 3.0

    # Classify without frames (pure logit calibration fallback)
    res = classifier.classify(frames=None, logits=logits, threshold=0.50)
    assert "predicted_class" in res
    assert "confidence" in res
    assert res["confidence"] > 0.0


def test_07_real_fighting_clip_inference():
    # If the real fighting clip exists locally, verify it predicts Fighting
    clip_path = "DCSASS Dataset/Fighting/Fighting007_x264.mp4/Fighting007_x264_1.mp4"
    if not os.path.exists(clip_path):
        pytest.skip("Dataset fighting clip not available in local workspace")

    from src.config import DEVICE, CHECKPOINTS_DIR
    from src.model import build_model
    from src.inference import extract_clip_frames

    model = build_model().to(DEVICE)
    ckpt = torch.load(f"{CHECKPOINTS_DIR}/best_model.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cap = cv2.VideoCapture(clip_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()

    clip_tensor = extract_clip_frames(frames).to(DEVICE)
    with torch.no_grad():
        logits = model(clip_tensor).squeeze(0).cpu().numpy()

    classifier = get_event_classifier()
    res = classifier.classify(frames=frames, logits=logits, threshold=0.50)

    # In the raw model, this clip was predicted as Robbery (45.4%).
    # The expert classifier must disambiguate and correctly classify as Fighting!
    assert res["raw_class"] == "Robbery"
    assert res["predicted_class"] == "Fighting"
    assert res["is_suspicious"] is True
    assert res["confidence"] >= 0.70
