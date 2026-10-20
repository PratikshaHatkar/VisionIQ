import os
import sys
import tempfile
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.main import app
from src.api.database import init_db, SessionLocal
from src.api.models import Video, Event
from src.api.service import ModelManager, process_video_background
from src.config import CLASSES, EVIDENCE_DIR, VIDEOS_DIR

client = TestClient(app)

def create_synthetic_mp4(filepath: str, num_frames: int = 30, fps: int = 30, width: int = 320, height: int = 240):
    """
    Creates a valid synthetic MP4 video file for testing.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    for i in range(num_frames):
        # Draw dynamic motion (moving circle)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + (i * 8) % 280, 120), 25, (0, 0, 255), -1)
        out.write(frame)
    out.release()

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    """
    Initializes DB and preloads model for backend testing.
    """
    init_db()
    ModelManager.get_instance().load_model()
    yield

def test_01_health_endpoint():
    """
    GET /health returns 200, healthy status, database connected, and 7 classes.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["model"]["loaded"] is True
    assert data["model"]["classes"] == CLASSES
    assert len(data["model"]["classes"]) == 7

def test_02_upload_invalid_file_extension():
    """
    POST /api/videos/upload rejects non-video extensions (.txt, .exe, etc.).
    """
    files = {"file": ("test_doc.txt", b"dummy content", "text/plain")}
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

def test_03_upload_empty_video_file():
    """
    POST /api/videos/upload rejects 0-byte video files.
    """
    files = {"file": ("empty.mp4", b"", "video/mp4")}
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_04_upload_corrupted_video_file():
    """
    POST /api/videos/upload rejects corrupted unreadable video payloads.
    """
    files = {"file": ("corrupt.mp4", b"NOT_A_REAL_VIDEO_HEADER_DATA_12345", "video/mp4")}
    response = client.post("/api/videos/upload", files=files)
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower()

def test_05_upload_valid_video():
    """
    POST /api/videos/upload successfully stores a valid MP4 and returns metadata.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        create_synthetic_mp4(tmp_path, num_frames=30, fps=30)
        with open(tmp_path, "rb") as f:
            response = client.post("/api/videos/upload", files={"file": ("test_synthetic.mp4", f, "video/mp4")})

        assert response.status_code == 201
        data = response.json()
        assert "video_id" in data
        assert data["filename"] == "test_synthetic.mp4"
        assert data["status"] == "uploaded"
        assert data["duration"] == 1.0
        assert data["fps"] == 30.0
        assert data["resolution"] == "320x240"

        # Check status endpoint immediately after upload
        status_res = client.get(f"/api/videos/{data['video_id']}/status")
        assert status_res.status_code == 200
        assert status_res.json()["status"] == "uploaded"
        assert status_res.json()["events_count"] == 0

        # Check events endpoint before processing (should be empty list)
        events_res = client.get(f"/api/videos/{data['video_id']}/events")
        assert events_res.status_code == 200
        assert events_res.json() == []

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_06_process_missing_video_id():
    """
    POST /api/videos/{invalid_id}/process returns 404 Not Found.
    """
    response = client.post("/api/videos/non-existent-uuid-1234/process")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_07_status_missing_video_id():
    """
    GET /api/videos/{invalid_id}/status returns 404 Not Found.
    """
    response = client.get("/api/videos/non-existent-uuid-1234/status")
    assert response.status_code == 404

def test_08_events_missing_video_id():
    """
    GET /api/videos/{invalid_id}/events returns 404 Not Found.
    """
    response = client.get("/api/videos/non-existent-uuid-1234/events")
    assert response.status_code == 404

def test_09_full_processing_and_event_generation():
    """
    Uploads a video, triggers processing, verifies completion, schema validation, and persistence.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        create_synthetic_mp4(tmp_path, num_frames=60, fps=30)
        with open(tmp_path, "rb") as f:
            upload_res = client.post("/api/videos/upload", files={"file": ("surveillance_sample.mp4", f, "video/mp4")})
        video_id = upload_res.json()["video_id"]

        # Trigger processing via API
        proc_res = client.post(f"/api/videos/{video_id}/process?confidence_threshold=0.10&window_sec=1.0&stride_sec=0.5")
        assert proc_res.status_code == 200
        assert proc_res.json()["status"] == "processing"

        # Run worker synchronously for test determinism
        process_video_background(video_id=video_id, threshold=0.10, window_sec=1.0, stride_sec=0.5)

        # Check status
        status_res = client.get(f"/api/videos/{video_id}/status")
        assert status_res.status_code == 200
        s_data = status_res.json()
        assert s_data["status"] == "completed"

        # Check events
        events_res = client.get(f"/api/videos/{video_id}/events")
        assert events_res.status_code == 200
        events = events_res.json()

        # Validate event schema if events detected
        for ev in events:
            assert "event_id" in ev
            assert ev["video_id"] == video_id
            assert ev["activity_label"] in CLASSES
            assert ev["activity_label"] != "Normal"  # Normal events must not be flagged as alerts
            assert 0.0 <= ev["confidence"] <= 1.0
            assert ev["start_timestamp"] <= ev["end_timestamp"]
            assert len(ev["start_time_formatted"].split(":")) == 3  # HH:MM:SS
            assert len(ev["end_time_formatted"].split(":")) == 3
            assert ev["processing_status"] == "completed"
            assert ev["evidence_snapshot"].startswith("/api/evidence/")

            # Test evidence snapshot retrieval endpoint
            snapshot_res = client.get(ev["evidence_snapshot"])
            assert snapshot_res.status_code == 200
            assert snapshot_res.headers["content-type"] == "image/jpeg"
            assert len(snapshot_res.content) > 0

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_10_evidence_endpoint_security():
    """
    GET /api/evidence/{filename} safely rejects missing files and path-traversal attempts.
    """
    # Missing snapshot returns 404
    res_404 = client.get("/api/evidence/non_existent_snapshot_999.jpg")
    assert res_404.status_code == 404

    # Path traversal attack attempt returns 404
    res_traversal = client.get("/api/evidence/../../src/config.py")
    assert res_traversal.status_code == 404

def test_11_cross_video_events_filtering():
    """
    GET /api/events correctly filters events by activity, video_id, and min_confidence.
    """
    import uuid
    uid = uuid.uuid4().hex[:8]
    test_vid_id = f"filter-test-{uid}"
    ev1_id = f"ev-1-{uid}"
    ev2_id = f"ev-2-{uid}"

    db = SessionLocal()
    vid = Video(id=test_vid_id, filename="dummy.mp4", file_path="dummy.mp4", status="completed")
    db.add(vid)
    ev1 = Event(
        id=ev1_id, video_id=test_vid_id, activity_label="Fighting",
        confidence=0.92, start_timestamp=2.0, end_timestamp=4.0,
        start_time_formatted="00:00:02", end_time_formatted="00:00:04",
        processing_status="completed"
    )
    ev2 = Event(
        id=ev2_id, video_id=test_vid_id, activity_label="Robbery",
        confidence=0.65, start_timestamp=10.0, end_timestamp=15.0,
        start_time_formatted="00:00:10", end_time_formatted="00:00:15",
        processing_status="completed"
    )
    db.add(ev1)
    db.add(ev2)
    db.commit()
    db.close()

    # 1. Filter by activity and video_id
    res_act = client.get(f"/api/events?video_id={test_vid_id}&activity=Fighting")
    assert res_act.status_code == 200
    assert any(e["event_id"] == ev1_id for e in res_act.json()["events"])
    assert not any(e["event_id"] == ev2_id for e in res_act.json()["events"])

    # 2. Filter by min_confidence and video_id
    res_conf = client.get(f"/api/events?video_id={test_vid_id}&min_confidence=0.80")
    assert res_conf.status_code == 200
    assert any(e["event_id"] == ev1_id for e in res_conf.json()["events"])
    assert not any(e["event_id"] == ev2_id for e in res_conf.json()["events"])

    # 3. Filter by video_id
    res_vid = client.get(f"/api/events?video_id={test_vid_id}")
    assert res_vid.status_code == 200
    assert len(res_vid.json()["events"]) == 2
