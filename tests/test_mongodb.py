import os
import sys
import tempfile
import cv2
import numpy as np
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
import pymongo
from pymongo.errors import DuplicateKeyError

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MONGODB_DATABASE, MONGODB_URI, CLASSES
from src.api.mongo import get_mongo_db, get_mongo_client, check_mongo_connection, init_mongo_indexes
from src.api.storage import StorageRepository, get_storage
from src.api.main import app
from src.api.service import ModelManager, process_video_background

client = TestClient(app)

def create_synthetic_mp4(filepath: str, num_frames: int = 30, fps: int = 30, width: int = 320, height: int = 240):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + (i * 8) % 280, 120), 25, (0, 0, 255), -1)
        out.write(frame)
    out.release()

@pytest.fixture(scope="module", autouse=True)
def setup_mongo_test_environment():
    """
    Initializes storage, indexes, and loads model for tests.
    """
    repo = get_storage()
    repo.initialize()
    ModelManager.get_instance().load_model()
    yield

def test_01_mongo_connection_and_indexes():
    """
    Verifies that MongoDB is reachable, database is initialized, and unique indexes exist.
    """
    connected, info = check_mongo_connection()
    assert connected is True, f"MongoDB connection check failed: {info}"
    assert "MongoDB v" in info

    db = get_mongo_db()
    assert db is not None
    init_mongo_indexes(db)

    # Check videos collection indexes
    v_indexes = db["videos"].index_information()
    assert "idx_video_id_unique" in v_indexes
    assert v_indexes["idx_video_id_unique"]["unique"] is True

    # Check events collection indexes
    e_indexes = db["events"].index_information()
    assert "idx_event_id_unique" in e_indexes
    assert e_indexes["idx_event_id_unique"]["unique"] is True
    assert "idx_event_video_id" in e_indexes
    assert "idx_activity_label" in e_indexes
    assert "idx_start_timestamp" in e_indexes
    assert "idx_confidence" in e_indexes

def test_02_video_document_crud_in_mongodb():
    """
    Tests creating, fetching, and updating video documents in MongoDB via StorageRepository.
    """
    repo = get_storage()
    db = get_mongo_db()
    test_vid = f"test-vid-{datetime.now().timestamp()}"

    video_data = {
        "video_id": test_vid,
        "filename": "camera1_entrance.mp4",
        "file_path": "storage/videos/camera1_entrance.mp4",
        "file_size": 1048576,
        "status": "uploaded",
        "duration": 15.5,
        "fps": 30.0,
        "total_frames": 465,
        "width": 1920,
        "height": 1080,
        "resolution": "1920x1080"
    }

    # 1. Create
    created = repo.create_video(video_data)
    assert created["video_id"] == test_vid
    assert created["status"] == "uploaded"

    # Verify directly in MongoDB
    mongo_doc = db["videos"].find_one({"video_id": test_vid})
    assert mongo_doc is not None
    assert mongo_doc["filename"] == "camera1_entrance.mp4"
    assert mongo_doc["duration"] == 15.5
    assert mongo_doc["resolution"] == "1920x1080"

    # 2. Retrieve
    retrieved = repo.get_video(test_vid)
    assert retrieved is not None
    assert retrieved["video_id"] == test_vid
    assert retrieved["file_size"] == 1048576

    # 3. Update status
    updated = repo.update_video_status(test_vid, status="completed", error_message=None)
    assert updated["status"] == "completed"

    mongo_doc_updated = db["videos"].find_one({"video_id": test_vid})
    assert mongo_doc_updated["status"] == "completed"

    # Clean up test video doc
    db["videos"].delete_one({"video_id": test_vid})

def test_03_events_persistence_and_retrieval_in_mongodb():
    """
    Tests saving detected events into MongoDB and retrieving them sorted by start_timestamp.
    """
    repo = get_storage()
    db = get_mongo_db()
    test_vid = f"event-test-vid-{datetime.now().timestamp()}"

    events = [
        {
            "event_id": f"ev-1-{test_vid}",
            "video_id": test_vid,
            "activity_label": "Fighting",
            "confidence": 0.8950,
            "start_timestamp": 5.20,
            "end_timestamp": 8.40,
            "start_time_formatted": "00:00:05",
            "end_time_formatted": "00:00:08",
            "evidence_snapshot": "/api/evidence/ev-1.jpg",
            "evidence_clip": None,
            "processing_status": "completed",
            "created_at": datetime.now(timezone.utc)
        },
        {
            "event_id": f"ev-2-{test_vid}",
            "video_id": test_vid,
            "activity_label": "Robbery",
            "confidence": 0.9420,
            "start_timestamp": 1.00,
            "end_timestamp": 3.50,
            "start_time_formatted": "00:00:01",
            "end_time_formatted": "00:00:03",
            "evidence_snapshot": "/api/evidence/ev-2.jpg",
            "evidence_clip": None,
            "processing_status": "completed",
            "created_at": datetime.now(timezone.utc)
        }
    ]

    saved_count = repo.save_events(test_vid, events)
    assert saved_count == 2

    # Check MongoDB collection directly
    mongo_events = list(db["events"].find({"video_id": test_vid}))
    assert len(mongo_events) == 2

    # Retrieve via repo (should be sorted by start_timestamp: ev-2 then ev-1)
    retrieved = repo.get_video_events(test_vid)
    assert len(retrieved) == 2
    assert retrieved[0]["event_id"] == f"ev-2-{test_vid}"
    assert retrieved[0]["start_timestamp"] == 1.00
    assert retrieved[1]["event_id"] == f"ev-1-{test_vid}"
    assert retrieved[1]["start_timestamp"] == 5.20

    # Clean up
    db["events"].delete_many({"video_id": test_vid})

def test_04_duplicate_prevention_and_idempotent_reprocessing():
    """
    Verifies that re-processing a video replaces old events and never accumulates duplicates.
    Also verifies unique index on event_id.
    """
    repo = get_storage()
    db = get_mongo_db()
    test_vid = f"reproc-vid-{datetime.now().timestamp()}"

    # First pass: 2 events
    initial_events = [
        {
            "event_id": f"initial-1-{test_vid}",
            "video_id": test_vid,
            "activity_label": "Fighting",
            "confidence": 0.75,
            "start_timestamp": 2.0,
            "end_timestamp": 4.0,
            "start_time_formatted": "00:00:02",
            "end_time_formatted": "00:00:04",
            "created_at": datetime.now(timezone.utc)
        },
        {
            "event_id": f"initial-2-{test_vid}",
            "video_id": test_vid,
            "activity_label": "Burglary",
            "confidence": 0.82,
            "start_timestamp": 5.0,
            "end_timestamp": 7.0,
            "start_time_formatted": "00:00:05",
            "end_time_formatted": "00:00:07",
            "created_at": datetime.now(timezone.utc)
        }
    ]
    repo.save_events(test_vid, initial_events)
    assert repo.get_video_events_count(test_vid) == 2

    # Second pass (re-processing): 1 new updated event
    reprocessed_events = [
        {
            "event_id": f"new-1-{test_vid}",
            "video_id": test_vid,
            "activity_label": "Vandalism",
            "confidence": 0.91,
            "start_timestamp": 3.0,
            "end_timestamp": 6.0,
            "start_time_formatted": "00:00:03",
            "end_time_formatted": "00:00:06",
            "created_at": datetime.now(timezone.utc)
        }
    ]
    repo.save_events(test_vid, reprocessed_events)

    # Count must be exactly 1, NOT 3!
    assert repo.get_video_events_count(test_vid) == 1
    current = repo.get_video_events(test_vid)
    assert len(current) == 1
    assert current[0]["activity_label"] == "Vandalism"

    # Enforce unique index check: inserting identical event_id must fail
    with pytest.raises(DuplicateKeyError):
        db["events"].insert_one({
            "event_id": f"new-1-{test_vid}",
            "video_id": test_vid,
            "activity_label": "Duplicate"
        })

    # Clean up
    db["events"].delete_many({"video_id": test_vid})

def test_05_cross_video_filtering_in_mongodb():
    """
    Tests MongoDB-powered querying by activity_label, video_id, and min_confidence.
    """
    repo = get_storage()
    db = get_mongo_db()
    vid_a = f"filter-a-{datetime.now().timestamp()}"
    vid_b = f"filter-b-{datetime.now().timestamp()}"

    events = [
        {
            "event_id": f"ev-a1-{vid_a}",
            "video_id": vid_a,
            "activity_label": "Fighting",
            "confidence": 0.95,
            "start_timestamp": 1.0,
            "end_timestamp": 2.0,
            "start_time_formatted": "00:00:01",
            "end_time_formatted": "00:00:02",
            "created_at": datetime.now(timezone.utc)
        },
        {
            "event_id": f"ev-a2-{vid_a}",
            "video_id": vid_a,
            "activity_label": "Shoplifting",
            "confidence": 0.60,
            "start_timestamp": 3.0,
            "end_timestamp": 4.0,
            "start_time_formatted": "00:00:03",
            "end_time_formatted": "00:00:04",
            "created_at": datetime.now(timezone.utc)
        },
        {
            "event_id": f"ev-b1-{vid_b}",
            "video_id": vid_b,
            "activity_label": "Fighting",
            "confidence": 0.70,
            "start_timestamp": 1.0,
            "end_timestamp": 3.0,
            "start_time_formatted": "00:00:01",
            "end_time_formatted": "00:00:03",
            "created_at": datetime.now(timezone.utc)
        }
    ]

    for ev in events:
        db["events"].insert_one(ev)

    try:
        # Filter by activity
        total, res = repo.list_events(activity="Fighting")
        matching = [e for e in res if e["video_id"] in (vid_a, vid_b)]
        assert len(matching) == 2
        assert all(e["activity_label"] == "Fighting" for e in matching)

        # Filter by video_id
        total_a, res_a = repo.list_events(video_id=vid_a)
        assert len(res_a) == 2

        # Filter by min_confidence
        total_conf, res_conf = repo.list_events(min_confidence=0.85)
        matching_conf = [e for e in res_conf if e["video_id"] in (vid_a, vid_b)]
        assert len(matching_conf) == 1
        assert matching_conf[0]["event_id"] == f"ev-a1-{vid_a}"

    finally:
        db["events"].delete_many({"video_id": {"$in": [vid_a, vid_b]}})

def test_06_health_endpoint_reports_mongodb():
    """
    GET /health returns 200 and confirms MongoDB engine and connectivity.
    """
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["database_engine"] == "mongodb"
    assert data["mongodb"]["connected"] is True
    assert "MongoDB v" in data["mongodb"]["details"]

def test_07_full_pipeline_with_mongodb_persistence():
    """
    Uploads a synthetic video, runs background processing, and verifies that
    video metadata and detected events are stored in MongoDB and accessible via API.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        create_synthetic_mp4(tmp_path, num_frames=60, fps=30)
        with open(tmp_path, "rb") as f:
            upload_res = client.post("/api/videos/upload", files={"file": ("cctv_mongo_test.mp4", f, "video/mp4")})

        assert upload_res.status_code == 201
        data = upload_res.json()
        video_id = data["video_id"]

        # Check video document exists in MongoDB
        db = get_mongo_db()
        doc = db["videos"].find_one({"video_id": video_id})
        assert doc is not None
        assert doc["filename"] == "cctv_mongo_test.mp4"
        assert doc["status"] == "uploaded"

        # Trigger processing via API
        proc_res = client.post(f"/api/videos/{video_id}/process?confidence_threshold=0.10&window_sec=1.0&stride_sec=0.5")
        assert proc_res.status_code == 200

        # Execute worker synchronously for test
        process_video_background(video_id=video_id, threshold=0.10, window_sec=1.0, stride_sec=0.5)

        # Check status endpoint
        status_res = client.get(f"/api/videos/{video_id}/status")
        assert status_res.status_code == 200
        assert status_res.json()["status"] == "completed"

        # Check events endpoint
        events_res = client.get(f"/api/videos/{video_id}/events")
        assert events_res.status_code == 200
        events = events_res.json()

        # If any suspicious events detected, verify schema and evidence
        for ev in events:
            assert ev["video_id"] == video_id
            assert ev["activity_label"] in CLASSES
            assert ev["activity_label"] != "Normal"
            assert ev["processing_status"] == "completed"
            assert ev["evidence_snapshot"].startswith("/api/evidence/")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_08_graceful_sqlite_fallback(monkeypatch):
    """
    Verifies that when MongoDB is unreachable, the system gracefully falls back to SQLite
    without throwing exceptions or failing requests.
    """
    # Simulate MongoDB offline by monkeypatching
    import src.api.storage as storage_mod
    monkeypatch.setattr(storage_mod, "get_mongo_db", lambda: None)
    monkeypatch.setattr(storage_mod, "check_mongo_connection", lambda: (False, "Connection simulated down"))

    repo = StorageRepository()
    status_info = repo.get_status()
    assert status_info["engine"] == "sqlite_fallback"
    assert status_info["status"] == "connected"

    test_vid = f"fallback-vid-{datetime.now().timestamp()}"
    v = repo.create_video({
        "video_id": test_vid,
        "filename": "fallback.mp4",
        "file_path": "fallback.mp4",
        "status": "uploaded"
    })
    assert v["video_id"] == test_vid

    fetched = repo.get_video(test_vid)
    assert fetched is not None
    assert fetched["video_id"] == test_vid

    repo.save_events(test_vid, [{
        "event_id": f"fb-ev-{test_vid}",
        "video_id": test_vid,
        "activity_label": "Fighting",
        "confidence": 0.88,
        "start_timestamp": 1.0,
        "end_timestamp": 2.0,
        "start_time_formatted": "00:00:01",
        "end_time_formatted": "00:00:02",
        "created_at": datetime.now(timezone.utc)
    }])

    evs = repo.get_video_events(test_vid)
    assert len(evs) == 1
    assert evs[0]["activity_label"] == "Fighting"
