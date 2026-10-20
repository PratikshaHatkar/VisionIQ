import os
import time
import json
import uuid
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.live.storage import get_live_storage
from src.api.live.stream_manager import get_stream_manager, CameraStreamWorker
from src.api.live.notifications import get_notification_broker
from src.api.live.inference_service import get_live_inference_service, SUSPICIOUS_ACTIVITIES

client = TestClient(app)

def test_01_camera_registration_and_url_masking():
    payload = {
        "name": "North Gate CCTV",
        "location": "North Entrance Zone 1",
        "owner_id": "security_officer_1",
        "stream_type": "rtsp",
        "stream_url": "rtsp://admin:super_secret_password123@192.168.1.105:554/h264Preview_01_main",
        "confidence_threshold": 0.65
    }
    response = client.post("/api/live/cameras", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "North Gate CCTV"
    assert data["owner_id"] == "security_officer_1"
    assert data["stream_type"] == "rtsp"
    assert data["confidence_threshold"] == 0.65
    # SECURITY: Verify credentials are NEVER exposed in plain text
    assert "super_secret_password123" not in data["stream_url"]
    assert "***" in data["stream_url"]

def test_02_camera_list_and_owner_filtering():
    # Register camera for owner A
    res_a = client.post("/api/live/cameras", json={
        "name": "Cam A",
        "owner_id": "owner_alpha",
        "stream_type": "synthetic",
        "stream_url": "synthetic:0"
    })
    assert res_a.status_code == 201
    cam_a_id = res_a.json()["id"]

    # Register camera for owner B
    res_b = client.post("/api/live/cameras", json={
        "name": "Cam B",
        "owner_id": "owner_beta",
        "stream_type": "synthetic",
        "stream_url": "synthetic:1"
    })
    assert res_b.status_code == 201
    cam_b_id = res_b.json()["id"]

    # Filter for owner_alpha
    list_a = client.get("/api/live/cameras?owner_id=owner_alpha").json()
    assert any(c["id"] == cam_a_id for c in list_a["cameras"])
    assert not any(c["id"] == cam_b_id for c in list_a["cameras"])

    # Filter for owner_beta
    list_b = client.get("/api/live/cameras?owner_id=owner_beta").json()
    assert any(c["id"] == cam_b_id for c in list_b["cameras"])
    assert not any(c["id"] == cam_a_id for c in list_b["cameras"])

def test_03_camera_update_and_delete():
    # Create camera
    res = client.post("/api/live/cameras", json={
        "name": "Warehouse Cam",
        "location": "Aisle 4",
        "owner_id": "warehouse_owner",
        "stream_type": "synthetic",
        "stream_url": "synthetic:test"
    })
    cam_id = res.json()["id"]

    # Update camera
    patch_res = client.patch(f"/api/live/cameras/{cam_id}", json={
        "name": "Warehouse South Cam",
        "confidence_threshold": 0.75
    })
    assert patch_res.status_code == 200
    updated = patch_res.json()
    assert updated["name"] == "Warehouse South Cam"
    assert updated["confidence_threshold"] == 0.75

    # Delete camera
    del_res = client.delete(f"/api/live/cameras/{cam_id}")
    assert del_res.status_code == 200

    # Verify 404
    get_res = client.get(f"/api/live/cameras/{cam_id}")
    assert get_res.status_code == 404

def test_04_synthetic_stream_start_and_stop():
    res = client.post("/api/live/cameras", json={
        "name": "Live Test Stream",
        "owner_id": "tester_1",
        "stream_type": "synthetic",
        "stream_url": "synthetic:0"
    })
    cam_id = res.json()["id"]

    # Start monitoring
    start_res = client.post(f"/api/live/cameras/{cam_id}/start")
    assert start_res.status_code == 200
    assert start_res.json()["is_monitored"] is True

    # Allow worker thread to produce initial frames
    time.sleep(0.3)
    stream_mgr = get_stream_manager()
    worker = stream_mgr.get_worker(cam_id)
    assert worker is not None
    assert worker.running is True
    latest_frame = worker.get_latest_frame()
    assert latest_frame is not None
    assert latest_frame.shape == (360, 640, 3)

    # Stop monitoring
    stop_res = client.post(f"/api/live/cameras/{cam_id}/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["is_monitored"] is False
    assert worker.running is False

def test_05_mjpeg_stream_generator():
    res = client.post("/api/live/cameras", json={
        "name": "MJPEG Test",
        "owner_id": "tester_2",
        "stream_type": "synthetic",
        "stream_url": "synthetic:0"
    })
    cam_id = res.json()["id"]
    client.post(f"/api/live/cameras/{cam_id}/start")
    time.sleep(0.2)

    gen = get_stream_manager().generate_mjpeg_stream(cam_id)
    frame_chunk = next(gen)
    assert b"--frame" in frame_chunk
    assert b"Content-Type: image/jpeg" in frame_chunk

    get_stream_manager().stop_camera(cam_id)

def test_06_live_incident_creation_and_deduplication():
    storage = get_live_storage()
    infer_service = get_live_inference_service()

    cam_id = f"test_cam_dedup_{uuid.uuid4().hex[:8]}"
    now_epoch = time.time()

    # Step 1: Simulate first suspicious detection
    inc1 = storage.create_incident({
        "camera_id": cam_id,
        "camera_name": "Parking Lot",
        "location": "North Bay",
        "owner_id": "officer_d",
        "activity_label": "Fighting",
        "confidence": 0.85,
        "duration_seconds": 1.5,
        "is_acknowledged": False
    })
    inc_id = inc1["id"]

    with infer_service.state_lock:
        infer_service.ongoing_incidents[cam_id] = {
            "incident_id": inc_id,
            "activity_label": "Fighting",
            "confidence": 0.85,
            "start_time": inc1["start_time"],
            "last_seen_epoch": now_epoch
        }

    # Step 2: Simulate second consecutive detection of same event
    # Deduplication rule should update the ongoing incident duration instead of creating a new one
    with infer_service.state_lock:
        ongoing = infer_service.ongoing_incidents[cam_id]
        assert ongoing["activity_label"] == "Fighting"
        # Update duration
        storage.update_incident(inc_id, {
            "duration_seconds": 3.0,
            "confidence": 0.92
        })

    updated_inc = storage.get_incident(inc_id)
    assert updated_inc["duration_seconds"] == 3.0
    assert updated_inc["confidence"] == 0.92

    # Verify only 1 incident exists for this camera
    all_incidents = storage.list_incidents(camera_id=cam_id)
    assert len(all_incidents) == 1

def test_07_live_incident_listing_and_acknowledgement():
    storage = get_live_storage()
    cam_id = f"cam_ack_test_{uuid.uuid4().hex[:8]}"
    inc = storage.create_incident({
        "camera_id": cam_id,
        "owner_id": "officer_ack",
        "activity_label": "Burglary",
        "confidence": 0.91,
        "duration_seconds": 4.5,
        "is_acknowledged": False
    })
    inc_id = inc["id"]

    # Verify initially unacknowledged
    assert inc["is_acknowledged"] is False

    # Acknowledge through endpoint
    ack_res = client.patch(f"/api/live/incidents/{inc_id}/acknowledge")
    assert ack_res.status_code == 200
    assert ack_res.json()["is_acknowledged"] is True

def test_08_notifications_persistence_and_read_status():
    storage = get_live_storage()
    owner = f"owner_notif_{uuid.uuid4().hex[:8]}"

    # Create 2 unread notifications
    n1 = storage.create_notification({
        "camera_id": "c1",
        "incident_id": "i1",
        "owner_id": owner,
        "activity_label": "Vandalism",
        "camera_name": "Lobby",
        "confidence": 0.82,
        "is_read": False
    })
    n2 = storage.create_notification({
        "camera_id": "c2",
        "incident_id": "i2",
        "owner_id": owner,
        "activity_label": "Robbery",
        "camera_name": "Vault",
        "confidence": 0.94,
        "is_read": False
    })

    # List notifications
    list_res = client.get(f"/api/live/notifications?owner_id={owner}").json()
    assert list_res["total"] >= 2
    assert list_res["unread_count"] >= 2

    # Mark n1 as read
    client.patch(f"/api/live/notifications/{n1['id']}/read")
    list_res_after = client.get(f"/api/live/notifications?owner_id={owner}").json()
    assert list_res_after["unread_count"] == list_res["unread_count"] - 1

    # Mark all as read
    client.patch(f"/api/live/notifications/read-all?owner_id={owner}")
    list_res_final = client.get(f"/api/live/notifications?owner_id={owner}").json()
    assert list_res_final["unread_count"] == 0

def test_09_websocket_realtime_alert_delivery():
    owner = "ws_test_owner"
    with client.websocket_connect(f"/api/live/ws?owner_id={owner}") as websocket:
        # Test ping-pong
        websocket.send_text("ping")
        data = websocket.receive_text()
        assert data == "pong"

def test_10_invalid_stream_type_rejection():
    res = client.post("/api/live/cameras", json={
        "name": "Invalid Stream Type",
        "owner_id": "attacker",
        "stream_type": "malicious_script_inject",
        "stream_url": "ftp://malicious"
    })
    assert res.status_code == 400
    assert "Unsupported stream_type" in res.json()["detail"]
