from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse

from src.api.live.schemas import (
    CameraCreate, CameraUpdate, CameraResponse, CamerasListResponse,
    MonitoringControlResponse, LiveIncidentResponse, LiveIncidentsListResponse,
    LiveNotificationResponse, LiveNotificationsListResponse, mask_stream_url
)
from src.api.live.storage import get_live_storage
from src.api.live.stream_manager import get_stream_manager
from src.api.live.notifications import get_notification_broker
from src.api.live.inference_service import get_live_inference_service

live_router = APIRouter()

def _to_camera_response(cam: dict) -> CameraResponse:
    cam_copy = dict(cam)
    cam_copy["stream_url"] = mask_stream_url(cam_copy.get("stream_url", ""))
    return CameraResponse(**cam_copy)

# -------------------------------------------------------------
# 1. Camera Management Endpoints
# -------------------------------------------------------------
@live_router.post("/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def register_camera(payload: CameraCreate):
    """
    Registers a new CCTV camera (RTSP stream, webcam device index, or simulated test feed).
    """
    # Basic URL/stream validation
    stream_type = payload.stream_type.lower()
    if stream_type not in ("rtsp", "webcam", "test_video", "synthetic"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported stream_type. Supported values: 'rtsp', 'webcam', 'test_video', 'synthetic'."
        )

    storage = get_live_storage()
    data = payload.model_dump()
    data["is_monitored"] = False
    data["status"] = "offline"

    created = storage.create_camera(data)
    return _to_camera_response(created)

@live_router.get("/cameras", response_model=CamerasListResponse)
def list_cameras(owner_id: Optional[str] = Query(None, description="Filter cameras by owner ID")):
    """
    Lists registered cameras for the given owner (or all if not specified).
    Stream credentials in RTSP URLs are strictly masked.
    """
    storage = get_live_storage()
    cameras = storage.list_cameras(owner_id=owner_id)
    return CamerasListResponse(
        total=len(cameras),
        cameras=[_to_camera_response(c) for c in cameras]
    )

@live_router.get("/cameras/{camera_id}", response_model=CameraResponse)
def get_camera_detail(camera_id: str):
    """
    Retrieves details for a single registered camera.
    """
    storage = get_live_storage()
    cam = storage.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return _to_camera_response(cam)

@live_router.patch("/cameras/{camera_id}", response_model=CameraResponse)
def update_camera(camera_id: str, payload: CameraUpdate):
    """
    Updates configuration for a registered camera.
    """
    storage = get_live_storage()
    cam = storage.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found.")

    updates = {k: v for k, v in payload.model_dump(exclude_unset=True).items() if v is not None}
    if not updates:
        return _to_camera_response(cam)

    updated = storage.update_camera(camera_id, updates)
    return _to_camera_response(updated)

@live_router.delete("/cameras/{camera_id}")
def delete_camera(camera_id: str):
    """
    Stops monitoring and deletes a registered camera and its associated records.
    """
    storage = get_live_storage()
    cam = storage.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found.")

    # Stop live worker if running
    get_stream_manager().stop_camera(camera_id)
    storage.delete_camera(camera_id)
    return {"message": f"Camera {camera_id} successfully deleted."}

# -------------------------------------------------------------
# 2. Live Monitoring Controls & Stream Preview
# -------------------------------------------------------------
@live_router.post("/cameras/{camera_id}/start", response_model=MonitoringControlResponse)
def start_camera_monitoring(camera_id: str):
    """
    Starts live monitoring worker thread for the specified camera.
    """
    storage = get_live_storage()
    cam = storage.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found.")

    stream_mgr = get_stream_manager()
    success, msg = stream_mgr.start_camera(camera_id, cam)

    # Ensure inference service is running
    get_live_inference_service().start()

    return MonitoringControlResponse(
        camera_id=camera_id,
        is_monitored=True,
        status="connecting",
        message=msg
    )

@live_router.post("/cameras/{camera_id}/stop", response_model=MonitoringControlResponse)
def stop_camera_monitoring(camera_id: str):
    """
    Stops live monitoring and releases camera capture resources.
    """
    storage = get_live_storage()
    cam = storage.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found.")

    stream_mgr = get_stream_manager()
    success, msg = stream_mgr.stop_camera(camera_id)

    return MonitoringControlResponse(
        camera_id=camera_id,
        is_monitored=False,
        status="offline",
        message=msg
    )

@live_router.get("/cameras/{camera_id}/stream")
def get_camera_live_stream(camera_id: str):
    """
    Streams multipart MJPEG video directly to HTML <img> tags or video components in the browser.
    """
    storage = get_live_storage()
    cam = storage.get_camera(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found.")

    stream_mgr = get_stream_manager()
    return StreamingResponse(
        stream_mgr.generate_mjpeg_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# -------------------------------------------------------------
# 3. Live Incidents History & Acknowledgement
# -------------------------------------------------------------
@live_router.get("/incidents", response_model=LiveIncidentsListResponse)
def list_live_incidents(
    owner_id: Optional[str] = Query(None, description="Filter incidents by owner ID"),
    camera_id: Optional[str] = Query(None, description="Filter incidents by camera ID"),
    activity_label: Optional[str] = Query(None, description="Filter by suspicious activity"),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Retrieves incident logs generated from live camera monitoring.
    """
    storage = get_live_storage()
    incidents = storage.list_incidents(
        owner_id=owner_id,
        camera_id=camera_id,
        activity_label=activity_label,
        limit=limit
    )
    return LiveIncidentsListResponse(
        total=len(incidents),
        incidents=[LiveIncidentResponse(**inc) for inc in incidents]
    )

@live_router.get("/incidents/{incident_id}", response_model=LiveIncidentResponse)
def get_live_incident_detail(incident_id: str):
    """
    Retrieves details for a single live incident.
    """
    storage = get_live_storage()
    inc = storage.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="Live incident not found.")
    return LiveIncidentResponse(**inc)

@live_router.patch("/incidents/{incident_id}/acknowledge", response_model=LiveIncidentResponse)
def acknowledge_live_incident(incident_id: str, owner_id: Optional[str] = Query(None)):
    """
    Marks an ongoing or past live incident as acknowledged by security personnel.
    """
    storage = get_live_storage()
    inc = storage.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="Live incident not found.")

    storage.acknowledge_incident(incident_id, owner_id=owner_id)
    updated = storage.get_incident(incident_id)
    return LiveIncidentResponse(**updated)

# -------------------------------------------------------------
# 4. Real-time Notifications & WebSocket
# -------------------------------------------------------------
@live_router.get("/notifications", response_model=LiveNotificationsListResponse)
def list_notifications(
    owner_id: Optional[str] = Query(None, description="Filter notifications by owner ID"),
    is_read: Optional[bool] = Query(None, description="Filter by read/unread state"),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Lists persisted notifications for the specified camera owner.
    """
    storage = get_live_storage()
    total, unread, notifs = storage.list_notifications(
        owner_id=owner_id,
        is_read=is_read,
        limit=limit
    )
    return LiveNotificationsListResponse(
        total=total,
        unread_count=unread,
        notifications=[LiveNotificationResponse(**n) for n in notifs]
    )

@live_router.patch("/notifications/{notification_id}/read")
def mark_notification_read(notification_id: str, owner_id: Optional[str] = Query(None)):
    """
    Marks a single notification as read.
    """
    storage = get_live_storage()
    storage.mark_notification_read(notification_id, owner_id=owner_id)
    return {"message": f"Notification {notification_id} marked as read."}

@live_router.patch("/notifications/read-all")
def mark_all_notifications_read(owner_id: str = Query(..., description="Owner ID to clear notifications")):
    """
    Marks all notifications for an owner as read.
    """
    storage = get_live_storage()
    storage.mark_all_notifications_read(owner_id=owner_id)
    return {"message": f"All notifications for owner {owner_id} marked as read."}

@live_router.websocket("/ws")
async def websocket_notifications_endpoint(
    websocket: WebSocket,
    owner_id: str = Query("default_owner")
):
    """
    WebSocket endpoint for delivering instant in-dashboard alert pushes when an incident is detected.
    Clients receive structured JSON containing the activity type, camera name, location, and snapshot link.
    """
    broker = get_notification_broker()
    await broker.connect(owner_id, websocket)
    try:
        while True:
            # Keep connection alive; clients can send lightweight pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        broker.disconnect(owner_id, websocket)
    except Exception as e:
        broker.disconnect(owner_id, websocket)
