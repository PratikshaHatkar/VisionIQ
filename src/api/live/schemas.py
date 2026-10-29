import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

def mask_stream_url(url: Optional[str]) -> Optional[str]:
    """
    Sanitizes RTSP/HTTP/credentialed URLs by masking sensitive passwords.
    Example: 'rtsp://admin:mypass123@192.168.1.50:554/live' -> 'rtsp://admin:***@192.168.1.50:554/live'
    """
    if not url:
        return url
    # Pattern matching protocol://user:pass@host
    return re.sub(r'(://[^:]+:)([^@]+)(@)', r'\1***\3', url)

class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Friendly camera name")
    location: Optional[str] = Field(None, max_length=200, description="Physical location or zone")
    owner_id: str = Field(default="default_owner", min_length=1, max_length=100, description="Owner identifier")
    stream_type: str = Field(default="rtsp", description="Stream type: 'rtsp', 'webcam', 'test_video', or 'synthetic'")
    stream_url: str = Field(..., min_length=1, max_length=500, description="RTSP URL, device index (e.g. 0), or test video path")
    confidence_threshold: float = Field(default=0.50, ge=0.10, le=0.99, description="Confidence threshold for alerting")

class CameraUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    location: Optional[str] = Field(None, max_length=200)
    stream_type: Optional[str] = None
    stream_url: Optional[str] = Field(None, min_length=1, max_length=500)
    confidence_threshold: Optional[float] = Field(None, ge=0.10, le=0.99)

class CameraResponse(BaseModel):
    id: str
    name: str
    location: Optional[str] = None
    owner_id: str
    stream_type: str
    stream_url: str
    is_monitored: bool
    status: str
    error_message: Optional[str] = None
    fps: Optional[float] = None
    resolution: Optional[str] = None
    confidence_threshold: float
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class CamerasListResponse(BaseModel):
    total: int
    cameras: List[CameraResponse]

class MonitoringControlResponse(BaseModel):
    camera_id: str
    is_monitored: bool
    status: str
    message: str

class LiveIncidentResponse(BaseModel):
    id: str
    camera_id: str
    owner_id: str
    camera_name: Optional[str] = None
    location: Optional[str] = None
    activity_label: str
    confidence: float
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    summary: Optional[str] = None
    evidence_snapshot: Optional[str] = None
    evidence_clip: Optional[str] = None
    is_acknowledged: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class LiveIncidentsListResponse(BaseModel):
    total: int
    incidents: List[LiveIncidentResponse]

class LiveNotificationResponse(BaseModel):
    id: str
    camera_id: str
    incident_id: str
    owner_id: str
    activity_label: str
    camera_name: str
    location: Optional[str] = None
    confidence: float
    summary: Optional[str] = None
    evidence_snapshot: Optional[str] = None
    is_read: bool
    is_acknowledged: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class LiveNotificationsListResponse(BaseModel):
    total: int
    unread_count: int
    notifications: List[LiveNotificationResponse]
