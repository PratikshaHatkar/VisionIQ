import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from src.api.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String(100), nullable=False)
    location = Column(String(200), nullable=True)
    owner_id = Column(String(100), nullable=False, default="default_owner", index=True)
    stream_type = Column(String(50), nullable=False, default="rtsp")  # rtsp, webcam, test_video, synthetic
    stream_url = Column(String(500), nullable=False)
    is_monitored = Column(Boolean, nullable=False, default=False, index=True)
    status = Column(String(50), nullable=False, default="offline", index=True)  # offline, connecting, online, reconnecting, error
    error_message = Column(Text, nullable=True)
    fps = Column(Float, nullable=True)
    resolution = Column(String(50), nullable=True)
    confidence_threshold = Column(Float, nullable=False, default=0.50)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    incidents = relationship("LiveIncident", back_populates="camera", cascade="all, delete-orphan", order_by="LiveIncident.created_at.desc()")
    notifications = relationship("LiveNotification", back_populates="camera", cascade="all, delete-orphan")

class LiveIncident(Base):
    __tablename__ = "live_incidents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    camera_id = Column(String(36), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    owner_id = Column(String(100), nullable=False, index=True)
    activity_label = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    start_time = Column(DateTime, default=utc_now, nullable=False)
    end_time = Column(DateTime, default=utc_now, nullable=False)
    duration_seconds = Column(Float, nullable=False, default=0.0)
    summary = Column(Text, nullable=True)
    evidence_snapshot = Column(String(500), nullable=True)
    evidence_clip = Column(String(500), nullable=True)
    is_acknowledged = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    camera = relationship("Camera", back_populates="incidents")
    notification = relationship("LiveNotification", back_populates="incident", uselist=False, cascade="all, delete-orphan")

class LiveNotification(Base):
    __tablename__ = "live_notifications"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    camera_id = Column(String(36), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    incident_id = Column(String(36), ForeignKey("live_incidents.id", ondelete="CASCADE"), nullable=False, index=True)
    owner_id = Column(String(100), nullable=False, index=True)
    activity_label = Column(String(50), nullable=False)
    camera_name = Column(String(100), nullable=False)
    location = Column(String(200), nullable=True)
    confidence = Column(Float, nullable=False)
    summary = Column(Text, nullable=True)
    evidence_snapshot = Column(String(500), nullable=True)
    is_read = Column(Boolean, nullable=False, default=False, index=True)
    is_acknowledged = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    camera = relationship("Camera", back_populates="notifications")
    incident = relationship("LiveIncident", back_populates="notification")
