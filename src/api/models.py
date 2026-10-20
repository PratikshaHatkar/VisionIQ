import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from src.api.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class Video(Base):
    __tablename__ = "videos"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False, default=0)
    status = Column(String(50), nullable=False, default="uploaded", index=True)  # uploaded, processing, completed, failed
    duration = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    events = relationship("Event", back_populates="video", cascade="all, delete-orphan", order_by="Event.start_timestamp")

class Event(Base):
    __tablename__ = "events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    activity_label = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    start_timestamp = Column(Float, nullable=False)
    end_timestamp = Column(Float, nullable=False)
    start_time_formatted = Column(String(20), nullable=False)
    end_time_formatted = Column(String(20), nullable=False)
    evidence_snapshot = Column(String(500), nullable=True)
    evidence_clip = Column(String(500), nullable=True)
    processing_status = Column(String(50), default="completed", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    video = relationship("Video", back_populates="events")
