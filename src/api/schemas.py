from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field

class HealthResponse(BaseModel):
    status: str
    api: str
    version: str
    model: Dict[str, Any]
    database: str
    database_engine: Optional[str] = None
    database_details: Optional[str] = None
    mongodb: Optional[Dict[str, Any]] = None

class VideoUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    video_id: str
    filename: str
    status: str
    file_size: int
    duration: Optional[float] = None
    fps: Optional[float] = None
    resolution: Optional[str] = None
    created_at: datetime

class VideoProcessResponse(BaseModel):
    video_id: str
    status: str
    message: str

class VideoStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    video_id: str
    filename: str
    status: str
    events_count: int = 0
    duration: Optional[float] = None
    fps: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class EventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    video_id: str
    activity_label: str
    confidence: float
    start_timestamp: float
    end_timestamp: float
    start_time_formatted: str
    end_time_formatted: str
    evidence_snapshot: Optional[str] = None
    evidence_clip: Optional[str] = None
    processing_status: str
    created_at: datetime

class EventsListResponse(BaseModel):
    total: int
    events: List[EventResponse]

class VideosListResponse(BaseModel):
    total: int
    videos: List[VideoStatusResponse]

class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search or question")
    video_id: Optional[str] = Field(None, description="Optional target video ID filter")
    activity: Optional[str] = Field(None, description="Optional activity class filter")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional minimum confidence threshold")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of events to return")

class ChatEventRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    video_id: str
    activity_label: str
    confidence: float
    start_timestamp: float
    end_timestamp: float
    start_time_formatted: str
    end_time_formatted: str
    evidence_snapshot: Optional[str] = None
    filename: Optional[str] = None
    relevance_score: Optional[float] = None

class ChatQueryResponse(BaseModel):
    query: str
    answer: str
    applied_filters: Dict[str, Any]
    events: List[ChatEventRecord]
    total: int
    retrieval_method: str
    limitations: List[str]
