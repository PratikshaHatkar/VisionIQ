import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, BackgroundTasks, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.config import (
    ALLOWED_EXTENSIONS, VIDEOS_DIR, EVIDENCE_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_WINDOW_SEC, DEFAULT_STRIDE_SEC
)
from src.api.database import get_db
from src.api.storage import get_storage
from src.api.schemas import (
    HealthResponse, VideoUploadResponse, VideoProcessResponse,
    VideoStatusResponse, VideosListResponse, EventResponse, EventsListResponse,
    ChatQueryRequest, ChatQueryResponse, ChatEventRecord
)
from src.api.service import (
    ModelManager, probe_video_file, process_video_background
)
from src.api.nlp.service import execute_chat_query

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize storage (SQLite tables & MongoDB indexes) and preload model
    storage = get_storage()
    storage.initialize()
    try:
        ModelManager.get_instance().load_model()
    except Exception as e:
        print(f"[Lifespan Warning] Could not preload model at startup: {e}")
    yield
    # Shutdown logic (if any)

app = FastAPI(
    title="VisionIQ Surveillance API",
    description="AI-Powered Event Processing Backend with MongoDB Persistence for CCTV Suspicious Activity Recognition",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for future frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# Root Endpoint (Friendly greeting + docs link)
# -------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    """
    Root endpoint redirecting or displaying API overview.
    """
    return {
        "name": "VisionIQ Surveillance API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health"
    }

# -------------------------------------------------------------
# 1. Health Endpoint
# -------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Returns API health, model loading status, and database connectivity.
    Reports MongoDB status, engine type, and fallback availability.
    """
    model_mgr = ModelManager.get_instance()
    storage = get_storage()
    db_info = storage.get_status()

    return HealthResponse(
        status="healthy",
        api="VisionIQ Surveillance API",
        version="1.0.0",
        model=model_mgr.get_info(),
        database=db_info["status"],
        database_engine=db_info["engine"],
        database_details=db_info["details"],
        mongodb={
            "connected": db_info["engine"] == "mongodb",
            "details": db_info["details"],
            "database": db_info["database"]
        }
    )

# -------------------------------------------------------------
# 2. Video Upload Endpoint
# -------------------------------------------------------------
@app.post("/api/videos/upload", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED, tags=["Videos"])
async def upload_video(
    file: UploadFile = File(...)
):
    """
    Validates and stores an uploaded CCTV video file, returns metadata and unique video_id.
    Persists video document to MongoDB and mirrors to SQLite.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format '{ext}'. Allowed formats: {sorted(list(ALLOWED_EXTENSIONS))}"
        )

    video_id = str(uuid.uuid4())
    safe_filename = f"{video_id}_{os.path.basename(file.filename)}"
    destination_path = os.path.join(VIDEOS_DIR, safe_filename)

    # Stream write to disk
    bytes_written = 0
    try:
        with open(destination_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)
                bytes_written += len(chunk)
    except Exception as e:
        if os.path.exists(destination_path):
            os.remove(destination_path)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    if bytes_written == 0:
        if os.path.exists(destination_path):
            os.remove(destination_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    # Probe video metadata using OpenCV
    try:
        meta = probe_video_file(destination_path)
    except Exception as e:
        if os.path.exists(destination_path):
            os.remove(destination_path)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted video: {str(e)}")

    # Store in database via StorageRepository
    storage = get_storage()
    video_record = storage.create_video({
        "video_id": video_id,
        "filename": file.filename,
        "file_path": destination_path,
        "file_size": bytes_written,
        "status": "uploaded",
        "duration": meta.get("duration"),
        "fps": meta.get("fps"),
        "total_frames": meta.get("total_frames"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "resolution": meta.get("resolution"),
        "created_at": datetime.now(timezone.utc)
    })

    return VideoUploadResponse(
        video_id=video_record["video_id"],
        filename=video_record["filename"],
        status=video_record["status"],
        file_size=video_record["file_size"],
        duration=video_record.get("duration"),
        fps=video_record.get("fps"),
        resolution=video_record.get("resolution") or (f"{video_record['width']}x{video_record['height']}" if video_record.get("width") and video_record.get("height") else None),
        created_at=video_record["created_at"]
    )

# -------------------------------------------------------------
# 3. Video Processing Endpoint
# -------------------------------------------------------------
@app.post("/api/videos/{video_id}/process", response_model=VideoProcessResponse, tags=["Videos"])
def process_video(
    video_id: str,
    background_tasks: BackgroundTasks,
    confidence_threshold: float = Query(DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
    window_sec: float = Query(DEFAULT_WINDOW_SEC, gt=0.0),
    stride_sec: float = Query(DEFAULT_STRIDE_SEC, gt=0.0)
):
    """
    Initiates asynchronous action recognition processing for an uploaded video.
    """
    storage = get_storage()
    video = storage.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")

    if video["status"] == "processing":
        raise HTTPException(status_code=409, detail="Video is already being processed.")

    if not os.path.exists(video["file_path"]):
        raise HTTPException(status_code=404, detail="Underlying video file is missing from disk.")

    # Mark as processing in storage
    storage.update_video_status(video_id, status="processing")

    # Launch background task
    background_tasks.add_task(
        process_video_background,
        video_id=video_id,
        threshold=confidence_threshold,
        window_sec=window_sec,
        stride_sec=stride_sec
    )

    return VideoProcessResponse(
        video_id=video["video_id"],
        status="processing",
        message="Video processing started in background."
    )

# -------------------------------------------------------------
# 4. List All Videos Endpoint
# -------------------------------------------------------------
@app.get("/api/videos", response_model=VideosListResponse, tags=["Videos"])
def list_videos(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Retrieves all uploaded videos with pagination, event counts, and processing statuses.
    """
    storage = get_storage()
    total, videos = storage.list_videos(limit=limit, offset=offset)

    return VideosListResponse(
        total=total,
        videos=[
            VideoStatusResponse(
                video_id=v["video_id"],
                filename=v["filename"],
                status=v["status"],
                events_count=storage.get_video_events_count(v["video_id"]),
                duration=v.get("duration"),
                fps=v.get("fps"),
                error_message=v.get("error_message"),
                created_at=v["created_at"],
                updated_at=v.get("updated_at") or v["created_at"]
            )
            for v in videos
        ]
    )

# -------------------------------------------------------------
# 5. Video Status Endpoint
# -------------------------------------------------------------
@app.get("/api/videos/{video_id}/status", response_model=VideoStatusResponse, tags=["Videos"])
def get_video_status(video_id: str):
    """
    Reports current status (uploaded, processing, completed, failed) and event count.
    """
    storage = get_storage()
    video = storage.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")

    events_count = storage.get_video_events_count(video_id)

    return VideoStatusResponse(
        video_id=video["video_id"],
        filename=video["filename"],
        status=video["status"],
        events_count=events_count,
        duration=video.get("duration"),
        fps=video.get("fps"),
        error_message=video.get("error_message"),
        created_at=video["created_at"],
        updated_at=video.get("updated_at") or video["created_at"]
    )

# -------------------------------------------------------------
# 5. Video Events Endpoint
# -------------------------------------------------------------
@app.get("/api/videos/{video_id}/events", response_model=List[EventResponse], tags=["Events"])
def get_video_events(video_id: str):
    """
    Returns detected suspicious activity events for a specific video.
    """
    storage = get_storage()
    video = storage.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")

    events = storage.get_video_events(video_id)
    return [
        EventResponse(
            event_id=ev["event_id"],
            video_id=ev["video_id"],
            activity_label=ev["activity_label"],
            confidence=ev["confidence"],
            start_timestamp=ev["start_timestamp"],
            end_timestamp=ev["end_timestamp"],
            start_time_formatted=ev["start_time_formatted"],
            end_time_formatted=ev["end_time_formatted"],
            evidence_snapshot=ev.get("evidence_snapshot"),
            evidence_clip=ev.get("evidence_clip"),
            processing_status=ev.get("processing_status", "completed"),
            created_at=ev["created_at"]
        )
        for ev in events
    ]

# -------------------------------------------------------------
# 6. Cross-Video Events Query & Filter Endpoint
# -------------------------------------------------------------
@app.get("/api/events", response_model=EventsListResponse, tags=["Events"])
def list_events(
    activity: Optional[str] = Query(None, description="Filter by activity category (e.g. Fighting, Robbery)"),
    video_id: Optional[str] = Query(None, description="Filter by specific video ID"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Retrieves events across all videos with optional filtering by activity, video ID, or minimum confidence.
    Leverages indexed queries in MongoDB with automatic SQLite fallback.
    """
    storage = get_storage()
    total, events = storage.list_events(
        activity=activity,
        video_id=video_id,
        min_confidence=min_confidence,
        limit=limit,
        offset=offset
    )

    return EventsListResponse(
        total=total,
        events=[
            EventResponse(
                event_id=ev["event_id"],
                video_id=ev["video_id"],
                activity_label=ev["activity_label"],
                confidence=ev["confidence"],
                start_timestamp=ev["start_timestamp"],
                end_timestamp=ev["end_timestamp"],
                start_time_formatted=ev["start_time_formatted"],
                end_time_formatted=ev["end_time_formatted"],
                evidence_snapshot=ev.get("evidence_snapshot"),
                evidence_clip=ev.get("evidence_clip"),
                processing_status=ev.get("processing_status", "completed"),
                created_at=ev["created_at"]
            )
            for ev in events
        ]
    )

# -------------------------------------------------------------
# 7. Evidence Snapshot Serving Endpoint
# -------------------------------------------------------------
@app.get("/api/evidence/{filename}", tags=["Evidence"])
def get_evidence_snapshot(filename: str):
    """
    Safely serves an evidence snapshot image from disk with path-traversal prevention.
    """
    clean_filename = os.path.basename(filename)
    file_path = os.path.join(EVIDENCE_DIR, clean_filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Evidence snapshot not found.")

    return FileResponse(file_path, media_type="image/jpeg")

# -------------------------------------------------------------
# 8. Semantic Search & Grounded AI Chatbot Endpoint
# -------------------------------------------------------------
@app.post("/api/chat/query", response_model=ChatQueryResponse, tags=["AI Chatbot"])
def chat_query(request: ChatQueryRequest):
    """
    Executes hybrid semantic search and grounded AI question answering over stored CCTV events.
    Matches activities, confidence scores, and time ranges, and grounds all answers in real evidence.
    """
    res = execute_chat_query(
        query=request.query,
        video_id=request.video_id,
        activity=request.activity,
        min_confidence=request.min_confidence,
        limit=request.limit
    )

    return ChatQueryResponse(
        query=res["query"],
        answer=res["answer"],
        applied_filters=res["applied_filters"],
        events=[
            ChatEventRecord(
                event_id=ev["event_id"],
                video_id=ev["video_id"],
                activity_label=ev["activity_label"],
                confidence=ev["confidence"],
                start_timestamp=ev["start_timestamp"],
                end_timestamp=ev["end_timestamp"],
                start_time_formatted=ev["start_time_formatted"],
                end_time_formatted=ev["end_time_formatted"],
                evidence_snapshot=ev.get("evidence_snapshot"),
                filename=ev.get("filename"),
                relevance_score=ev.get("relevance_score")
            )
            for ev in res["events"]
        ],
        total=res["total"],
        retrieval_method=res["retrieval_method"],
        limitations=res["limitations"]
    )
