import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from src.config import MONGODB_DATABASE
from src.api.mongo import get_mongo_db, get_mongo_client, check_mongo_connection, init_mongo_indexes
from src.api.database import SessionLocal, init_db as init_sqlite_db
from src.api.models import Video, Event

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class StorageRepository:
    """
    Unified storage repository supporting MongoDB as primary store with
    automatic SQLite fallback and dual-sync persistence.
    """

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """
        Initializes both SQLite schemas and MongoDB indexes.
        """
        # 1. Initialize SQLite tables
        try:
            init_sqlite_db()
        except Exception as e:
            print(f"[Storage] SQLite init warning: {e}")

        # 2. Initialize MongoDB indexes if connected
        try:
            db = get_mongo_db()
            if db is not None:
                init_mongo_indexes(db)
                print(f"[Storage] MongoDB indexes verified for database '{MONGODB_DATABASE}'.")
        except Exception as e:
            print(f"[Storage] MongoDB init warning: {e}")

        self._initialized = True

    def get_status(self) -> Dict[str, Any]:
        """
        Checks connectivity to MongoDB and returns status information.
        """
        is_mongo_up, mongo_info = check_mongo_connection()
        if is_mongo_up:
            return {
                "status": "connected",
                "engine": "mongodb",
                "details": mongo_info,
                "database": MONGODB_DATABASE,
                "fallback_available": True
            }
        else:
            return {
                "status": "connected",  # Service is operational via SQLite fallback
                "engine": "sqlite_fallback",
                "details": f"SQLite active ({mongo_info})",
                "database": "sqlite",
                "fallback_available": True
            }

    # -------------------------------------------------------------
    # Video Operations
    # -------------------------------------------------------------
    def create_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persists a newly uploaded video metadata record into MongoDB and SQLite.
        """
        video_id = video_data.get("video_id") or str(uuid.uuid4())
        now = video_data.get("created_at") or utc_now()
        
        doc = {
            "video_id": video_id,
            "filename": video_data["filename"],
            "file_path": video_data["file_path"],
            "file_size": video_data.get("file_size", 0),
            "status": video_data.get("status", "uploaded"),
            "duration": video_data.get("duration"),
            "fps": video_data.get("fps"),
            "total_frames": video_data.get("total_frames"),
            "width": video_data.get("width"),
            "height": video_data.get("height"),
            "resolution": video_data.get("resolution"),
            "error_message": video_data.get("error_message"),
            "created_at": now,
            "updated_at": now
        }

        # 1. Save to MongoDB
        db = get_mongo_db()
        if db is not None:
            try:
                db["videos"].update_one(
                    {"video_id": video_id},
                    {"$set": doc},
                    upsert=True
                )
            except Exception as e:
                print(f"[Storage Error] Failed to insert video into MongoDB: {e}")

        # 2. Mirror to SQLite
        try:
            with SessionLocal() as sq_db:
                existing = sq_db.query(Video).filter(Video.id == video_id).first()
                if not existing:
                    v_model = Video(
                        id=video_id,
                        filename=doc["filename"],
                        file_path=doc["file_path"],
                        file_size=doc["file_size"],
                        status=doc["status"],
                        duration=doc["duration"],
                        fps=doc["fps"],
                        total_frames=doc["total_frames"],
                        width=doc["width"],
                        height=doc["height"],
                        error_message=doc["error_message"],
                        created_at=doc["created_at"],
                        updated_at=doc["updated_at"]
                    )
                    sq_db.add(v_model)
                    sq_db.commit()
        except Exception as e:
            print(f"[Storage Warning] Mirror to SQLite failed for video {video_id}: {e}")

        return doc

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a video record by video_id from MongoDB or SQLite.
        """
        # Check MongoDB first
        db = get_mongo_db()
        if db is not None:
            try:
                doc = db["videos"].find_one({"video_id": video_id}, {"_id": 0})
                if doc:
                    return doc
            except Exception as e:
                print(f"[Storage Error] Mongo get_video failed: {e}")

        # Fallback to SQLite
        try:
            with SessionLocal() as sq_db:
                v = sq_db.query(Video).filter(Video.id == video_id).first()
                if v:
                    return {
                        "video_id": v.id,
                        "filename": v.filename,
                        "file_path": v.file_path,
                        "file_size": v.file_size,
                        "status": v.status,
                        "duration": v.duration,
                        "fps": v.fps,
                        "total_frames": v.total_frames,
                        "width": v.width,
                        "height": v.height,
                        "resolution": f"{v.width}x{v.height}" if v.width and v.height else None,
                        "error_message": v.error_message,
                        "created_at": v.created_at,
                        "updated_at": v.updated_at
                    }
        except Exception as e:
            print(f"[Storage Error] SQLite get_video failed: {e}")

        return None

    def update_video_status(
        self,
        video_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Updates the status and optional error_message of a video.
        """
        now = utc_now()
        update_fields = {
            "status": status,
            "error_message": error_message,
            "updated_at": now
        }

        # 1. Update in MongoDB
        db = get_mongo_db()
        if db is not None:
            try:
                db["videos"].update_one(
                    {"video_id": video_id},
                    {"$set": update_fields}
                )
            except Exception as e:
                print(f"[Storage Error] Mongo update_video_status failed: {e}")

        # 2. Update in SQLite
        try:
            with SessionLocal() as sq_db:
                v = sq_db.query(Video).filter(Video.id == video_id).first()
                if v:
                    v.status = status
                    v.error_message = error_message
                    v.updated_at = now
                    sq_db.commit()
        except Exception as e:
            print(f"[Storage Error] SQLite update_video_status failed: {e}")

        return self.get_video(video_id)

    # -------------------------------------------------------------
    # Event Operations
    # -------------------------------------------------------------
    def save_events(self, video_id: str, events_data: List[Dict[str, Any]]) -> int:
        """
        Idempotently persists detected events for a video.
        Clears previous events for this video_id before inserting to prevent duplicates.
        """
        # 1. MongoDB Save
        db = get_mongo_db()
        if db is not None:
            try:
                # Remove existing events for this video to ensure idempotence
                db["events"].delete_many({"video_id": video_id})
                if events_data:
                    # Clean documents for Mongo insertion
                    mongo_docs = []
                    for ev in events_data:
                        edoc = dict(ev)
                        if "_id" in edoc:
                            del edoc["_id"]
                        mongo_docs.append(edoc)
                    db["events"].insert_many(mongo_docs)
            except Exception as e:
                print(f"[Storage Error] Mongo save_events failed: {e}")

        # 2. Mirror to SQLite
        try:
            with SessionLocal() as sq_db:
                sq_db.query(Event).filter(Event.video_id == video_id).delete()
                for ed in events_data:
                    ev_model = Event(
                        id=ed["event_id"],
                        video_id=ed["video_id"],
                        activity_label=ed["activity_label"],
                        confidence=ed["confidence"],
                        start_timestamp=ed["start_timestamp"],
                        end_timestamp=ed["end_timestamp"],
                        start_time_formatted=ed["start_time_formatted"],
                        end_time_formatted=ed["end_time_formatted"],
                        evidence_snapshot=ed.get("evidence_snapshot"),
                        evidence_clip=ed.get("evidence_clip"),
                        processing_status=ed.get("processing_status", "completed"),
                        created_at=ed.get("created_at") or utc_now()
                    )
                    sq_db.add(ev_model)
                sq_db.commit()
        except Exception as e:
            print(f"[Storage Error] SQLite save_events failed: {e}")

        return len(events_data)

    def get_video_events(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all detected events for a given video, ordered by start_timestamp.
        """
        # 1. Check MongoDB
        db = get_mongo_db()
        if db is not None:
            try:
                docs = list(db["events"].find({"video_id": video_id}, {"_id": 0}).sort("start_timestamp", 1))
                if docs:
                    return docs
            except Exception as e:
                print(f"[Storage Error] Mongo get_video_events failed: {e}")

        # 2. Check SQLite fallback
        try:
            with SessionLocal() as sq_db:
                events = sq_db.query(Event).filter(Event.video_id == video_id).order_by(Event.start_timestamp).all()
                if events:
                    return [
                        {
                            "event_id": ev.id,
                            "video_id": ev.video_id,
                            "activity_label": ev.activity_label,
                            "confidence": ev.confidence,
                            "start_timestamp": ev.start_timestamp,
                            "end_timestamp": ev.end_timestamp,
                            "start_time_formatted": ev.start_time_formatted,
                            "end_time_formatted": ev.end_time_formatted,
                            "evidence_snapshot": ev.evidence_snapshot,
                            "evidence_clip": ev.evidence_clip,
                            "processing_status": ev.processing_status,
                            "created_at": ev.created_at
                        }
                        for ev in events
                    ]
        except Exception as e:
            print(f"[Storage Error] SQLite get_video_events failed: {e}")

        return []

    def get_video_events_count(self, video_id: str) -> int:
        """
        Returns count of events associated with a video.
        """
        db = get_mongo_db()
        if db is not None:
            try:
                count = db["events"].count_documents({"video_id": video_id})
                if count > 0:
                    return count
            except Exception as e:
                print(f"[Storage Error] Mongo count failed: {e}")

        try:
            with SessionLocal() as sq_db:
                return sq_db.query(Event).filter(Event.video_id == video_id).count()
        except Exception:
            return 0

    def list_events(
        self,
        activity: Optional[str] = None,
        video_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Searches and filters events across videos with pagination.
        """
        db = get_mongo_db()
        if db is not None:
            try:
                query_filter = {}
                if activity:
                    query_filter["activity_label"] = {"$regex": f"^{activity.strip()}$", "$options": "i"}
                if video_id:
                    query_filter["video_id"] = video_id.strip()
                if min_confidence is not None:
                    query_filter["confidence"] = {"$gte": float(min_confidence)}

                total = db["events"].count_documents(query_filter)
                if total > 0 or not self._has_sqlite_events(activity, video_id, min_confidence):
                    cursor = db["events"].find(
                        query_filter,
                        {"_id": 0}
                    ).sort("created_at", -1).skip(offset).limit(limit)
                    return total, list(cursor)
            except Exception as e:
                print(f"[Storage Error] Mongo list_events failed: {e}")

        # SQLite query
        return self._list_events_sqlite(activity, video_id, min_confidence, limit, offset)

    def _has_sqlite_events(
        self,
        activity: Optional[str],
        video_id: Optional[str],
        min_confidence: Optional[float]
    ) -> bool:
        try:
            with SessionLocal() as sq_db:
                q = sq_db.query(Event)
                if activity:
                    q = q.filter(Event.activity_label.ilike(activity.strip()))
                if video_id:
                    q = q.filter(Event.video_id == video_id.strip())
                if min_confidence is not None:
                    q = q.filter(Event.confidence >= min_confidence)
                return q.count() > 0
        except Exception:
            return False

    def _list_events_sqlite(
        self,
        activity: Optional[str],
        video_id: Optional[str],
        min_confidence: Optional[float],
        limit: int,
        offset: int
    ) -> Tuple[int, List[Dict[str, Any]]]:
        try:
            with SessionLocal() as sq_db:
                q = sq_db.query(Event)
                if activity:
                    q = q.filter(Event.activity_label.ilike(activity.strip()))
                if video_id:
                    q = q.filter(Event.video_id == video_id.strip())
                if min_confidence is not None:
                    q = q.filter(Event.confidence >= min_confidence)

                total = q.count()
                events = q.order_by(Event.created_at.desc()).offset(offset).limit(limit).all()

                results = [
                    {
                        "event_id": ev.id,
                        "video_id": ev.video_id,
                        "activity_label": ev.activity_label,
                        "confidence": ev.confidence,
                        "start_timestamp": ev.start_timestamp,
                        "end_timestamp": ev.end_timestamp,
                        "start_time_formatted": ev.start_time_formatted,
                        "end_time_formatted": ev.end_time_formatted,
                        "evidence_snapshot": ev.evidence_snapshot,
                        "evidence_clip": ev.evidence_clip,
                        "processing_status": ev.processing_status,
                        "created_at": ev.created_at
                    }
                    for ev in events
                ]
                return total, results
        except Exception as e:
            print(f"[Storage Error] SQLite _list_events_sqlite failed: {e}")
            return 0, []

    def list_videos(self, limit: int = 50, offset: int = 0) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Retrieves uploaded videos with pagination from MongoDB or SQLite fallback.
        """
        db = get_mongo_db()
        if db is not None:
            try:
                total = db["videos"].count_documents({})
                cursor = db["videos"].find({}, {"_id": 0}).sort("created_at", -1).skip(offset).limit(limit)
                return total, list(cursor)
            except Exception as e:
                print(f"[Storage Error] Mongo list_videos failed: {e}")

        try:
            with SessionLocal() as sq_db:
                total = sq_db.query(Video).count()
                videos = sq_db.query(Video).order_by(Video.created_at.desc()).offset(offset).limit(limit).all()
                results = [
                    {
                        "video_id": v.id,
                        "filename": v.filename,
                        "file_path": v.file_path,
                        "file_size": v.file_size,
                        "status": v.status,
                        "duration": v.duration,
                        "fps": v.fps,
                        "total_frames": v.total_frames,
                        "width": v.width,
                        "height": v.height,
                        "resolution": f"{v.width}x{v.height}" if v.width and v.height else None,
                        "error_message": v.error_message,
                        "created_at": v.created_at,
                        "updated_at": v.updated_at
                    }
                    for v in videos
                ]
                return total, results
        except Exception as e:
            print(f"[Storage Error] SQLite list_videos failed: {e}")
            return 0, []

# Global singleton repository instance
storage = StorageRepository()

def get_storage() -> StorageRepository:
    """
    Dependency or accessor for the unified storage repository.
    """
    return storage
