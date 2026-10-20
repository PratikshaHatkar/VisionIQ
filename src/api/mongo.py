import os
from typing import Optional, Tuple
import pymongo
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from src.config import MONGODB_URI, MONGODB_DATABASE, MONGODB_TIMEOUT_MS

_mongo_client: Optional[MongoClient] = None

def get_mongo_client(uri: Optional[str] = None, timeout_ms: Optional[int] = None) -> Optional[MongoClient]:
    """
    Returns a connected MongoClient instance, or None if the MongoDB server is unreachable.
    """
    global _mongo_client
    target_uri = uri or MONGODB_URI
    timeout = timeout_ms or MONGODB_TIMEOUT_MS

    if _mongo_client is not None:
        try:
            _mongo_client.admin.command('ping')
            return _mongo_client
        except Exception:
            _mongo_client = None

    try:
        client = MongoClient(
            target_uri,
            serverSelectionTimeoutMS=timeout,
            connectTimeoutMS=timeout
        )
        # Validate connection with a lightweight ping
        client.admin.command('ping')
        _mongo_client = client
        return _mongo_client
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
        return None

def get_mongo_db(db_name: Optional[str] = None) -> Optional[Database]:
    """
    Returns the target MongoDB database if connected.
    """
    client = get_mongo_client()
    if client is not None:
        return client[db_name or MONGODB_DATABASE]
    return None

def init_mongo_indexes(db: Database) -> None:
    """
    Ensures required unique and search indexes on the 'videos' and 'events' collections.
    """
    # 1. Videos Collection Indexes
    videos_col = db["videos"]
    videos_col.create_index([("video_id", pymongo.ASCENDING)], unique=True, name="idx_video_id_unique")
    videos_col.create_index([("processing_status", pymongo.ASCENDING)], name="idx_processing_status")
    videos_col.create_index([("created_at", pymongo.DESCENDING)], name="idx_videos_created_at")

    # 2. Events Collection Indexes
    events_col = db["events"]
    events_col.create_index([("event_id", pymongo.ASCENDING)], unique=True, name="idx_event_id_unique")
    events_col.create_index([("video_id", pymongo.ASCENDING)], name="idx_event_video_id")
    events_col.create_index([("activity_label", pymongo.ASCENDING)], name="idx_activity_label")
    events_col.create_index([("start_timestamp", pymongo.ASCENDING)], name="idx_start_timestamp")
    events_col.create_index([("confidence", pymongo.DESCENDING)], name="idx_confidence")
    events_col.create_index([("created_at", pymongo.DESCENDING)], name="idx_events_created_at")

def check_mongo_connection() -> Tuple[bool, str]:
    """
    Pings MongoDB and returns connection status and version/error details.
    """
    client = get_mongo_client()
    if client is None:
        return False, "MongoDB connection unavailable (check if daemon is running or MONGODB_URI is valid)"
    try:
        info = client.server_info()
        version = info.get("version", "unknown")
        return True, f"MongoDB v{version} connected"
    except Exception as e:
        return False, f"MongoDB ping failed: {str(e)}"
