import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import pymongo
from sqlalchemy import text as sa_text

from src.api.mongo import get_mongo_db, check_mongo_connection
from src.api.database import SessionLocal, engine, Base
from src.api.live.models import Camera, LiveIncident, LiveNotification
from src.api.live.schemas import mask_stream_url

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class LiveStorageRepository:
    """
    Unified dual-engine persistence repository for Live Surveillance entities
    (Cameras, Live Incidents, and Notifications) supporting MongoDB with SQLite fallback.
    """

    def __init__(self):
        self._initialized = False
        try:
            self.initialize()
        except Exception:
            pass

    def initialize(self):
        # 1. Initialize SQLite tables for live surveillance models
        try:
            Base.metadata.create_all(bind=engine)
            # Automatic schema migration for existing SQLite database
            with engine.connect() as conn:
                try:
                    conn.execute(sa_text("ALTER TABLE live_incidents ADD COLUMN summary TEXT"))
                    conn.commit()
                except Exception:
                    pass
                try:
                    conn.execute(sa_text("ALTER TABLE live_notifications ADD COLUMN summary TEXT"))
                    conn.commit()
                except Exception:
                    pass
        except Exception as e:
            print(f"[LiveStorage] SQLite schema warning: {e}")

        # 2. Ensure MongoDB indexes for live surveillance collections
        try:
            db = get_mongo_db()
            if db is not None:
                db["cameras"].create_index([("camera_id", pymongo.ASCENDING)], unique=True, name="idx_cam_id_uniq")
                db["cameras"].create_index([("owner_id", pymongo.ASCENDING)], name="idx_cam_owner")
                db["cameras"].create_index([("status", pymongo.ASCENDING)], name="idx_cam_status")

                db["live_incidents"].create_index([("incident_id", pymongo.ASCENDING)], unique=True, name="idx_inc_id_uniq")
                db["live_incidents"].create_index([("camera_id", pymongo.ASCENDING)], name="idx_inc_cam")
                db["live_incidents"].create_index([("owner_id", pymongo.ASCENDING)], name="idx_inc_owner")
                db["live_incidents"].create_index([("created_at", pymongo.DESCENDING)], name="idx_inc_created")

                db["live_notifications"].create_index([("notification_id", pymongo.ASCENDING)], unique=True, name="idx_notif_id_uniq")
                db["live_notifications"].create_index([("owner_id", pymongo.ASCENDING)], name="idx_notif_owner")
                db["live_notifications"].create_index([("is_read", pymongo.ASCENDING)], name="idx_notif_read")
                db["live_notifications"].create_index([("created_at", pymongo.DESCENDING)], name="idx_notif_created")
                print("[LiveStorage] Live Surveillance MongoDB collections and indexes verified.")
        except Exception as e:
            print(f"[LiveStorage] MongoDB index creation warning: {e}")

        self._initialized = True

    # -------------------------------------------------------------
    # Cameras CRUD
    # -------------------------------------------------------------
    def create_camera(self, data: Dict[str, Any]) -> Dict[str, Any]:
        camera_id = data.get("id") or str(uuid.uuid4())
        now = data.get("created_at") or utc_now()

        doc = {
            "camera_id": camera_id,
            "id": camera_id,
            "name": data["name"],
            "location": data.get("location"),
            "owner_id": data.get("owner_id", "default_owner"),
            "stream_type": data.get("stream_type", "rtsp"),
            "stream_url": data["stream_url"],
            "is_monitored": bool(data.get("is_monitored", False)),
            "status": data.get("status", "offline"),
            "error_message": data.get("error_message"),
            "fps": data.get("fps"),
            "resolution": data.get("resolution"),
            "confidence_threshold": float(data.get("confidence_threshold", 0.50)),
            "created_at": now,
            "updated_at": now
        }

        # 1. Save to MongoDB
        db = get_mongo_db()
        if db is not None:
            try:
                db["cameras"].update_one({"camera_id": camera_id}, {"$set": doc}, upsert=True)
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB create_camera failed: {e}")

        # 2. Dual-write to SQLite
        try:
            with SessionLocal() as session:
                cam = Camera(
                    id=camera_id,
                    name=doc["name"],
                    location=doc["location"],
                    owner_id=doc["owner_id"],
                    stream_type=doc["stream_type"],
                    stream_url=doc["stream_url"],
                    is_monitored=doc["is_monitored"],
                    status=doc["status"],
                    error_message=doc["error_message"],
                    fps=doc["fps"],
                    resolution=doc["resolution"],
                    confidence_threshold=doc["confidence_threshold"],
                    created_at=doc["created_at"],
                    updated_at=doc["updated_at"]
                )
                session.merge(cam)
                session.commit()
        except Exception as e:
            print(f"[LiveStorage Error] SQLite create_camera failed: {e}")

        return doc

    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        db = get_mongo_db()
        if db is not None:
            try:
                res = db["cameras"].find_one({"$or": [{"camera_id": camera_id}, {"id": camera_id}]})
                if res:
                    res["id"] = res.get("id") or res.get("camera_id")
                    res.pop("_id", None)
                    return res
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB get_camera failed: {e}")

        # SQLite fallback
        try:
            with SessionLocal() as session:
                cam = session.query(Camera).filter(Camera.id == camera_id).first()
                if cam:
                    return {
                        "id": cam.id,
                        "name": cam.name,
                        "location": cam.location,
                        "owner_id": cam.owner_id,
                        "stream_type": cam.stream_type,
                        "stream_url": cam.stream_url,
                        "is_monitored": cam.is_monitored,
                        "status": cam.status,
                        "error_message": cam.error_message,
                        "fps": cam.fps,
                        "resolution": cam.resolution,
                        "confidence_threshold": cam.confidence_threshold,
                        "created_at": cam.created_at,
                        "updated_at": cam.updated_at
                    }
        except Exception as e:
            print(f"[LiveStorage Error] SQLite get_camera failed: {e}")

        return None

    def update_camera(self, camera_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        updates["updated_at"] = utc_now()

        db = get_mongo_db()
        if db is not None:
            try:
                db["cameras"].update_one(
                    {"$or": [{"camera_id": camera_id}, {"id": camera_id}]},
                    {"$set": updates}
                )
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB update_camera failed: {e}")

        # SQLite update
        try:
            with SessionLocal() as session:
                cam = session.query(Camera).filter(Camera.id == camera_id).first()
                if cam:
                    for k, v in updates.items():
                        if hasattr(cam, k):
                            setattr(cam, k, v)
                    session.commit()
        except Exception as e:
            print(f"[LiveStorage Error] SQLite update_camera failed: {e}")

        return self.get_camera(camera_id)

    def delete_camera(self, camera_id: str) -> bool:
        db = get_mongo_db()
        if db is not None:
            try:
                db["cameras"].delete_one({"$or": [{"camera_id": camera_id}, {"id": camera_id}]})
                db["live_incidents"].delete_many({"camera_id": camera_id})
                db["live_notifications"].delete_many({"camera_id": camera_id})
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB delete_camera failed: {e}")

        try:
            with SessionLocal() as session:
                cam = session.query(Camera).filter(Camera.id == camera_id).first()
                if cam:
                    session.delete(cam)
                    session.commit()
                    return True
        except Exception as e:
            print(f"[LiveStorage Error] SQLite delete_camera failed: {e}")

        return True

    def list_cameras(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        db = get_mongo_db()
        if db is not None:
            try:
                q = {}
                if owner_id:
                    q["owner_id"] = owner_id
                cursor = db["cameras"].find(q).sort("created_at", pymongo.DESCENDING)
                results = []
                for doc in cursor:
                    doc["id"] = doc.get("id") or doc.get("camera_id")
                    doc.pop("_id", None)
                    results.append(doc)
                return results
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB list_cameras failed: {e}")

        # SQLite fallback
        try:
            with SessionLocal() as session:
                query = session.query(Camera)
                if owner_id:
                    query = query.filter(Camera.owner_id == owner_id)
                cameras = query.order_by(Camera.created_at.desc()).all()
                return [
                    {
                        "id": c.id,
                        "name": c.name,
                        "location": c.location,
                        "owner_id": c.owner_id,
                        "stream_type": c.stream_type,
                        "stream_url": c.stream_url,
                        "is_monitored": c.is_monitored,
                        "status": c.status,
                        "error_message": c.error_message,
                        "fps": c.fps,
                        "resolution": c.resolution,
                        "confidence_threshold": c.confidence_threshold,
                        "created_at": c.created_at,
                        "updated_at": c.updated_at
                    }
                    for c in cameras
                ]
        except Exception as e:
            print(f"[LiveStorage Error] SQLite list_cameras failed: {e}")
            return []

    # -------------------------------------------------------------
    # Live Incidents CRUD
    # -------------------------------------------------------------
    def create_incident(self, data: Dict[str, Any]) -> Dict[str, Any]:
        incident_id = data.get("id") or str(uuid.uuid4())
        now = data.get("created_at") or utc_now()

        doc = {
            "incident_id": incident_id,
            "id": incident_id,
            "camera_id": data["camera_id"],
            "owner_id": data["owner_id"],
            "camera_name": data.get("camera_name"),
            "location": data.get("location"),
            "activity_label": data["activity_label"],
            "confidence": round(float(data["confidence"]), 4),
            "start_time": data.get("start_time") or now,
            "end_time": data.get("end_time") or now,
            "duration_seconds": round(float(data.get("duration_seconds", 0.0)), 2),
            "summary": data.get("summary"),
            "evidence_snapshot": data.get("evidence_snapshot"),
            "evidence_clip": data.get("evidence_clip"),
            "is_acknowledged": bool(data.get("is_acknowledged", False)),
            "created_at": now
        }

        # MongoDB
        db = get_mongo_db()
        if db is not None:
            try:
                db["live_incidents"].update_one({"incident_id": incident_id}, {"$set": doc}, upsert=True)
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB create_incident failed: {e}")

        # SQLite
        try:
            with SessionLocal() as session:
                inc = LiveIncident(
                    id=incident_id,
                    camera_id=doc["camera_id"],
                    owner_id=doc["owner_id"],
                    activity_label=doc["activity_label"],
                    confidence=doc["confidence"],
                    start_time=doc["start_time"],
                    end_time=doc["end_time"],
                    duration_seconds=doc["duration_seconds"],
                    summary=doc.get("summary"),
                    evidence_snapshot=doc["evidence_snapshot"],
                    evidence_clip=doc["evidence_clip"],
                    is_acknowledged=doc["is_acknowledged"],
                    created_at=doc["created_at"]
                )
                session.merge(inc)
                session.commit()
        except Exception as e:
            print(f"[LiveStorage Error] SQLite create_incident failed: {e}")

        return doc

    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        db = get_mongo_db()
        if db is not None:
            try:
                res = db["live_incidents"].find_one({"$or": [{"incident_id": incident_id}, {"id": incident_id}]})
                if res:
                    res["id"] = res.get("id") or res.get("incident_id")
                    res.pop("_id", None)
                    return res
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB get_incident failed: {e}")

        try:
            with SessionLocal() as session:
                inc = session.query(LiveIncident).filter(LiveIncident.id == incident_id).first()
                if inc:
                    return {
                        "id": inc.id,
                        "camera_id": inc.camera_id,
                        "owner_id": inc.owner_id,
                        "camera_name": inc.camera.name if inc.camera else None,
                        "location": inc.camera.location if inc.camera else None,
                        "activity_label": inc.activity_label,
                        "confidence": inc.confidence,
                        "start_time": inc.start_time,
                        "end_time": inc.end_time,
                        "duration_seconds": inc.duration_seconds,
                        "summary": inc.summary,
                        "evidence_snapshot": inc.evidence_snapshot,
                        "evidence_clip": inc.evidence_clip,
                        "is_acknowledged": inc.is_acknowledged,
                        "created_at": inc.created_at
                    }
        except Exception as e:
            print(f"[LiveStorage Error] SQLite get_incident failed: {e}")

        return None

    def update_incident(self, incident_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        db = get_mongo_db()
        if db is not None:
            try:
                db["live_incidents"].update_one(
                    {"$or": [{"incident_id": incident_id}, {"id": incident_id}]},
                    {"$set": updates}
                )
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB update_incident failed: {e}")

        try:
            with SessionLocal() as session:
                inc = session.query(LiveIncident).filter(LiveIncident.id == incident_id).first()
                if inc:
                    for k, v in updates.items():
                        if hasattr(inc, k):
                            setattr(inc, k, v)
                    session.commit()
        except Exception as e:
            print(f"[LiveStorage Error] SQLite update_incident failed: {e}")

        return self.get_incident(incident_id)

    def list_incidents(
        self,
        owner_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        activity_label: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        db = get_mongo_db()
        if db is not None:
            try:
                q: Dict[str, Any] = {}
                if owner_id:
                    q["owner_id"] = owner_id
                if camera_id:
                    q["camera_id"] = camera_id
                if activity_label:
                    q["activity_label"] = activity_label

                cursor = db["live_incidents"].find(q).sort("created_at", pymongo.DESCENDING).limit(limit)
                results = []
                for doc in cursor:
                    doc["id"] = doc.get("id") or doc.get("incident_id")
                    doc.pop("_id", None)
                    results.append(doc)
                return results
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB list_incidents failed: {e}")

        # SQLite fallback
        try:
            with SessionLocal() as session:
                query = session.query(LiveIncident)
                if owner_id:
                    query = query.filter(LiveIncident.owner_id == owner_id)
                if camera_id:
                    query = query.filter(LiveIncident.camera_id == camera_id)
                if activity_label:
                    query = query.filter(LiveIncident.activity_label == activity_label)
                incidents = query.order_by(LiveIncident.created_at.desc()).limit(limit).all()
                return [
                    {
                        "id": inc.id,
                        "camera_id": inc.camera_id,
                        "owner_id": inc.owner_id,
                        "camera_name": inc.camera.name if inc.camera else None,
                        "location": inc.camera.location if inc.camera else None,
                        "activity_label": inc.activity_label,
                        "confidence": inc.confidence,
                        "start_time": inc.start_time,
                        "end_time": inc.end_time,
                        "duration_seconds": inc.duration_seconds,
                        "summary": inc.summary,
                        "evidence_snapshot": inc.evidence_snapshot,
                        "evidence_clip": inc.evidence_clip,
                        "is_acknowledged": inc.is_acknowledged,
                        "created_at": inc.created_at
                    }
                    for inc in incidents
                ]
        except Exception as e:
            print(f"[LiveStorage Error] SQLite list_incidents failed: {e}")
            return []

    # -------------------------------------------------------------
    # Notifications CRUD
    # -------------------------------------------------------------
    def create_notification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        notif_id = data.get("id") or str(uuid.uuid4())
        now = data.get("created_at") or utc_now()

        doc = {
            "notification_id": notif_id,
            "id": notif_id,
            "camera_id": data["camera_id"],
            "incident_id": data["incident_id"],
            "owner_id": data["owner_id"],
            "activity_label": data["activity_label"],
            "camera_name": data["camera_name"],
            "location": data.get("location"),
            "confidence": round(float(data["confidence"]), 4),
            "summary": data.get("summary"),
            "evidence_snapshot": data.get("evidence_snapshot"),
            "is_read": bool(data.get("is_read", False)),
            "is_acknowledged": bool(data.get("is_acknowledged", False)),
            "created_at": now
        }

        db = get_mongo_db()
        if db is not None:
            try:
                db["live_notifications"].update_one({"notification_id": notif_id}, {"$set": doc}, upsert=True)
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB create_notification failed: {e}")

        try:
            with SessionLocal() as session:
                notif = LiveNotification(
                    id=notif_id,
                    camera_id=doc["camera_id"],
                    incident_id=doc["incident_id"],
                    owner_id=doc["owner_id"],
                    activity_label=doc["activity_label"],
                    camera_name=doc["camera_name"],
                    location=doc["location"],
                    confidence=doc["confidence"],
                    summary=doc.get("summary"),
                    evidence_snapshot=doc["evidence_snapshot"],
                    is_read=doc["is_read"],
                    is_acknowledged=doc["is_acknowledged"],
                    created_at=doc["created_at"]
                )
                session.merge(notif)
                session.commit()
        except Exception as e:
            print(f"[LiveStorage Error] SQLite create_notification failed: {e}")

        return doc

    def list_notifications(
        self,
        owner_id: Optional[str] = None,
        is_read: Optional[bool] = None,
        limit: int = 50
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        """
        Returns (total_count, unread_count, notifications_list).
        """
        db = get_mongo_db()
        if db is not None:
            try:
                q: Dict[str, Any] = {}
                if owner_id:
                    q["owner_id"] = owner_id
                if is_read is not None:
                    q["is_read"] = is_read

                unread_q = {"is_read": False}
                if owner_id:
                    unread_q["owner_id"] = owner_id

                total = db["live_notifications"].count_documents(q)
                unread = db["live_notifications"].count_documents(unread_q)

                cursor = db["live_notifications"].find(q).sort("created_at", pymongo.DESCENDING).limit(limit)
                results = []
                for doc in cursor:
                    doc["id"] = doc.get("id") or doc.get("notification_id")
                    doc.pop("_id", None)
                    results.append(doc)
                return total, unread, results
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB list_notifications failed: {e}")

        try:
            with SessionLocal() as session:
                query = session.query(LiveNotification)
                if owner_id:
                    query = query.filter(LiveNotification.owner_id == owner_id)
                if is_read is not None:
                    query = query.filter(LiveNotification.is_read == is_read)

                total = query.count()
                unread_query = session.query(LiveNotification).filter(LiveNotification.is_read == False)
                if owner_id:
                    unread_query = unread_query.filter(LiveNotification.owner_id == owner_id)
                unread = unread_query.count()

                notifs = query.order_by(LiveNotification.created_at.desc()).limit(limit).all()
                results = [
                    {
                        "id": n.id,
                        "camera_id": n.camera_id,
                        "incident_id": n.incident_id,
                        "owner_id": n.owner_id,
                        "activity_label": n.activity_label,
                        "camera_name": n.camera_name,
                        "location": n.location,
                        "confidence": n.confidence,
                        "summary": n.summary,
                        "evidence_snapshot": n.evidence_snapshot,
                        "is_read": n.is_read,
                        "is_acknowledged": n.is_acknowledged,
                        "created_at": n.created_at
                    }
                    for n in notifs
                ]
                return total, unread, results
        except Exception as e:
            print(f"[LiveStorage Error] SQLite list_notifications failed: {e}")
            return 0, 0, []

    def mark_notification_read(self, notification_id: str, owner_id: Optional[str] = None) -> bool:
        db = get_mongo_db()
        if db is not None:
            try:
                q: Dict[str, Any] = {"$or": [{"notification_id": notification_id}, {"id": notification_id}]}
                if owner_id:
                    q["owner_id"] = owner_id
                db["live_notifications"].update_one(q, {"$set": {"is_read": True}})
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB mark_read failed: {e}")

        try:
            with SessionLocal() as session:
                query = session.query(LiveNotification).filter(LiveNotification.id == notification_id)
                if owner_id:
                    query = query.filter(LiveNotification.owner_id == owner_id)
                notif = query.first()
                if notif:
                    notif.is_read = True
                    session.commit()
                    return True
        except Exception as e:
            print(f"[LiveStorage Error] SQLite mark_read failed: {e}")

        return True

    def mark_all_notifications_read(self, owner_id: str) -> bool:
        db = get_mongo_db()
        if db is not None:
            try:
                db["live_notifications"].update_many({"owner_id": owner_id}, {"$set": {"is_read": True}})
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB mark_all_read failed: {e}")

        try:
            with SessionLocal() as session:
                session.query(LiveNotification).filter(LiveNotification.owner_id == owner_id).update({"is_read": True})
                session.commit()
                return True
        except Exception as e:
            print(f"[LiveStorage Error] SQLite mark_all_read failed: {e}")

        return True

    def acknowledge_incident(self, incident_id: str, owner_id: Optional[str] = None) -> bool:
        db = get_mongo_db()
        if db is not None:
            try:
                q: Dict[str, Any] = {"$or": [{"incident_id": incident_id}, {"id": incident_id}]}
                if owner_id:
                    q["owner_id"] = owner_id
                db["live_incidents"].update_one(q, {"$set": {"is_acknowledged": True}})
                db["live_notifications"].update_many({"incident_id": incident_id}, {"$set": {"is_acknowledged": True, "is_read": True}})
            except Exception as e:
                print(f"[LiveStorage Error] MongoDB acknowledge failed: {e}")

        try:
            with SessionLocal() as session:
                query = session.query(LiveIncident).filter(LiveIncident.id == incident_id)
                if owner_id:
                    query = query.filter(LiveIncident.owner_id == owner_id)
                inc = query.first()
                if inc:
                    inc.is_acknowledged = True
                    session.query(LiveNotification).filter(LiveNotification.incident_id == incident_id).update({
                        "is_acknowledged": True,
                        "is_read": True
                    })
                    session.commit()
                    return True
        except Exception as e:
            print(f"[LiveStorage Error] SQLite acknowledge failed: {e}")

        return True

# Global repository instance
live_storage = LiveStorageRepository()

def get_live_storage() -> LiveStorageRepository:
    return live_storage
