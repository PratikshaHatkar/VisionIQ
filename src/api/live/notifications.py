import asyncio
import json
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from fastapi import WebSocket

from src.api.live.storage import get_live_storage

class NotificationBroker:
    """
    Manages in-memory WebSocket connections and asynchronous dispatch of real-time alerts.
    Scoped by owner_id for owner-based authorization.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        # Maps owner_id -> list of active WebSocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.conn_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    async def connect(self, owner_id: str, websocket: WebSocket):
        await websocket.accept()
        with self.conn_lock:
            if owner_id not in self.active_connections:
                self.active_connections[owner_id] = []
            self.active_connections[owner_id].append(websocket)
        print(f"[NotificationBroker] Client connected for owner '{owner_id}'. Active: {len(self.active_connections[owner_id])}")

    def disconnect(self, owner_id: str, websocket: WebSocket):
        with self.conn_lock:
            if owner_id in self.active_connections:
                if websocket in self.active_connections[owner_id]:
                    self.active_connections[owner_id].remove(websocket)
                if not self.active_connections[owner_id]:
                    del self.active_connections[owner_id]
        print(f"[NotificationBroker] Client disconnected for owner '{owner_id}'.")

    def broadcast_incident_sync(self, notif_data: Dict[str, Any]):
        """
        Thread-safe entrypoint called from inference worker to persist and broadcast an alert.
        Dispatches async WebSocket message without blocking the inference loop.
        """
        owner_id = notif_data.get("owner_id", "default_owner")

        # 1. Persist notification to DB (MongoDB/SQLite)
        try:
            storage = get_live_storage()
            created_notif = storage.create_notification(notif_data)
        except Exception as e:
            print(f"[NotificationBroker] DB persistence error: {e}")
            created_notif = notif_data

        # 2. Prepare payload
        payload = {
            "type": "NEW_INCIDENT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notification": {
                "id": created_notif.get("id"),
                "incident_id": created_notif.get("incident_id"),
                "camera_id": created_notif.get("camera_id"),
                "camera_name": created_notif.get("camera_name"),
                "location": created_notif.get("location"),
                "activity_label": created_notif.get("activity_label"),
                "confidence": created_notif.get("confidence"),
                "evidence_snapshot": created_notif.get("evidence_snapshot"),
                "is_read": False,
                "is_acknowledged": False,
                "created_at": created_notif.get("created_at").isoformat() if isinstance(created_notif.get("created_at"), datetime) else str(created_notif.get("created_at"))
            }
        }

        # 3. Schedule broadcast to connected websockets
        with self.conn_lock:
            connections = list(self.active_connections.get(owner_id, []))

        if not connections:
            return

        message_str = json.dumps(payload)

        # Run async dispatch in background thread event loop
        def _dispatch():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            async def _send_all():
                for ws in connections:
                    try:
                        await ws.send_text(message_str)
                    except Exception as err:
                        print(f"[NotificationBroker] WS send failed: {err}")
            loop.run_until_complete(_send_all())
            loop.close()

        dispatch_thread = threading.Thread(target=_dispatch, daemon=True)
        dispatch_thread.start()

def get_notification_broker() -> NotificationBroker:
    return NotificationBroker.get_instance()
