import os
import time
import threading
import collections
from datetime import datetime, timezone
from typing import Dict, Optional, Generator, Tuple
import cv2
import numpy as np

from src.api.live.storage import get_live_storage

class CameraStreamWorker:
    """
    Independent worker thread managing a single camera stream.
    Features:
    - Bounded frame buffer (max 32 frames) for memory safety (~15MB per stream).
    - Thread-safe start, stop, and status updates.
    - Automatic reconnection with exponential backoff on RTSP/stream drops.
    - Supports RTSP/IP CCTV, local webcams, looped test videos, and synthetic simulation streams.
    - Real-time frame pacing matching source FPS.
    """

    def __init__(self, camera_id: str, camera_data: dict):
        self.camera_id = camera_id
        self.name = camera_data.get("name", "Camera")
        self.location = camera_data.get("location") or "Zone"
        self.owner_id = camera_data.get("owner_id", "default_owner")
        self.stream_type = camera_data.get("stream_type", "rtsp").lower()
        self.stream_url = camera_data.get("stream_url", "")
        self.confidence_threshold = float(camera_data.get("confidence_threshold", 0.50))

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_buffer: collections.deque = collections.deque(maxlen=32)
        self.buffer_lock = threading.Lock()

        self.status = "connecting"
        self.error_message: Optional[str] = None
        self.fps: float = 0.0
        self.resolution: str = "Unknown"
        self.last_frame_time = time.time()
        self.consecutive_failures = 0

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """
        Safely instantiates and opens an OpenCV VideoCapture based on stream_type.
        Includes driver fallbacks and TCP transport enforcement for network streams.
        """
        try:
            if self.stream_type == "webcam":
                # Convert 'webcam:0' or '0' to integer device index
                idx_str = self.stream_url.replace("webcam:", "").strip()
                dev_idx = int(idx_str) if idx_str.isdigit() else 0
                
                # Try DirectShow first on Windows, then fallback to CAP_ANY
                cap = None
                if os.name == 'nt':
                    try:
                        cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
                        if not cap.isOpened():
                            cap.release()
                            cap = None
                    except Exception:
                        cap = None
                if cap is None:
                    cap = cv2.VideoCapture(dev_idx, cv2.CAP_ANY)

            elif self.stream_type in ("test_video", "video"):
                path = self.stream_url
                # If a directory is provided, locate the first playable video file inside
                if os.path.isdir(path):
                    candidates = [
                        os.path.join(path, f) for f in os.listdir(path)
                        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
                    ]
                    if candidates:
                        path = candidates[0]
                cap = cv2.VideoCapture(path)

            elif self.stream_type == "synthetic":
                # Check if synthetic mode requests an anomaly simulation video
                url_lower = self.stream_url.lower()
                if any(k in url_lower for k in ("anomaly", "burglary", "robbery", "fighting", "test")):
                    candidate_paths = [
                        os.path.join("storage", "videos", "2b2a49fb-4345-44c2-a2f3-dff4db7ee02c_testing video.mp4"),
                        "sample_cctv.mp4"
                    ]
                    for cp in candidate_paths:
                        if os.path.isfile(cp):
                            cap = cv2.VideoCapture(cp)
                            if cap and cap.isOpened():
                                return cap
                return None  # Standard algorithmic generator handled in loop

            else:
                # RTSP or network IP camera stream
                # Enforce TCP transport for RTSP to prevent packet drop corruption
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                cap = cv2.VideoCapture(self.stream_url)

            if cap and cap.isOpened():
                return cap
            return None
        except Exception as e:
            self.error_message = str(e)
            return None

    def start(self):
        if self.running:
            return
        self.running = True
        self.status = "connecting"
        self.thread = threading.Thread(target=self._run_loop, name=f"CamWorker-{self.camera_id}", daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.status = "offline"
        with self.buffer_lock:
            self.frame_buffer.clear()
        self._sync_status_to_db("offline")

    def _sync_status_to_db(self, status: str, err: Optional[str] = None):
        self.status = status
        self.error_message = err
        try:
            storage = get_live_storage()
            storage.update_camera(self.camera_id, {
                "status": status,
                "is_monitored": (status in ("online", "connecting", "reconnecting")),
                "error_message": err,
                "fps": self.fps,
                "resolution": self.resolution
            })
        except Exception as e:
            print(f"[CamWorker {self.camera_id}] DB status sync error: {e}")

    def _generate_synthetic_frame(self, frame_num: int) -> np.ndarray:
        """
        Generates a clean synthetic CCTV surveillance frame with motion
        for environments without physical cameras.
        """
        h, w = 360, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Background subtle gradient
        for y in range(h):
            val = int(30 + 20 * (y / h))
            frame[y, :] = [val, val + 5, val + 10]

        # Draw grid
        for x in range(0, w, 80):
            cv2.line(frame, (x, 0), (x, h), (45, 50, 60), 1)
        for y in range(0, h, 60):
            cv2.line(frame, (0, y), (w, y), (45, 50, 60), 1)

        # Simulated moving object / motion
        pos_x = int(100 + (frame_num * 5) % 440)
        cv2.circle(frame, (pos_x, 180), 25, (0, 165, 255), -1)
        cv2.rectangle(frame, (pos_x - 30, 205), (pos_x + 30, 280), (50, 200, 50), 2)

        # CCTV On-Screen Display (OSD)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"LIVE REC | {self.name.upper()} ({self.location})", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"CAM-ID: {self.camera_id[:8]} | TIME: {now_str}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def _run_loop(self):
        self._sync_status_to_db("connecting")
        synthetic_counter = 0

        while self.running:
            # Check if using purely algorithmic synthetic stream
            if self.stream_type == "synthetic" and not any(k in self.stream_url.lower() for k in ("anomaly", "burglary", "robbery", "fighting", "test")):
                self.status = "online"
                self.fps = 20.0
                self.resolution = "640x360"
                self._sync_status_to_db("online")

                while self.running:
                    frame = self._generate_synthetic_frame(synthetic_counter)
                    synthetic_counter += 1
                    with self.buffer_lock:
                        self.frame_buffer.append(frame)
                    time.sleep(0.05)  # ~20 FPS pacing
                break

            cap = self._open_capture()
            if cap is None:
                if self.stream_type == "synthetic":
                    # Fallback to algorithmic generator
                    self.stream_url = "synthetic:0"
                    continue

                self.consecutive_failures += 1
                backoff = min(15, 2 * self.consecutive_failures)
                self._sync_status_to_db("reconnecting", f"Stream open failed. Retrying in {backoff}s...")
                time.sleep(backoff)
                continue

            # Stream opened successfully
            self.consecutive_failures = 0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            inferred_fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = inferred_fps if (inferred_fps and inferred_fps > 0 and not np.isnan(inferred_fps)) else 25.0
            self.resolution = f"{w}x{h}" if (w and h) else "720p"
            self._sync_status_to_db("online")

            last_fps_calc = time.time()
            frames_counted = 0
            # Native frame delay for file-based streams
            frame_delay = (1.0 / self.fps) if self.fps > 0 else 0.033

            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # If test video or synthetic video reached end, loop it back
                    if self.stream_type in ("test_video", "video", "synthetic"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print(f"[CamWorker {self.camera_id}] Read error or stream disconnected.")
                        break

                with self.buffer_lock:
                    self.frame_buffer.append(frame)

                frames_counted += 1
                now = time.time()
                if now - last_fps_calc >= 2.0:
                    self.fps = round(frames_counted / (now - last_fps_calc), 1)
                    frames_counted = 0
                    last_fps_calc = now

                # Real-time pacing: sleep by natural framerate for video files, or short yield for webcams
                if self.stream_type in ("test_video", "video", "synthetic"):
                    time.sleep(frame_delay)
                else:
                    time.sleep(0.005)

            cap.release()
            if self.running:
                self._sync_status_to_db("reconnecting", "Stream disconnected. Reconnecting...")
                time.sleep(3.0)

        self._sync_status_to_db("offline")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.buffer_lock:
            if self.frame_buffer:
                return self.frame_buffer[-1].copy()
            return None

    def get_frame_window(self, num_frames: int = 16) -> Optional[list]:
        """
        Retrieves a uniform sample of num_frames from the bounded frame buffer.
        """
        with self.buffer_lock:
            buf_len = len(self.frame_buffer)
            if buf_len < 8:  # Not enough temporal context yet
                return None
            snapshot = list(self.frame_buffer)

        indices = np.linspace(0, len(snapshot) - 1, num_frames, dtype=int)
        return [snapshot[i] for i in indices]

class StreamManager:
    """
    Central singleton managing camera stream worker threads and MJPEG feeds.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.workers: Dict[str, CameraStreamWorker] = {}
        self.workers_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start_camera(self, camera_id: str, camera_data: dict) -> Tuple[bool, str]:
        with self.workers_lock:
            existing = self.workers.get(camera_id)
            if existing and existing.running:
                return True, "Camera stream already running."

            worker = CameraStreamWorker(camera_id, camera_data)
            self.workers[camera_id] = worker
            worker.start()
            return True, "Camera stream started."

    def stop_camera(self, camera_id: str) -> Tuple[bool, str]:
        with self.workers_lock:
            worker = self.workers.pop(camera_id, None)
            if worker:
                worker.stop()
                return True, "Camera stream stopped."
            else:
                # Ensure DB updated even if not active in memory
                get_live_storage().update_camera(camera_id, {
                    "is_monitored": False,
                    "status": "offline"
                })
                return True, "Camera marked offline."

    def get_worker(self, camera_id: str) -> Optional[CameraStreamWorker]:
        with self.workers_lock:
            return self.workers.get(camera_id)

    def generate_mjpeg_stream(self, camera_id: str) -> Generator[bytes, None, None]:
        """
        Yields multipart JPEG frames for live browser preview.
        """
        worker = self.get_worker(camera_id)
        placeholder = None

        while True:
            frame = None
            if worker and worker.running:
                frame = worker.get_latest_frame()

            if frame is None:
                if placeholder is None:
                    placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "CAMERA OFFLINE / CONNECTING", (120, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
                frame = placeholder

            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.066)  # ~15 FPS MJPEG delivery

    def shutdown_all(self):
        with self.workers_lock:
            for cid, worker in list(self.workers.items()):
                try:
                    worker.stop()
                except Exception as e:
                    print(f"[StreamManager] Error stopping worker {cid}: {e}")
            self.workers.clear()

def get_stream_manager() -> StreamManager:
    return StreamManager.get_instance()
