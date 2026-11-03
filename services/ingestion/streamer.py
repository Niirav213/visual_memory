"""
Visual Memory AI — Video Ingestion & Streaming Service
Decoupled multi-source video ingestion (webcam, file, RTSP, or synthetic test generator)
with frame sampling, backpressure, and MJPEG preview streaming.
"""

import time
import threading
import queue
from typing import Generator, Optional, Tuple, Dict, Any
import cv2
import numpy as np


class VideoIngestionService:
    """
    Decoupled video ingestion with bounded queues and frame sampling.
    Prioritizes real-time physical webcam capture (using DirectShow on Windows)
    with graceful fallback to synthetic frames only when no hardware camera exists.
    """

    def __init__(
        self,
        source: Any = 0,
        sample_fps: int = 15,
        target_resolution: Tuple[int, int] = (640, 480),
        max_queue_size: int = 2,
    ):
        self.source = source
        self.sample_fps = sample_fps
        self.target_resolution = target_resolution
        self.max_queue_size = max_queue_size

        self.frame_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._dropped_frames = 0
        self._last_frame_ts = time.time()
        self._is_synthetic = False
        self._camera_name = "Detecting..."
        self._source_lock = threading.Lock()

        print(f"[IngestionService] Initializing for source '{source}' at {sample_fps} FPS.")

    def start(self):
        """Start ingestion worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[IngestionService] Background capture loop started.")

    def stop(self):
        """Stop ingestion."""
        self._running = False
        with self._source_lock:
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
        print("[IngestionService] Ingestion stopped.")

    def set_source(self, new_source: Any) -> bool:
        """Dynamically switch video input source."""
        with self._source_lock:
            self.source = new_source
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = self._open_capture(new_source)
            if self._cap and self._cap.isOpened():
                self._is_synthetic = False
                print(f"[IngestionService] Switched source to: {new_source}")
                return True
            else:
                self._is_synthetic = True
                print(f"[IngestionService] Failed to open new source: {new_source}, using synthetic")
                return False

    def _open_capture(self, src: Any) -> Optional[cv2.VideoCapture]:
        """Open physical webcam with DirectShow on Windows, or file/RTSP."""
        cap = None
        is_int_source = False
        int_src = 0
        try:
            int_src = int(src)
            is_int_source = True
        except (ValueError, TypeError):
            is_int_source = False

        if is_int_source:
            # On Windows, cv2.CAP_DSHOW provides fast, reliable camera access
            try:
                print(f"[IngestionService] Attempting to open camera {int_src} with cv2.CAP_DSHOW...")
                cap = cv2.VideoCapture(int_src, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self._camera_name = f"Webcam Device {int_src} (DirectShow)"
                        print(f"[IngestionService] Successfully verified real webcam feed on device {int_src}!")
                        return cap
                    else:
                        cap.release()
            except Exception as e:
                print(f"[IngestionService] CAP_DSHOW attempt exception: {e}")

            # Fallback to standard open
            try:
                print(f"[IngestionService] Attempting standard cv2.VideoCapture({int_src})...")
                cap = cv2.VideoCapture(int_src)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self._camera_name = f"Webcam Device {int_src} (Standard)"
                        print(f"[IngestionService] Successfully verified real webcam feed (standard) on device {int_src}!")
                        return cap
                    else:
                        cap.release()
            except Exception as e:
                print(f"[IngestionService] Standard capture open exception: {e}")
        else:
            # Video file or RTSP stream
            try:
                cap = cv2.VideoCapture(str(src))
                if cap.isOpened():
                    self._camera_name = f"Stream / Video: {src}"
                    return cap
            except Exception as e:
                print(f"[IngestionService] Video source open exception: {e}")

        return None

    def _capture_loop(self):
        with self._source_lock:
            self._cap = self._open_capture(self.source)
            if self._cap and self._cap.isOpened():
                self._is_synthetic = False
                print(f"[IngestionService] Hardware camera active: {self._camera_name}")
            else:
                self._is_synthetic = True
                self._camera_name = "Synthetic Test Scene (No hardware camera detected)"
                print("[IngestionService] Physical camera not available. Running high-fidelity synthetic visual stream.")

        sleep_interval = 1.0 / max(1, self.sample_fps)
        sim_tick = 0

        while self._running:
            start_t = time.time()
            frame = None

            with self._source_lock:
                if not self._is_synthetic and self._cap and self._cap.isOpened():
                    ret, read_frame = self._cap.read()
                    if ret and read_frame is not None:
                        frame = read_frame
                        if self.target_resolution:
                            frame = cv2.resize(frame, self.target_resolution)
                    else:
                        # Video file rewind or brief frame drop
                        if isinstance(self.source, str) and not self.source.isdigit():
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, read_frame = self._cap.read()
                            if ret and read_frame is not None:
                                frame = read_frame
                        if frame is None:
                            # Single frame drop: don't permanently switch to synthetic immediately
                            sim_tick += 1
                            frame = self._generate_synthetic_frame(sim_tick)
                else:
                    sim_tick += 1
                    frame = self._generate_synthetic_frame(sim_tick)

            if frame is None:
                sim_tick += 1
                frame = self._generate_synthetic_frame(sim_tick)

            self._frame_count += 1
            self._last_frame_ts = time.time()

            # Enqueue with backpressure: drop stale frame so consumer stays real-time
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                self._dropped_frames += 1
                try:
                    _ = self.frame_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except Exception:
                    pass

            with self.lock:
                if self.latest_annotated_frame is None:
                    self.latest_annotated_frame = frame.copy()

            elapsed = time.time() - start_t
            if elapsed < sleep_interval:
                time.sleep(sleep_interval - elapsed)

    def _generate_synthetic_frame(self, tick: int) -> np.ndarray:
        """
        Generate realistic simulated room/desk scene with objects
        (laptop, water bottle, phone, coffee cup, backpack)
        to enable real-time demo without physical camera requirement.
        """
        w, h = self.target_resolution
        # Clean modern desk background
        frame = np.full((h, w, 3), (35, 30, 28), dtype=np.uint8)

        # Desk surface
        desk_top = int(h * 0.45)
        cv2.rectangle(frame, (0, desk_top), (w, h), (55, 48, 42), -1)
        cv2.line(frame, (0, desk_top), (w, desk_top), (75, 68, 60), 2)

        # Study Desk Left / Right / Center Zone dividers
        z1 = int(w * 0.33)
        z2 = int(w * 0.66)
        cv2.line(frame, (z1, desk_top), (z1, h), (60, 52, 45), 1)
        cv2.line(frame, (z2, desk_top), (z2, h), (60, 52, 45), 1)

        # Simulated Object 1: Laptop in center desk
        lx1, ly1, lx2, ly2 = int(w * 0.38), int(h * 0.52), int(w * 0.62), int(h * 0.82)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (180, 185, 190), -1)
        cv2.rectangle(frame, (lx1 + 8, ly1 + 8), (lx2 - 8, ly2 - 25), (40, 40, 50), -1)
        cv2.putText(frame, "LAPTOP", (lx1 + 18, ly1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 240), 1)

        # Simulated Object 2: Bottle on right desk (oscillating slight placement)
        bx1, by1 = int(w * 0.75), int(h * 0.50)
        bx2, by2 = int(w * 0.82), int(h * 0.82)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 140, 50), -1)
        cv2.rectangle(frame, (bx1 + 8, by1 - 18), (bx2 - 8, by1), (120, 90, 30), -1)
        cv2.putText(frame, "BOTTLE", (bx1 - 2, by1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Simulated Object 3: Phone on left desk
        px1, py1 = int(w * 0.15), int(h * 0.62)
        px2, py2 = int(w * 0.24), int(h * 0.82)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (30, 30, 30), -1)
        cv2.rectangle(frame, (px1 + 3, py1 + 3), (px2 - 3, py2 - 3), (60, 60, 70), -1)
        cv2.putText(frame, "PHONE", (px1 + 4, py1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 220, 180), 1)

        # Simulated Object 4: Cup near laptop
        cx1, cy1 = int(w * 0.28), int(h * 0.55)
        cx2, cy2 = int(w * 0.35), int(h * 0.70)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (60, 80, 220), -1)
        cv2.putText(frame, "CUP", (cx1 + 4, cy1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Add camera timestamp HUD
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"CAM-01 [STUDY_DESK] {ts_str}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        return frame

    def update_annotated_frame(self, frame: np.ndarray):
        """Update live annotated frame for web MJPEG stream."""
        with self.lock:
            self.latest_annotated_frame = frame.copy()

    def get_latest_jpeg(self) -> bytes:
        """Encode latest annotated frame to JPEG for MJPEG stream."""
        with self.lock:
            frame = self.latest_annotated_frame
            if frame is None:
                frame = self._generate_synthetic_frame(0)
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes() if ret else b""

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """Yield multipart MJPEG byte stream for HTTP response."""
        while self._running:
            frame_bytes = self.get_latest_jpeg()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(1.0 / max(1, self.sample_fps))

    @property
    def metrics(self) -> Dict[str, Any]:
        return {
            "source": str(self.source),
            "camera_name": self._camera_name,
            "is_synthetic": self._is_synthetic,
            "sample_fps": self.sample_fps,
            "queue_size": self.frame_queue.qsize(),
            "dropped_frames": self._dropped_frames,
            "total_frames": self._frame_count,
            "healthy": (time.time() - self._last_frame_ts) < 5.0,
        }
