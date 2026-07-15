"""
Visual Memory AI — Object Tracking Module
Uses Ultralytics built-in BoT-SORT tracker for persistent multi-object tracking.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackedObject:
    """Represents a tracked object with persistent identity across frames."""
    track_id: int                        # Unique persistent track ID
    bbox: Tuple[int, int, int, int]      # (x1, y1, x2, y2) bounding box
    class_name: str                       # Human-readable class name
    confidence: float                     # Detection confidence [0, 1]
    class_id: int                         # YOLO class index
    timestamp: str                        # ISO format timestamp of detection

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Area of the bounding box in pixels."""
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def get_region(self, frame_width: int, frame_height: int) -> str:
        """
        Estimate spatial region of the object within the frame.
        Returns a human-readable location description.
        """
        cx, cy = self.center
        col = "left" if cx < frame_width / 3 else ("center" if cx < 2 * frame_width / 3 else "right")
        row = "top" if cy < frame_height / 3 else ("middle" if cy < 2 * frame_height / 3 else "bottom")
        return f"{row}-{col}"

    def crop_from_frame(self, frame: np.ndarray, padding: int = 10) -> np.ndarray:
        """
        Extract the object crop from the frame with optional padding.

        Args:
            frame: Full BGR frame as numpy array
            padding: Pixel padding around the bounding box

        Returns:
            Cropped BGR image of the object
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return frame[y1:y2, x1:x2]


class ObjectTracker:
    """
    Multi-object tracker using Ultralytics built-in BoT-SORT.

    Provides persistent object IDs across video frames, enabling
    the memory system to track individual objects over time.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_type: str = "botsort.yaml",
        confidence: float = 0.45,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Initialize the object tracker.

        Args:
            model_name: YOLO model variant
            tracker_type: Tracker config ('botsort.yaml' or 'bytetrack.yaml')
            confidence: Minimum detection confidence
            target_classes: List of class names to track
        """
        self.model = YOLO(model_name)
        self.tracker_type = tracker_type
        self.confidence = confidence
        self.target_classes = target_classes

        # Build class filter
        self._class_name_to_id = {v: k for k, v in self.model.names.items()}
        self._target_class_ids = None
        if target_classes:
            self._target_class_ids = []
            for name in target_classes:
                if name in self._class_name_to_id:
                    self._target_class_ids.append(self._class_name_to_id[name])

        # Track history for trajectory analysis
        self._track_history: dict = {}  # track_id → list of (center_x, center_y, timestamp)

        print(f"[Tracker] Loaded {model_name} with {tracker_type} tracker")
        if target_classes:
            print(f"[Tracker] Tracking {len(target_classes)} classes")

    def track(self, frame: np.ndarray) -> List[TrackedObject]:
        """
        Run detection + tracking on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            List of TrackedObject with persistent track IDs
        """
        timestamp = datetime.now().isoformat()

        # Run YOLO tracking
        results = self.model.track(
            frame,
            conf=self.confidence,
            classes=self._target_class_ids,
            tracker=self.tracker_type,
            persist=True,
            verbose=False,
        )

        tracked_objects = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # Extract bounding box
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Extract tracking info
                track_id = int(boxes.id[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                confidence = float(boxes.conf[i].cpu().numpy())

                obj = TrackedObject(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    class_name=class_name,
                    confidence=confidence,
                    class_id=class_id,
                    timestamp=timestamp,
                )

                tracked_objects.append(obj)

                # Update track history
                cx, cy = obj.center
                if track_id not in self._track_history:
                    self._track_history[track_id] = []
                self._track_history[track_id].append((cx, cy, timestamp))

                # Keep only last 100 positions per track
                if len(self._track_history[track_id]) > 100:
                    self._track_history[track_id] = self._track_history[track_id][-100:]

        return tracked_objects

    def get_track_history(self, track_id: int) -> list:
        """Get the position history for a specific track ID."""
        return self._track_history.get(track_id, [])

    def get_active_tracks(self) -> List[int]:
        """Get list of all track IDs that have been seen."""
        return list(self._track_history.keys())

    def reset(self):
        """Reset the tracker state."""
        self._track_history.clear()
        # Reinitialize the model to reset internal tracker state
        self.model = YOLO(self.model.model_name if hasattr(self.model, 'model_name') else "yolov8n.pt")
        print("[Tracker] Reset complete")
