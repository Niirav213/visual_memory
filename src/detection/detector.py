"""
Visual Memory AI — Object Detection Module
Uses YOLOv8 (Ultralytics) for real-time object detection.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Represents a single detected object in a frame."""
    bbox: Tuple[int, int, int, int]     # (x1, y1, x2, y2) bounding box
    class_name: str                      # Human-readable class name
    confidence: float                    # Detection confidence [0, 1]
    class_id: int                        # YOLO class index

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
        # Divide frame into a 3x3 grid
        col = "left" if cx < frame_width / 3 else ("center" if cx < 2 * frame_width / 3 else "right")
        row = "top" if cy < frame_height / 3 else ("middle" if cy < 2 * frame_height / 3 else "bottom")
        return f"{row}-{col}"


class ObjectDetector:
    """
    YOLOv8-based object detector.

    Wraps Ultralytics YOLO model for single-frame detection with
    configurable confidence thresholds and class filtering.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.45,
        target_classes: Optional[List[str]] = None,
    ):
        """
        Initialize the object detector.

        Args:
            model_name: YOLO model variant (e.g., 'yolov8n.pt', 'yolov8s.pt')
            confidence: Minimum confidence threshold for detections
            target_classes: List of class names to detect. If None, all 80 COCO classes.
        """
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.target_classes = target_classes

        # Build reverse lookup: class_name → class_id from YOLO's names dict
        self._class_name_to_id = {v: k for k, v in self.model.names.items()}

        # Resolve target class IDs for filtering
        self._target_class_ids = None
        if target_classes:
            self._target_class_ids = []
            for name in target_classes:
                if name in self._class_name_to_id:
                    self._target_class_ids.append(self._class_name_to_id[name])

        print(f"[Detector] Loaded {model_name} | Confidence: {confidence}")
        if target_classes:
            print(f"[Detector] Filtering to {len(target_classes)} classes: {target_classes}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            List of Detection objects found in the frame
        """
        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self._target_class_ids,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # Extract bounding box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Extract class info
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                confidence = float(boxes.conf[i].cpu().numpy())

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_name=class_name,
                    confidence=confidence,
                    class_id=class_id,
                ))

        return detections

    def get_class_names(self) -> dict:
        """Return the full YOLO class names dictionary."""
        return self.model.names
