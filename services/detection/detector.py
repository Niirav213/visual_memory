"""
Visual Memory AI — Detection Service
Production YOLO inference engine with strict object-only focus.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO
from packages.shared_types.models import DetectionItem


# Everyday object classes prioritized in scope
DEFAULT_OBJECT_CLASSES = [
    "cell phone",
    "laptop",
    "bottle",
    "cup",
    "backpack",
    "handbag",
    "book",
    "remote",
    "keyboard",
    "mouse",
    "clock",
    "scissors",
    "umbrella",
    "vase",
    "chair",
    "tv",
]


class DetectionService:
    """
    Decoupled object detection service using Ultralytics YOLOv8.
    Applies strict object-centric filtering (excluding person identification)
    and spatial zone mapping.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        allowed_classes: Optional[List[str]] = None,
        zone_definitions: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = allowed_classes or DEFAULT_OBJECT_CLASSES
        self.zone_definitions = zone_definitions or {}

        # Load YOLO model
        self.model = YOLO(model_name)

        # Build class filter mapping
        self._name_to_id: Dict[str, int] = {v: k for k, v in self.model.names.items()}
        self._target_class_ids: List[int] = [
            self._name_to_id[c] for c in self.allowed_classes if c in self._name_to_id
        ]

        print(f"[DetectionService] Initialized {model_name} with {len(self._target_class_ids)} target classes.")

    def detect(self, frame: np.ndarray, camera_zone_default: str = "study_desk") -> List[DetectionItem]:
        """
        Run object detection on an RGB/BGR numpy frame.
        """
        h, w = frame.shape[:2]
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=self._target_class_ids,
            verbose=False,
        )

        detections: List[DetectionItem] = []
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue

            boxes = res.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.model.names[class_id]
                confidence = float(boxes.conf[i].cpu().numpy())

                zone_id = self.resolve_zone(x1, y1, x2, y2, w, h, camera_zone_default)

                detections.append(DetectionItem(
                    bbox_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    class_name=class_name,
                    confidence=round(confidence, 3),
                    class_id=class_id,
                    zone_id=zone_id,
                ))

        return detections

    def resolve_zone(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        frame_w: int,
        frame_h: int,
        default_zone: str = "study_desk"
    ) -> str:
        """
        Map bounding box center to configured named zones or spatial grid zones.
        """
        cx = (x1 + x2) / 2.0 / max(1, frame_w)
        cy = (y1 + y2) / 2.0 / max(1, frame_h)

        # Check configured custom zone polygons / normalized bounding boxes
        for zone_name, (zx1, zy1, zx2, zy2) in self.zone_definitions.items():
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                return zone_name

        # Semantic grid fallback
        h_pos = "left" if cx < 0.33 else ("center" if cx < 0.66 else "right")
        v_pos = "top" if cy < 0.33 else ("mid" if cy < 0.66 else "bottom")

        if default_zone:
            return f"{default_zone}_{h_pos}"
        return f"{v_pos}_{h_pos}"

    def load_model(self, model_path: str):
        """Dynamically load new YOLO model weights."""
        print(f"[DetectionService] Loading new model weights from: {model_path}")
        self.model_name = model_path
        self.model = YOLO(model_path)
        self._name_to_id = {v: k for k, v in self.model.names.items()}
        self._target_class_ids = [
            self._name_to_id[c] for c in self.allowed_classes if c in self._name_to_id
        ]
        print(f"[DetectionService] Successfully loaded: {model_path}")
