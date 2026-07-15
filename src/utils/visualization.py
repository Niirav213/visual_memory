"""
Visual Memory AI — Visualization Utilities
Drawing bounding boxes, labels, track IDs, and info overlays on frames.
"""

import cv2
import numpy as np
from typing import List, Tuple

# Curated color palette for tracked objects (BGR format)
PALETTE = [
    (255, 107, 107),   # Coral red
    (78, 205, 196),    # Teal
    (255, 195, 0),     # Amber
    (130, 88, 255),    # Purple
    (0, 184, 148),     # Green
    (253, 121, 168),   # Pink
    (108, 92, 231),    # Indigo
    (0, 206, 209),     # Turquoise
    (255, 159, 67),    # Orange
    (46, 213, 115),    # Emerald
    (116, 185, 255),   # Sky blue
    (255, 234, 167),   # Cream
]


def get_color(track_id: int) -> Tuple[int, int, int]:
    """Get a consistent color for a track ID."""
    return PALETTE[track_id % len(PALETTE)]


def draw_tracked_objects(
    frame: np.ndarray,
    tracked_objects: list,
    show_confidence: bool = True,
    show_region: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, and track IDs on a frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for obj in tracked_objects:
        color = get_color(obj.track_id)
        x1, y1, x2, y2 = obj.bbox

        # Draw bounding box with rounded-corner effect
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Build label text
        label = f"#{obj.track_id} {obj.class_name}"
        if show_confidence:
            label += f" {obj.confidence:.0%}"
        if show_region:
            label += f" [{obj.get_region(w, h)}]"

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(y1 - th - 10, 0)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 8, y1), color, -1, cv2.LINE_AA)

        # Draw label text
        cv2.putText(
            annotated, label, (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Draw center dot
        cx, cy = obj.center
        cv2.circle(annotated, (cx, cy), 3, color, -1, cv2.LINE_AA)

    return annotated


def draw_detections(
    frame: np.ndarray,
    detections: list,
) -> np.ndarray:
    """Draw detection boxes without track IDs (for standalone detection mode)."""
    annotated = frame.copy()

    for det in detections:
        color = (78, 205, 196)  # Teal
        x1, y1, x2, y2 = det.bbox

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(y1 - th - 10, 0)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 8, y1), color, -1, cv2.LINE_AA)
        cv2.putText(
            annotated, label, (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


def create_info_overlay(
    frame: np.ndarray,
    stats: dict,
) -> np.ndarray:
    """
    Add a semi-transparent stats overlay to the top of the frame.
    stats: dict with keys like 'fps', 'objects', 'memories', 'tracking'
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Semi-transparent dark bar at top
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    # Stats text
    x_pos = 10
    items = []
    if "fps" in stats:
        items.append(f"FPS: {stats['fps']:.1f}")
    if "objects" in stats:
        items.append(f"Objects: {stats['objects']}")
    if "memories" in stats:
        items.append(f"Memories: {stats['memories']}")
    if "tracking" in stats:
        items.append(f"Tracks: {stats['tracking']}")

    status_text = "  |  ".join(items)
    cv2.putText(
        annotated, status_text, (x_pos, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1, cv2.LINE_AA,
    )

    return annotated


def get_direction_description(region: str) -> str:
    """Get a human-readable direction description from a screen region."""
    region_map = {
        "top-left": "up and left",
        "top-center": "up",
        "top-right": "up and right",
        "middle-left": "left",
        "middle-center": "forward",
        "middle-right": "right",
        "bottom-left": "down and left",
        "bottom-center": "down",
        "bottom-right": "down and right"
    }
    return region_map.get(region, "towards the last seen position")


def draw_guidance_overlay(
    frame: np.ndarray,
    target_name: str,
    target_detected: bool,
    target_bbox: tuple = None,
    latest_mem: object = None,
) -> np.ndarray:
    """
    Draw bounding boxes, arrows, lines to center, and banners for guidance instructions.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    cx, cy = w // 2, h // 2

    # Draw semi-transparent dark banner at bottom for guidance text
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    if target_detected and target_bbox is not None:
        # Glow green colors
        color = (46, 213, 115)  # Emerald BGR
        x1, y1, x2, y2 = target_bbox
        
        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        
        # Center of target
        tcx = (x1 + x2) // 2
        tcy = (y1 + y2) // 2
        cv2.circle(annotated, (tcx, tcy), 6, color, -1, cv2.LINE_AA)
        
        # Center of screen dot
        cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
        
        # Guidance line from center of screen to center of target
        cv2.line(annotated, (cx, cy), (tcx, tcy), color, 2, cv2.LINE_AA)
        
        # Banner message
        msg = f"TARGET LOCKED: {target_name.upper()} LOCATED"
        cv2.putText(
            annotated, msg, (20, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
        )
    else:
        # Guidance arrows based on last known region
        if latest_mem is not None:
            region = getattr(latest_mem, "region", "middle-center")
            color = (255, 159, 67)  # Orange BGR
            
            # Map region to direction vector (relative to center)
            dx, dy = 0, 0
            if "left" in region:
                dx = -1
            elif "right" in region:
                dx = 1
                
            if "top" in region:
                dy = -1
            elif "bottom" in region:
                dy = 1
                
            if dx != 0 or dy != 0:
                # Draw big arrow pointing from center
                start_pt = (cx - dx * 30, cy - dy * 30)
                end_pt = (cx + dx * 80, cy + dy * 80)
                cv2.arrowedLine(annotated, start_pt, end_pt, color, 5, tipLength=0.3)
                
            # Text instructions
            dir_desc = get_direction_description(region).upper()
            msg = f"PAN CAMERA {dir_desc} (LAST SEEN AT {region.upper()})"
            cv2.putText(
                annotated, msg, (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA
            )
        else:
            # No memory of target yet
            color = (108, 92, 231)  # Indigo BGR
            msg = f"SEARCHING FOR {target_name.upper()}..."
            cv2.putText(
                annotated, msg, (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
            )
            
    return annotated
