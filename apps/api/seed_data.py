"""
Visual Memory AI — Seed Data Generator
Populates rich historical memory events and sample snapshots for demonstration.
"""

import os
from datetime import datetime, timezone, timedelta
import cv2
import numpy as np
from services.memory_worker.persistence import MemoryPersistenceService
from services.embedding.embedder import EmbeddingService


def generate_seed_image(title: str, subtitle: str, color_bgr: tuple, w: int = 640, h: int = 480) -> np.ndarray:
    """Create a stylized snapshot frame with modern cybernetic styling."""
    img = np.full((h, w, 3), (25, 20, 20), dtype=np.uint8)

    # Ambient gradient
    cv2.rectangle(img, (0, int(h * 0.45)), (w, h), (40, 35, 32), -1)
    cv2.line(img, (0, int(h * 0.45)), (w, int(h * 0.45)), (60, 55, 50), 2)

    # Object representation box
    ox1, oy1, ox2, oy2 = int(w * 0.35), int(h * 0.5), int(w * 0.65), int(h * 0.8)
    cv2.rectangle(img, (ox1, oy1), (ox2, oy2), color_bgr, -1)
    cv2.rectangle(img, (ox1, oy1), (ox2, oy2), (255, 255, 255), 2)

    # Labels
    cv2.putText(img, title.upper(), (ox1 + 10, oy1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, subtitle, (ox1 + 10, oy1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # HUD Stamp
    cv2.putText(img, "VISUAL MEMORY AI // EVIDENCE SNAPSHOT", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return img


def seed_sample_memories(memory_service: MemoryPersistenceService, embedder_service: EmbeddingService):
    """Seed initial memories if database is empty."""
    stats = memory_service.get_stats()
    if stats["total_events"] >= 10:
        print(f"[SeedData] Database already contains {stats['total_events']} events. Skipping seed.")
        return

    print("[SeedData] Seeding initial visual memories...")
    now = datetime.now(timezone.utc)

    sample_objects = [
        {
            "class_name": "laptop",
            "camera_id": "camera_01",
            "zone_id": "study_desk",
            "track_id": "track_101",
            "confidence": 0.96,
            "bbox": (210, 140, 470, 360),
            "minutes_ago": 3,
            "color": (190, 160, 70),
            "event_label": "observed",
        },
        {
            "class_name": "cell phone",
            "camera_id": "camera_01",
            "zone_id": "study_desk_left",
            "track_id": "track_102",
            "confidence": 0.92,
            "bbox": (80, 240, 180, 370),
            "minutes_ago": 12,
            "color": (160, 210, 100),
            "event_label": "observed",
        },
        {
            "class_name": "bottle",
            "camera_id": "camera_01",
            "zone_id": "study_desk_right",
            "track_id": "track_103",
            "confidence": 0.89,
            "bbox": (480, 180, 560, 380),
            "minutes_ago": 25,
            "color": (220, 120, 60),
            "event_label": "observed",
        },
        {
            "class_name": "cup",
            "camera_id": "camera_02",
            "zone_id": "kitchen_counter",
            "track_id": "track_104",
            "confidence": 0.94,
            "bbox": (150, 190, 260, 310),
            "minutes_ago": 45,
            "color": (80, 90, 230),
            "event_label": "observed",
        },
        {
            "class_name": "backpack",
            "camera_id": "camera_02",
            "zone_id": "entryway_shelf",
            "track_id": "track_105",
            "confidence": 0.91,
            "bbox": (320, 110, 510, 390),
            "minutes_ago": 70,
            "color": (140, 80, 160),
            "event_label": "first_seen",
        },
        {
            "class_name": "book",
            "camera_id": "camera_01",
            "zone_id": "study_desk_center",
            "track_id": "track_106",
            "confidence": 0.88,
            "bbox": (180, 210, 310, 320),
            "minutes_ago": 95,
            "color": (90, 160, 210),
            "event_label": "observed",
        },
        {
            "class_name": "laptop",
            "camera_id": "camera_01",
            "zone_id": "study_desk_center",
            "track_id": "track_101",
            "confidence": 0.95,
            "bbox": (210, 140, 470, 360),
            "minutes_ago": 120,
            "color": (190, 160, 70),
            "event_label": "reappeared",
        },
        {
            "class_name": "bottle",
            "camera_id": "camera_02",
            "zone_id": "kitchen_counter",
            "track_id": "track_107",
            "confidence": 0.93,
            "bbox": (420, 160, 500, 350),
            "minutes_ago": 150,
            "color": (220, 120, 60),
            "event_label": "moved",
        },
        {
            "class_name": "cell phone",
            "camera_id": "camera_01",
            "zone_id": "study_desk_left",
            "track_id": "track_102",
            "confidence": 0.91,
            "bbox": (80, 240, 180, 370),
            "minutes_ago": 180,
            "color": (160, 210, 100),
            "event_label": "first_seen",
        },
        {
            "class_name": "remote",
            "camera_id": "camera_02",
            "zone_id": "living_room_table",
            "track_id": "track_108",
            "confidence": 0.87,
            "bbox": (220, 260, 340, 340),
            "minutes_ago": 240,
            "color": (100, 100, 120),
            "event_label": "observed",
        },
    ]

    for item in sample_objects:
        t_event = now - timedelta(minutes=item["minutes_ago"])
        frame = generate_seed_image(
            item["class_name"],
            f"Zone: {item['zone_id']} | Conf: {item['confidence']:.0%}",
            item["color"]
        )
        emb = embedder_service.embed_text(f"a photo of a {item['class_name']} in {item['zone_id']}")

        memory_service.record_sighting(
            camera_id=item["camera_id"],
            zone_id=item["zone_id"],
            track_id=item["track_id"],
            class_name=item["class_name"],
            confidence=item["confidence"],
            bbox_xyxy=item["bbox"],
            frame=frame,
            embedding=emb,
            event_label=item["event_label"],
            force=True,
            now_dt=t_event,
        )

    print(f"[SeedData] Successfully seeded {len(sample_objects)} visual memory events.")
