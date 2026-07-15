"""
Visual Memory AI — CLI Entry Point
Headless video processing pipeline for object detection, tracking, and memory creation.
Run: python main.py
"""

import os
import sys
import time
import cv2
import yaml
import numpy as np
from datetime import datetime

from src.tracking.tracker import ObjectTracker
from src.memory.embedder import VisualEmbedder
from src.memory.vector_store import VectorStore
from src.memory.memory_db import MemoryDatabase, MemoryEntry
from src.query.engine import QueryEngine
from src.utils.visualization import (
    draw_tracked_objects, 
    create_info_overlay, 
    draw_guidance_overlay, 
    get_direction_description
)

import threading
speech_rec_available = True
try:
    import speech_recognition as sr
except ImportError:
    speech_rec_available = False

tts_available = True
try:
    import pyttsx3
except ImportError:
    tts_available = False

def recognize_speech() -> str:
    if not speech_rec_available:
        return "ERROR_IMPORT"
    
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("\n[*] Listening... Please speak your question now.")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=4, phrase_time_limit=4)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return "ERROR_UNKNOWN"
        except sr.RequestError:
            return "ERROR_REQUEST"
    except Exception as e:
        return f"ERROR_MIC: {str(e)}"

def speak_background(text: str):
    if not tts_available:
        return
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("  Visual Memory AI — From Seeing to Remembering")
    print("=" * 60)
    print()

    # Load configuration
    config = load_config()

    # Determine which model weights to load
    model_path = config["detection"]["model"]
    custom_weights_path = "data/weights/custom_best.pt"
    if os.path.exists(custom_weights_path):
        model_path = custom_weights_path
        print(f"[*] Custom trained model found! Loading: {model_path}")
    else:
        print(f"[*] Loading standard model: {model_path}")

    # Initialize all modules
    print("[*] Initializing modules...")
    tracker = ObjectTracker(
        model_name=model_path,
        tracker_type=config["tracking"]["tracker"],
        confidence=config["detection"]["confidence"],
        target_classes=config["detection"]["target_classes"],
    )

    embedder = VisualEmbedder(
        model_name=config["memory"]["embedding_model"],
    )

    vector_store = VectorStore(
        dimension=config["memory"]["embedding_dim"],
    )
    # Load existing index if available
    vector_store.load(
        config["memory"]["faiss_index_path"],
        config["memory"]["id_map_path"],
    )

    memory_db = MemoryDatabase(
        db_path=config["memory"]["db_path"],
    )

    query_engine = QueryEngine(embedder, vector_store, memory_db)

    # Track when each object was last stored to enforce interval
    last_store_times = {}
    memory_interval = config["memory"]["memory_interval_seconds"]

    # Open camera/video source
    source = config["camera"]["source"]
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])

    print(f"[*] Camera opened: {source}")
    print(f"[*] Press 'q' to quit, 'a' to open query menu")
    print()

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()

    # Active search target for guidance
    active_search = None
    guidance_located_announced = False
    last_guidance_voice_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[*] End of video stream")
                break

            frame_count += 1

            # Run detection + tracking
            tracked_objects = tracker.track(frame)

            # Target guidance logic
            target_detected = False
            target_bbox = None
            
            if active_search:
                for obj in tracked_objects:
                    if obj.class_name == active_search:
                        target_detected = True
                        target_bbox = obj.bbox
                        break

            # Store memories for tracked objects
            h, w = frame.shape[:2]
            now = datetime.now()

            for obj in tracked_objects:
                key = f"{obj.class_name}_{obj.track_id}"

                # Enforce memory interval
                if key in last_store_times:
                    elapsed = (now - last_store_times[key]).total_seconds()
                    if elapsed < memory_interval:
                        continue

                # Create memory entry
                region = obj.get_region(w, h)

                # Generate CLIP embedding from object crop
                crop = obj.crop_from_frame(frame)
                if crop.size == 0:
                    continue

                embedding = embedder.embed_image(crop)

                # Store in SQLite
                entry = MemoryEntry(
                    memory_id=None,
                    object_name=obj.class_name,
                    track_id=obj.track_id,
                    timestamp=obj.timestamp,
                    bbox_x1=obj.bbox[0],
                    bbox_y1=obj.bbox[1],
                    bbox_x2=obj.bbox[2],
                    bbox_y2=obj.bbox[3],
                    region=region,
                    confidence=obj.confidence,
                    embedding_id=None,
                )
                memory_id = memory_db.store_memory(entry)

                # Store embedding in FAISS
                vector_store.add(embedding, memory_id)

                # Save visual crop to disk
                crop_dir = "data/crops"
                os.makedirs(crop_dir, exist_ok=True)
                cv2.imwrite(os.path.join(crop_dir, f"{memory_id}.jpg"), crop)

                last_store_times[key] = now
                print(f"  [MEMORY] Stored: {obj.class_name} #{obj.track_id} @ {region} ({obj.confidence:.0%})")

            # Calculate FPS
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_timer
                fps = 10.0 / max(elapsed, 0.001)
                fps_timer = time.time()

            # Draw annotations
            annotated = draw_tracked_objects(frame, tracked_objects)
            
            # Apply directional guidance visual overlay
            if active_search:
                latest_mem = memory_db.get_latest_memory(active_search)
                annotated = draw_guidance_overlay(
                    annotated,
                    active_search,
                    target_detected,
                    target_bbox,
                    latest_mem
                )
                
                # Voice guidance logic
                now_time = time.time()
                guidance_msg = None
                
                if target_detected:
                    if not guidance_located_announced:
                        guidance_msg = f"Target {active_search} located directly ahead!"
                        guidance_located_announced = True
                        last_guidance_voice_time = now_time
                else:
                    guidance_located_announced = False
                    if latest_mem and (now_time - last_guidance_voice_time > 8.0):
                        dir_desc = get_direction_description(latest_mem.region)
                        guidance_msg = f"Pan your camera {dir_desc}."
                        last_guidance_voice_time = now_time
                        
                if guidance_msg:
                    speak_background(guidance_msg)

            annotated = create_info_overlay(annotated, {
                "fps": fps,
                "objects": len(tracked_objects),
                "memories": memory_db.get_total_count(),
                "tracking": len(tracker.get_active_tracks()),
            })

            # Display
            cv2.imshow("Visual Memory AI", annotated)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                # Interactive query mode
                print("\n[?] Query Mode: (t) Type question, (v) Voice command, (c) Clear search target")
                mode = input("Select mode (t/v/c): ").lower().strip()
                
                question = ""
                if mode == "t":
                    question = input("\n[?] Ask a question: ").strip()
                elif mode == "v":
                    recognized = recognize_speech()
                    if recognized == "ERROR_IMPORT":
                        print("[ERROR] Audio libraries not installed.")
                    elif recognized == "ERROR_UNKNOWN":
                        print("[WARNING] Could not understand audio.")
                    elif recognized.startswith("ERROR_MIC"):
                        print("[WARNING] Microphone error.")
                    elif recognized == "ERROR_REQUEST":
                        print("[ERROR] Google Speech API is unreachable.")
                    else:
                        print(f"\n[Voice Command] {recognized}")
                        question = recognized
                elif mode == "c":
                    active_search = None
                    guidance_located_announced = False
                    print("[*] Guidance target cleared.")
                    continue
                else:
                    print("[WARNING] Invalid mode selected.")
                    continue
                
                if question:
                    answer, matched_mem = query_engine.query(question)
                    print(f"\n[AI] {answer}\n")
                    speak_background(answer)
                    
                    # Lock target for guidance
                    target_obj = query_engine._extract_object(question)
                    if target_obj:
                        active_search = target_obj
                        guidance_located_announced = False
                        print(f"[*] Guidance locked on target: {active_search.upper()}")

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")

    finally:
        # Save state
        print("\n[*] Saving memory state...")
        vector_store.save(
            config["memory"]["faiss_index_path"],
            config["memory"]["id_map_path"],
        )
        cap.release()
        cv2.destroyAllWindows()

        # Print summary
        total = memory_db.get_total_count()
        objects = memory_db.get_unique_objects()
        print(f"\n[*] Session complete: {total} memories, {len(objects)} unique objects")
        print(f"[*] Objects seen: {', '.join(objects) if objects else 'None'}")
        memory_db.close()


if __name__ == "__main__":
    main()
