"""Quick integration test for the Visual Memory AI pipeline."""
import numpy as np
import sys

def test_pipeline():
    print("=" * 50)
    print("  Visual Memory AI — Integration Test")
    print("=" * 50)

    # 1. Test Memory Database
    print("\n[1/5] Testing MemoryDatabase...")
    from src.memory.memory_db import MemoryDatabase, MemoryEntry
    db = MemoryDatabase("data/test_memories.db")
    entry = MemoryEntry(
        memory_id=None, object_name="laptop", track_id=1,
        timestamp="2026-06-05T10:00:00", bbox_x1=100, bbox_y1=100,
        bbox_x2=300, bbox_y2=300, region="middle-center",
        confidence=0.92, embedding_id=None,
    )
    mid = db.store_memory(entry)
    retrieved = db.get_memory(mid)
    assert retrieved is not None
    assert retrieved.object_name == "laptop"
    print(f"  ✓ Stored and retrieved memory #{mid}: {retrieved.object_name}")
    
    latest = db.get_latest_memory("laptop")
    assert latest is not None
    print(f"  ✓ Latest memory for 'laptop': {latest.region} @ {latest.timestamp}")
    
    # Store more entries
    for obj in ["cell phone", "bottle", "cup"]:
        e = MemoryEntry(
            memory_id=None, object_name=obj, track_id=2,
            timestamp="2026-06-05T10:05:00", bbox_x1=50, bbox_y1=50,
            bbox_x2=200, bbox_y2=200, region="top-left",
            confidence=0.85, embedding_id=None,
        )
        db.store_memory(e)
    
    objects = db.get_unique_objects()
    print(f"  ✓ Unique objects: {objects}")
    print(f"  ✓ Total memories: {db.get_total_count()}")

    # 2. Test Vector Store
    print("\n[2/5] Testing VectorStore...")
    from src.memory.vector_store import VectorStore
    vs = VectorStore(dimension=512)
    
    # Create fake embeddings
    emb1 = np.random.randn(512).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)
    emb2 = np.random.randn(512).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)
    
    vs.add(emb1, memory_id=1)
    vs.add(emb2, memory_id=2)
    print(f"  ✓ Added 2 embeddings, store size: {vs.size}")
    
    results = vs.search(emb1, top_k=2)
    print(f"  ✓ Search results: {results}")
    assert results[0][0] == 1  # emb1 should match itself best
    print(f"  ✓ Self-similarity match correct!")

    # Save and reload
    vs.save("data/test_index.faiss", "data/test_id_map.json")
    vs2 = VectorStore(dimension=512)
    vs2.load("data/test_index.faiss", "data/test_id_map.json")
    assert vs2.size == 2
    print(f"  ✓ Save/load works, reloaded size: {vs2.size}")

    # 3. Test CLIP Embedder
    print("\n[3/5] Testing VisualEmbedder (CLIP)...")
    from src.memory.embedder import VisualEmbedder
    embedder = VisualEmbedder(model_name="ViT-B/32")
    
    # Create a synthetic image (red square on blue background)
    test_img = np.zeros((224, 224, 3), dtype=np.uint8)
    test_img[:, :] = [255, 0, 0]  # Blue (BGR)
    test_img[50:174, 50:174] = [0, 0, 255]  # Red square
    
    img_emb = embedder.embed_image(test_img)
    assert img_emb.shape == (512,)
    assert abs(np.linalg.norm(img_emb) - 1.0) < 0.01  # Normalized
    print(f"  ✓ Image embedding shape: {img_emb.shape}, norm: {np.linalg.norm(img_emb):.4f}")
    
    txt_emb = embedder.embed_text("a red square on blue background")
    assert txt_emb.shape == (512,)
    print(f"  ✓ Text embedding shape: {txt_emb.shape}, norm: {np.linalg.norm(txt_emb):.4f}")
    
    similarity = np.dot(img_emb, txt_emb)
    print(f"  ✓ Image-text similarity: {similarity:.4f}")

    # 4. Test Query Engine
    print("\n[4/5] Testing QueryEngine...")
    from src.query.engine import QueryEngine
    qe = QueryEngine(embedder, vs, db)
    
    answer1, mem1 = qe.query("Where is my laptop?")
    print(f"  Q: Where is my laptop?")
    print(f"  A: {answer1}")
    assert mem1 is not None
    assert mem1.object_name == "laptop"
    
    answer2, mem2 = qe.query("What objects have you seen?")
    print(f"\n  Q: What objects have you seen?")
    print(f"  A: {answer2}")
    assert mem2 is None
    
    answer3, mem3 = qe.query("When did I last see my phone?")
    print(f"\n  Q: When did I last see my phone?")
    print(f"  A: {answer3}")
    assert mem3 is not None
    assert mem3.object_name == "cell phone"

    # 5. Test Detector initialization
    print("\n[5/5] Testing ObjectDetector initialization...")
    from src.detection.detector import ObjectDetector
    detector = ObjectDetector(
        model_name="yolov8n.pt",
        confidence=0.45,
        target_classes=["laptop", "cell phone", "bottle"],
    )
    # Test on synthetic image
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_frame)
    print(f"  ✓ Detector loaded, ran on blank frame: {len(detections)} detections (expected 0)")

    # Close database connection to release file lock on Windows
    db.close()

    # Cleanup test files
    import os
    for f in ["data/test_memories.db", "data/test_index.faiss", "data/test_id_map.json"]:
        if os.path.exists(f):
            os.remove(f)

    print("\n" + "=" * 50)
    print("  ALL TESTS PASSED ✓")
    print("=" * 50)

if __name__ == "__main__":
    test_pipeline()
