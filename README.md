# 🧠 Visual Memory AI — From Seeing to Remembering

> **Scalable, production-oriented visual memory system for everyday objects.**  
> Continuously records object sightings across camera feeds, associates observations with spatial zones and timestamps, and answers natural-language questions with verifiable visual evidence rather than assumptions.

---

## 🎯 Key Scope Clarification
This system is strictly designed to remember **everyday physical objects** (e.g., laptop, phone, bottle, cup, bag, book, keys, remote, scissors) and answer **where and when each object was last observed**.  
*Pedestrian tracking, person identification, and face re-identification features are deliberately excluded from this object-centric architecture.*

---

## 🏗️ Target Architecture

```
Camera Feeds (Webcam / RTSP / Video)
       │
       ▼
[1. Ingestion Service] (Frame Sampling & Bounded Queue)
       │
       ▼
[2. Detection Service] (YOLOv8 Object-Centric Inference)
       │
       ▼
[3. Tracking Service] (BoT-SORT / ByteTrack Short-Term Identity)
       │
       ▼
[4. Embedding Service] (OpenCLIP ViT-B/32 Visual Crop Projections)
       │
       ▼
[5. Memory Worker & Persistence] (SQLite Metadata + FAISS Vectors + Media Storage)
       │
       ▼
[6. Query Engine & FastAPI] (SQL Filters + Vector Similarity + Temporal Logic)
       │
       ▼
[7. Next.js Web Dashboard] (Overview • Live Vision • Search • Timeline • Ask • Observability)
```

---

## 💻 Production Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Next.js 14, React 18, TypeScript, Tailwind CSS, Lucide | Production-grade dashboard with glassmorphism UI |
| **API Backend** | FastAPI, Pydantic v2, Uvicorn | Typed endpoints, validation, Swagger docs, WebSockets |
| **Video Ingestion** | OpenCV, MJPEG Streamer, Bounded Queue | Multi-source camera ingestion with backpressure protection |
| **Object Detection** | Ultralytics YOLOv8 (`yolov8n.pt`) | Real-time everyday object detection |
| **Object Tracking** | BoT-SORT / ByteTrack | Intra-stream track continuity without false long-term assumptions |
| **Visual Embeddings** | OpenCLIP / CLIP (`ViT-B/32`) | Shared text-image semantic embedding space (512-d) |
| **Persistence** | SQLite + FAISS (`IndexFlatIP`) | Indexed metadata store & vector similarity search |
| **Containerization** | Docker, Docker Compose | Multi-container local & cloud deployments |
| **Observability** | psutil, Telemetry Collectors | Real-time CPU, RAM, FPS, p50/p95/p99 latency metrics |

---

## 📁 Repository Structure

```
visual-memory-ai/
├── apps/
│   ├── api/                    # Production FastAPI backend
│   │   ├── main.py             # REST endpoints, WebSocket & worker coordination
│   │   └── seed_data.py        # Realistic initial memories & snapshot seeder
│   └── web/                    # Next.js 14 + React + TypeScript frontend
│       ├── app/
│       │   ├── page.tsx          # 1. Overview Dashboard
│       │   ├── live/page.tsx     # 2. Live Vision & Spatial Tracking
│       │   ├── search/page.tsx   # 3. Visual Memory Search
│       │   ├── timeline/page.tsx # 4. Memory Chronological Timeline
│       │   ├── ask/page.tsx      # 5. Ask Visual Memory (NL Q&A + Speech)
│       │   └── observability/    # 6. System Observability & Worker Telemetry
│       ├── components/           # Reusable UI components (Navbar, Modals, StatCards)
│       └── lib/                  # Typed API client & data interfaces
├── services/
│   ├── ingestion/              # Multi-source video streamer with backpressure
│   ├── detection/              # YOLOv8 object detector with zone mapping
│   ├── tracking/               # BoT-SORT multi-object tracker
│   ├── embedding/              # CLIP visual crop and text embedder
│   ├── memory_worker/          # Idempotent SQLite persistence & FAISS index
│   └── query_engine/           # Evidence-backed hybrid NL query engine
├── packages/
│   ├── shared_types/           # Shared Pydantic schemas (MemoryEvent, QueryResponse)
│   └── api_client/             # Reusable API client SDK
├── ml/
│   ├── configs/                # YAML configuration files (config.yaml)
│   ├── evaluation/             # Benchmark suite (benchmark.py)
│   └── experiments/            # Model fine-tuning scripts
├── datasets/
│   ├── manifests/              # Object memory benchmark ground-truth JSON
│   └── evaluation-splits/      # Test splits
├── infrastructure/
│   ├── docker/                 # Production Dockerfiles (API & Web)
│   └── monitoring/             # Telemetry configurations
├── tests/
│   ├── unit/                   # Pytest unit tests (Query engine, persistence)
│   └── integration/            # FastAPI endpoint integration tests
├── data/                       # Persistent databases & evidence media snapshots
├── docker-compose.yml          # Multi-container orchestration
├── app.py                      # Original Streamlit prototype (internal experiment tool)
└── README.md
```

---

## 🌐 Next.js Frontend Dashboard (6 Core Pages)

1. **Overview (`/`)**: High-level KPIs (Total Sightings, Unique Objects, Active Cameras, FPS), active camera cards, and a real-time recent sighting feed with snapshot inspection.
2. **Live Vision (`/live`)**: Live camera stream viewing `/api/cameras/camera_01/stream` with bounding box overlays, target inspector, camera switcher, and stream pause/resume controls.
3. **Visual Memory Search (`/search`)**: Multi-attribute filtering by object class, camera, zone (`study_desk`, `kitchen_counter`, `living_room_table`, `entryway_shelf`), and semantic text search powered by CLIP embeddings.
4. **Memory Timeline (`/timeline`)**: Chronological visual history with automatic detection of:
   - `Observed` (standard sightings)
   - `Moved` (spatial zone transitions)
   - `Visibility Gaps` (unobserved intervals)
   - `Reappeared` (reappearance events)
5. **Ask Visual Memory (`/ask`)**: Natural-language question interface answering:
   - *"Where is my laptop?"*
   - *"When did I last see my phone?"*
   - *"How many times have you seen a bottle?"*
   - *"Find the red cup near the window"*
   - Supported by exact zone badges, camera nodes, time-ago, and clickable evidence snapshots with Web Speech read-aloud support.
6. **System Observability (`/observability`)**: Telemetry breakdown across all 6 decoupled worker threads, CPU/RAM usage meters, queue lag gauges, and p50/p95/p99 execution latency distributions.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ (with PyTorch, Ultralytics, and FastAPI)
- Node.js 18+ and npm

### 1. Start the FastAPI Core Backend
```bash
# From the project root:
python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```
*API Swagger Documentation will be available at: `http://localhost:8000/docs`*

### 2. Start the Next.js Frontend
```bash
cd apps/web
npm run dev
```
*Web dashboard will be live at: `http://localhost:3000`*

### 3. Run Automated Tests
```bash
# Run unit tests
python -m pytest tests/unit/test_query_engine.py -v

# Run API integration tests
python -m pytest tests/integration/test_api_endpoints.py -v
```

### 4. Run Evaluation Benchmark
```bash
python ml/evaluation/benchmark.py
```

### 5. Run with Docker Compose
```bash
docker-compose up --build
```

---

## 📋 Standard Memory Event Schema

```json
{
  "event_id": "evt_000184",
  "camera_id": "camera_01",
  "zone_id": "study_desk",
  "track_id": "track_0042",
  "object_identity_id": null,
  "class_name": "laptop",
  "confidence": 0.94,
  "timestamp_utc": "2026-10-26T09:30:15Z",
  "bbox_xyxy": [210, 120, 480, 360],
  "embedding_id": "vec_000184",
  "snapshot_uri": "/media/snapshots/evt_000184.jpg",
  "crop_uri": "/media/crops/evt_000184.jpg",
  "event_label": "observed"
}
```

---

## 👥 Core Project Contributors
- **Navya Thomas**
- **Kishor N.**
- **Nirav Parmar**
- **Prarabdh Parmar**
