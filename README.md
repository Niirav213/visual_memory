# 🧠 Visual Memory AI — From Seeing to Remembering

> An AI system that detects objects, tracks them across video frames, stores visual memories, and answers natural language questions about where and when objects were last seen.

## 🏗️ Architecture

```
Camera/Video → YOLOv8 Detection → BoT-SORT Tracking → CLIP Embedding → FAISS + SQLite Memory
                                                                              ↓
                                          User Query → CLIP Text Embedding → Vector Search → Response
```

### Core Modules

| Module | Technology | Purpose |
|--------|-----------|---------|
| **Detection** | YOLOv8 (Ultralytics) | Real-time object detection |
| **Tracking** | BoT-SORT | Persistent object identity across frames |
| **Embedding** | CLIP ViT-B/32 | Visual & text feature extraction |
| **Memory Store** | FAISS + SQLite | Vector similarity search + metadata |
| **Query Engine** | Template NLP | Natural language question answering |
| **Dashboard** | Streamlit | Interactive web interface |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

### 3. Or run the CLI version

```bash
python main.py
```

## 📁 Project Structure

```
visualmemory/
├── config/
│   └── settings.yaml       # Configuration (models, thresholds, paths)
├── src/
│   ├── detection/
│   │   └── detector.py      # YOLOv8 object detection
│   ├── tracking/
│   │   └── tracker.py       # BoT-SORT multi-object tracking
│   ├── memory/
│   │   ├── embedder.py      # CLIP visual embeddings
│   │   ├── vector_store.py  # FAISS vector index
│   │   └── memory_db.py     # SQLite memory metadata
│   ├── query/
│   │   └── engine.py        # NL query processing
│   └── utils/
│       └── visualization.py # Drawing & annotation helpers
├── data/                    # Persistent storage (auto-created)
├── app.py                   # Streamlit dashboard
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
└── README.md
```

## 💬 Example Queries

- "Where is my laptop?"
- "When did I last see my phone?"
- "How many times have you seen a bottle?"
- "What objects have you seen?"
- "Where is the cup?"

## 🛠️ Configuration

Edit `config/settings.yaml` to customize:

- **Detection model**: `yolov8n.pt` (nano), `yolov8s.pt` (small), `yolov8m.pt` (medium)
- **Confidence threshold**: Default 0.45
- **Target classes**: Filter to specific object types
- **Memory interval**: Min seconds between re-storing same object (default 5s)
- **Camera source**: Webcam index or video file path

## 👥 Team

- Navya Thomas
- Kishor N.
- Nirav Parmar
- Prarabdh Parmar
