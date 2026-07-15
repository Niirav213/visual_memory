# 🧠 Visual Memory AI — From Seeing to Remembering

> An AI system that detects objects, tracks them across video frames, stores visual memories, and answers natural language questions about where and when objects were last seen — like giving a camera a memory.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## ✨ Features

- 🎯 **Real-time object detection** powered by YOLOv8
- 🔗 **Persistent tracking** across frames with BoT-SORT, so the same object keeps its identity over time
- 🧬 **Visual & text embeddings** via CLIP ViT-B/32 for semantic search
- 🗂️ **Hybrid memory store** — FAISS for fast vector similarity search, SQLite for structured metadata
- 💬 **Natural language queries** — ask where or when something was last seen
- 📊 **Interactive dashboard** built with Streamlit, plus a lightweight CLI mode

---

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

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- A webcam or video file for input
- (Optional) CUDA-capable GPU for faster inference

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/visualmemory.git
cd visualmemory
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

### 5. Or Run the CLI Version

```bash
python main.py
```

---

## 📁 Project Structure

```
visualmemory/
├── config/
│   └── settings.yaml        # Configuration (models, thresholds, paths)
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
├── data/                     # Persistent storage (auto-created)
├── app.py                    # Streamlit dashboard
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 💬 Example Queries

Once the system has observed a scene for a while, try asking:

- "Where is my laptop?"
- "When did I last see my phone?"
- "How many times have you seen a bottle?"
- "What objects have you seen?"
- "Where is the cup?"

---

## 🛠️ Configuration

All runtime behavior is controlled from `config/settings.yaml`:

| Setting | Description | Default |
|---------|-------------|---------|
| **Detection model** | `yolov8n.pt` (nano), `yolov8s.pt` (small), `yolov8m.pt` (medium) | `yolov8n.pt` |
| **Confidence threshold** | Minimum detection confidence to accept | `0.45` |
| **Target classes** | Filter detections to specific object types | all classes |
| **Memory interval** | Minimum seconds between re-storing the same object | `5` |
| **Camera source** | Webcam index or path to a video file | `0` |

---

## 🧩 How It Works

1. **Detect** — Each frame is passed through YOLOv8 to locate objects.
2. **Track** — BoT-SORT assigns and maintains a consistent ID for each object across frames.
3. **Embed** — CLIP converts the cropped object image into a vector embedding.
4. **Store** — The embedding and metadata (timestamp, location, track ID) are saved to FAISS and SQLite.
5. **Query** — A natural language question is embedded with CLIP's text encoder and matched against stored memories to generate a response.

---

## 🧪 Troubleshooting

| Issue | Possible Fix |
|-------|---------------|
| Camera not detected | Check `camera_source` in `settings.yaml`, try a different index (0, 1, 2...) |
| Slow performance | Switch to `yolov8n.pt`, or enable GPU acceleration with CUDA |
| No query results | Ensure the object has been observed and stored before querying |
| Import errors | Confirm dependencies installed with `pip install -r requirements.txt` in the active virtual environment |

---

## 🤝 Contributing

Contributions are welcome! Please open an issue to discuss proposed changes, or submit a pull request with a clear description of what was changed and why.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 👥 Team

- Navya Thomas
- Kishor N.
- Nirav Parmar
- Prarabdh Parmar