# TruthLens AI — Deepfake Detection

TruthLens AI is a full-stack deepfake detection platform that analyzes images, videos, and audio files to determine whether they are authentic or AI-generated/manipulated.

---

## Features

- **Multi-media support** — detect deepfakes in images, videos, and audio files
- **Multi-model detection pipeline** — HuggingFace pre-trained model (primary), OpenCLIP semantic analysis, and DCT frequency-domain analysis (fallbacks)
- **Cloud API integration** — images and audio are additionally verified via the [Reality Defender](https://realitydefender.com/) API
- **Visual heatmaps** — suspicious regions are highlighted with a color-coded overlay for fake images and video frames
- **Video frame analysis** — up to 15 frames are sampled from each video; 5 representative frames with heatmaps are returned to the client
- **Confidence scoring** — every result includes a verdict (`REAL` / `FAKE`) and a percentage confidence score
- **REST API** — clean JSON API consumed by the Next.js frontend

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js (React) |
| Backend API | Python · Flask · Flask-CORS |
| Primary detector | HuggingFace `AutoModelForImageClassification` |
| Semantic fallback | OpenCLIP (`ViT-B-32`, LAION-2B) |
| Frequency fallback | DCT analysis (SciPy) |
| Cloud API | Reality Defender SDK |
| Computer vision | OpenCV · Pillow |
| GPU acceleration | PyTorch (CUDA) |

---

## Project Structure

```
deepfake/
├── backend/
│   ├── api/
│   │   ├── app.py            # Flask REST API
│   │   └── requirements.txt  # Python dependencies
│   └── detection/
│       └── detector.py       # Detection models & heatmap generation
└── frontend/
    └── truthlensAI/          # Next.js application
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+ (for the frontend)
- A CUDA-capable GPU (recommended; falls back to CPU automatically)
- A [Reality Defender](https://realitydefender.com/) API key

### Backend Setup

```bash
cd backend/api

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py
```

The server starts on **http://localhost:5000**.

> **Note:** A pre-trained HuggingFace deepfake detection model is expected at `models/deepfake_detector_v1/` relative to the repository root. Download any compatible image classification model from [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=image-classification) and save it to that path (e.g. `huggingface-cli download <model-id> --local-dir models/deepfake_detector_v1`). The detector falls back to CLIP + DCT analysis automatically if the model is not found.

### Frontend Setup

```bash
cd frontend/truthlensAI

npm install
npm run dev
```

The frontend starts on **http://localhost:3000**.

---

## API Reference

### `GET /api/health`

Returns the current status of the API and loaded models.

**Response**
```json
{
  "status": "healthy",
  "detector": "ready",
  "models": ["GANDCTAnalysis", "OpenCLIP"]
}
```

`GANDCTAnalysis` refers to the DCT frequency-domain fallback analyzer; `OpenCLIP` is the CLIP semantic fallback.

---

### `POST /api/analyze`

Analyzes an uploaded file and returns a deepfake detection result.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | The media file to analyze |

**Supported formats**

| Type | Extensions |
|---|---|
| Image | `png`, `jpg`, `jpeg` |
| Video | `mp4`, `mov`, `webm`, `avi`, `mkv` |
| Audio | `mp3`, `wav`, `flac`, `aac`, `m4a`, `ogg` |

Maximum file size: **100 MB**

**Response**
```json
{
  "verdict": "FAKE",
  "confidence": 87,
  "source": "cloud_api",
  "heatmap": "data:image/png;base64,...",
  "scores": {
    "appearance": 87,
    "frequency": 87,
    "final": 87
  },
  "metadata": {
    "type": "image",
    "resolution": "1920x1080",
    "fileSize": 204800,
    "format": "JPG"
  },
  "warnings": []
}
```

For **video** responses, `videoFrames` and `videoHeatmaps` arrays (base64-encoded) are also included when the verdict is `FAKE`.

---

### `POST /api/batch-analyze`

> ⚠️ Not yet implemented — returns `501 Not Implemented`.

---

## Detection Pipeline

```
File upload
    │
    ├─ Image ──► Reality Defender Cloud API ──► Heatmap (if FAKE)
    │
    ├─ Audio ──► Reality Defender Cloud API
    │
    └─ Video ──► HuggingFace model (primary)
                     │
                     └─ CLIP + DCT fallback (if model unavailable)
                              │
                              └─ Frame previews + Heatmaps (if FAKE)
```

A score between 0 and 1 is produced by each model (0 = real, 1 = fake). Scores above **0.5** result in a `FAKE` verdict.

---

## Environment Variables

| Variable | Description |
|---|---|
| `REALITY_DEFENDER_API_KEY` | API key for the Reality Defender cloud service |

> **Security:** The API key is currently hardcoded in `app.py`. Before committing or deploying, remove it from the source code and load it from an environment variable or a secrets manager instead (e.g. `os.environ["REALITY_DEFENDER_API_KEY"]`).

---

## License

This project is provided for educational and research purposes. Please review the licenses of all third-party models and APIs before using them in a production environment.
