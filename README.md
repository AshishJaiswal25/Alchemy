# Alchemy

**A modern, async-first universal data ingestion and parsing platform for GenAI applications.**

> A unified, async-capable data ingestion architecture using the best open-source tools available in 2026.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-green)](https://fastapi.tiangolo.com)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-yellow)](LICENSE)

---

## Key Features

| Component | Technology | Highlights |
|---|---|---|
| Document OCR | **Docling** | Tables, equations, multi-column; native RAG output |
| Image Understanding | **Qwen2-VL-7B AWQ** | Captioning, OCR, table extraction |
| Audio/Video | **Distil-Whisper Large-v3** | ~6× faster than base Whisper, low WER |
| Speaker Labels | **pyannote 3.x diarization** | Speaker-labeled transcripts |
| Web Crawling | **Crawl4AI** (async) | No browser startup cost, LLM-native markdown |
| Server | **Async + job queue** | Concurrent requests, warm models |
| Output | **Semantic chunks + structured JSON** | Production-ready RAG output |
| VRAM | **~8 GB** (AWQ + Flash Attn 2) | Fits on a T4 GPU |
| Batch Processing | **✅ /parse/batch** | Multiple files in parallel |
| Streaming | **✅ SSE streaming** | Progressive output for large docs |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Server                     │
│  /parse/document  /parse/image  /parse/audio  etc.  │
└─────────────────┬───────────────────────────────────┘
                  │
         ┌────────▼────────┐
         │   Async Job     │  ← Fire-and-forget or blocking mode
         │   Queue (ARQ)   │  ← N concurrent workers
         └────────┬────────┘
                  │
      ┌───────────┼─────────────────┐
      │           │                 │
  ┌───▼────┐ ┌────▼─────┐  ┌───────▼──────┐  ┌──────────┐
  │Docling │ │Qwen2-VL  │  │Distil-Whisper│  │Crawl4AI  │
  │(docs)  │ │AWQ (imgs)│  │+ pyannote    │  │(web)     │
  └────────┘ └──────────┘  └──────────────┘  └──────────┘
      │
  ┌───▼──────────────┐
  │ SemanticChunker  │  ← RAG-ready output with overlap
  └──────────────────┘
```

---

## Installation

### Requirements
- Python 3.10+
- Linux (required for some ML dependencies)
- NVIDIA GPU with 8+ GB VRAM recommended (runs on CPU with reduced performance)
- `ffmpeg` installed system-wide for video/audio processing

### Quick Start

```bash
git clone https://github.com/your-username/alchemy
cd alchemy

# Create virtual environment
conda create -n alchemy python=3.10
conda activate alchemy

# Install
pip install -e .

# Optional: diarization support (requires HuggingFace token)
pip install -e ".[diarization]"

# Pre-download model weights (recommended)
python download_models.py --all

# Start server
python server.py --all
```

### Docker (recommended for production)

```bash
# GPU
docker compose up

# CPU only
docker run -p 8000:8000 \
  -e WHISPER_COMPUTE_TYPE=int8 \
  -e DOCLING_DEVICE=cpu \
  alchemy
```

---

## Usage

### Server Flags

```bash
python server.py --host 0.0.0.0 --port 8000 --all

# Load only specific parsers
python server.py --documents      # Docling + Qwen2-VL
python server.py --media          # Distil-Whisper
python server.py --web            # Crawl4AI
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status and loaded models |
| POST | `/parse/document` | Parse PDF, DOCX, PPTX, HTML |
| POST | `/parse/document/stream` | Stream pages via SSE |
| POST | `/parse/image` | OCR, captioning, object detection |
| POST | `/parse/audio` | Transcribe with optional diarization |
| POST | `/parse/video` | Transcribe + optional keyframe extraction |
| POST | `/parse/web` | Crawl and parse web pages |
| POST | `/parse/web/batch` | Crawl multiple URLs concurrently |
| POST | `/parse/batch` | Process multiple files in parallel |
| GET | `/job/{job_id}` | Poll async job status |

Interactive docs: `http://localhost:8000/docs`

### Python SDK

```python
from alchemy_sdk import AlchemyClient
import asyncio

async def main():
    async with AlchemyClient("http://localhost:8000") as client:

        # Parse a PDF — get markdown + tables + RAG chunks
        result = await client.parse_document("research_paper.pdf")
        print(result.markdown[:500])
        print(f"Tables: {len(result.tables)}, Chunks: {len(result.chunks)}")

        # Transcribe audio with speaker labels
        result = await client.parse_audio("interview.mp3", diarize=True)
        print(result.markdown)  # [00:00:05 → 00:00:12] SPEAKER_00: Hello...

        # Caption an image
        result = await client.parse_image("chart.png", task="table_extraction")
        print(result.markdown)

        # Crawl a web page
        result = await client.parse_web("https://arxiv.org/abs/2501.00001")
        print(result.chunks[0].text)

        # Batch process multiple files
        results = await client.parse_batch(["a.pdf", "b.pdf", "c.pdf"])

asyncio.run(main())
```

### cURL Examples

```bash
# Parse a PDF
curl -X POST -F "file=@paper.pdf" http://localhost:8000/parse/document

# Transcribe audio with diarization
curl -X POST \
  -F "file=@interview.mp3" \
  -F "diarize=true" \
  http://localhost:8000/parse/audio

# Crawl a web page
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "max_depth": 2}' \
  http://localhost:8000/parse/web

# Async video processing
curl -X POST -F "file=@lecture.mp4" -F "async_mode=true" \
  http://localhost:8000/parse/video
# Returns: {"job_id": "abc-123", "status": "pending"}

# Poll job status
curl http://localhost:8000/job/abc-123
```

---

## Configuration

All settings can be overridden via environment variables or a `.env` file:

```env
# Server
MAX_WORKERS=4

# Document parser
DOCLING_TABLE_MODE=accurate        # fast | accurate
DOCLING_OCR_ENABLED=true

# Image parser
VISION_MODEL=Qwen/Qwen2-VL-7B-Instruct-AWQ
VISION_MAX_NEW_TOKENS=1024

# Audio parser
WHISPER_MODEL=distil-whisper/distil-large-v3
WHISPER_COMPUTE_TYPE=float16       # float16 | int8

# Diarization (optional)
DIARIZATION_ENABLED=true
HUGGINGFACE_TOKEN=hf_xxx

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64
SEMANTIC_CHUNKING=true
```

---

## Output Schema

All endpoints return a unified `ParseResponse`:

```json
{
  "job_id": "uuid",
  "status": "done",
  "result": {
    "source": "paper.pdf",
    "content_type": "document",
    "markdown": "# Introduction\n\nThis paper...",
    "chunks": [
      {
        "index": 0,
        "text": "Introduction section text...",
        "section": "Introduction",
        "tokens": 384
      }
    ],
    "tables": [
      {
        "caption": "Table 1: Results",
        "headers": ["Method", "Accuracy", "F1"],
        "rows": [["Ours", "94.2", "93.8"]],
        "markdown": "| Method | Accuracy | F1 |\n|..."
      }
    ],
    "metadata": {
      "num_pages": 12,
      "num_tables": 3,
      "language": "en"
    }
  }
}
```

---

## Models Used

| Parser | Model | Size | VRAM |
|---|---|---|---|
| Documents | Docling (IBM) | — | ~2 GB |
| Images | Qwen2-VL-7B-Instruct-AWQ | 7B (4-bit) | ~6 GB |
| Audio/Video | distil-whisper/distil-large-v3 | 756M | ~2 GB |
| Diarization | pyannote/speaker-diarization-3.1 | — | ~1 GB |
| Web | Crawl4AI (no model needed) | — | 0 |

Total with all models: ~8–9 GB VRAM (fits on T4)

---

## Roadmap

- [ ] LangChain / LlamaIndex / Haystack document loaders
- [ ] Redis-backed job persistence (survive server restarts)
- [ ] Structured schema extraction with Pydantic output validation
- [ ] Gradio UI
- [ ] Support for GOT-OCR 2.0 as alternative document parser
- [ ] Embedding endpoint for direct vector output

---

## License

GPL-3.0. See [LICENSE](LICENSE).

Note: Qwen2-VL is licensed under Apache 2.0. Docling is Apache 2.0. Distil-Whisper weights are MIT.
