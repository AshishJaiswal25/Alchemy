"""
Alchemy — Modern Data Ingestion & Parsing Platform
Rebuilt with Docling, Distil-Whisper, Crawl4AI, and async-first architecture.
"""

import asyncio
import argparse
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from alchemy.config import Settings
from alchemy.models.manager import ModelManager
from alchemy.queue.worker import JobQueue, Job, JobStatus
from alchemy.schemas import (
    ParseResponse,
    JobResponse,
    JobStatusResponse,
    WebParseRequest,
    ProcessImageRequest,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("alchemy")

# ─── Settings & Global State ──────────────────────────────────────────────────
settings = Settings()
model_manager: Optional[ModelManager] = None
job_queue: Optional[JobQueue] = None


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, job_queue

    logger.info("🚀 Starting Alchemy...")
    model_manager = ModelManager(settings)
    await model_manager.initialize()

    job_queue = JobQueue(model_manager, max_workers=settings.max_workers)
    await job_queue.start()

    logger.info("✅ Alchemy is ready.")
    yield

    logger.info("🛑 Shutting down Alchemy...")
    await job_queue.stop()
    await model_manager.cleanup()


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Alchemy",
    description="Transform any data into structured, LLM-ready output. Documents, images, audio, video, and web — all in one place.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static UI ────────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Alchemy API — visit /docs for API documentation"})


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "models_loaded": model_manager.loaded_models() if model_manager else [],
        "queue_size": job_queue.queue_size() if job_queue else 0,
    }


# ─── Job Status ───────────────────────────────────────────────────────────────
@app.get("/job/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Poll the status of an async parse job."""
    if not job_queue:
        raise HTTPException(503, "Server not ready")
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        result=job.result,
        error=job.error,
    )


# ─── Document Endpoints ───────────────────────────────────────────────────────
@app.post("/parse/document", response_model=JobResponse, tags=["Documents"])
async def parse_document(
    file: UploadFile = File(...),
    extract_tables: bool = Form(True),
    extract_images: bool = Form(True),
    output_format: str = Form("markdown"),  # markdown | json | chunks
    async_mode: bool = Form(False),
):
    """
    Parse PDF, DOCX, PPTX, or other document formats.
    Uses Docling for high-quality structured extraction.
    Set async_mode=true to get a job_id and poll for results.
    """
    _assert_ready()
    content = await file.read()
    job = Job(
        id=str(uuid.uuid4()),
        task="parse_document",
        payload={
            "content": content,
            "filename": file.filename,
            "content_type": file.content_type,
            "extract_tables": extract_tables,
            "extract_images": extract_images,
            "output_format": output_format,
        },
    )
    if async_mode:
        await job_queue.enqueue(job)
        return JobResponse(job_id=job.id, status=JobStatus.PENDING)
    else:
        result = await job_queue.run_sync(job)
        return JobResponse(job_id=job.id, status=JobStatus.DONE, result=result)


@app.post("/parse/document/stream", tags=["Documents"])
async def parse_document_stream(
    file: UploadFile = File(...),
    extract_tables: bool = Form(True),
):
    """Stream parsed pages back as Server-Sent Events (SSE)."""
    _assert_ready()
    content = await file.read()

    async def event_stream():
        async for chunk in model_manager.document_parser.parse_streaming(
            content, file.filename, extract_tables=extract_tables
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Image Endpoints ──────────────────────────────────────────────────────────
@app.post("/parse/image", response_model=JobResponse, tags=["Images"])
async def parse_image(
    file: UploadFile = File(...),
    task: str = Form("detailed_caption"),
    prompt: Optional[str] = Form(None),
    async_mode: bool = Form(False),
):
    """
    Parse images using Qwen2-VL or InternVL2.
    
    Tasks: ocr | caption | detailed_caption | object_detection | table_extraction | qa
    """
    _assert_ready()
    content = await file.read()
    job = Job(
        id=str(uuid.uuid4()),
        task="parse_image",
        payload={
            "content": content,
            "filename": file.filename,
            "task": task,
            "prompt": prompt,
        },
    )
    if async_mode:
        await job_queue.enqueue(job)
        return JobResponse(job_id=job.id, status=JobStatus.PENDING)
    result = await job_queue.run_sync(job)
    return JobResponse(job_id=job.id, status=JobStatus.DONE, result=result)


# ─── Media Endpoints ──────────────────────────────────────────────────────────
@app.post("/parse/audio", response_model=JobResponse, tags=["Media"])
async def parse_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(False),
    async_mode: bool = Form(False),
):
    """
    Transcribe audio using Distil-Whisper Large-v3.
    Enable diarize=true for speaker-labeled transcripts.
    """
    _assert_ready()
    content = await file.read()
    job = Job(
        id=str(uuid.uuid4()),
        task="parse_audio",
        payload={
            "content": content,
            "filename": file.filename,
            "language": language,
            "diarize": diarize,
        },
    )
    if async_mode:
        await job_queue.enqueue(job)
        return JobResponse(job_id=job.id, status=JobStatus.PENDING)
    result = await job_queue.run_sync(job)
    return JobResponse(job_id=job.id, status=JobStatus.DONE, result=result)


@app.post("/parse/video", response_model=JobResponse, tags=["Media"])
async def parse_video(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarize: bool = Form(False),
    extract_frames: bool = Form(False),
    async_mode: bool = Form(True),
):
    """
    Transcribe video audio and optionally extract & caption keyframes.
    Defaults to async mode given typical video file sizes.
    """
    _assert_ready()
    content = await file.read()
    job = Job(
        id=str(uuid.uuid4()),
        task="parse_video",
        payload={
            "content": content,
            "filename": file.filename,
            "language": language,
            "diarize": diarize,
            "extract_frames": extract_frames,
        },
    )
    if async_mode:
        await job_queue.enqueue(job)
        return JobResponse(job_id=job.id, status=JobStatus.PENDING)
    result = await job_queue.run_sync(job)
    return JobResponse(job_id=job.id, status=JobStatus.DONE, result=result)


# ─── Web Endpoints ────────────────────────────────────────────────────────────
@app.post("/parse/web", response_model=JobResponse, tags=["Web"])
async def parse_web(request: WebParseRequest):
    """
    Parse web pages using Crawl4AI async crawler.
    Supports JS-rendered pages, auth headers, and custom extraction schemas.
    """
    _assert_ready()
    job = Job(
        id=str(uuid.uuid4()),
        task="parse_web",
        payload=request.model_dump(),
    )
    if request.async_mode:
        await job_queue.enqueue(job)
        return JobResponse(job_id=job.id, status=JobStatus.PENDING)
    result = await job_queue.run_sync(job)
    return JobResponse(job_id=job.id, status=JobStatus.DONE, result=result)


@app.post("/parse/web/batch", response_model=list[JobResponse], tags=["Web"])
async def parse_web_batch(urls: list[str], max_depth: int = 1):
    """Crawl multiple URLs concurrently."""
    _assert_ready()
    jobs = [
        Job(
            id=str(uuid.uuid4()),
            task="parse_web",
            payload={"url": url, "max_depth": max_depth, "async_mode": True},
        )
        for url in urls
    ]
    for job in jobs:
        await job_queue.enqueue(job)
    return [JobResponse(job_id=j.id, status=JobStatus.PENDING) for j in jobs]


# ─── Batch Document Endpoint ──────────────────────────────────────────────────
@app.post("/parse/batch", response_model=list[JobResponse], tags=["Documents"])
async def parse_batch(
    files: list[UploadFile] = File(...),
    output_format: str = Form("markdown"),
):
    """Submit multiple files for parallel processing."""
    _assert_ready()
    jobs = []
    for file in files:
        content = await file.read()
        job = Job(
            id=str(uuid.uuid4()),
            task="parse_document",
            payload={
                "content": content,
                "filename": file.filename,
                "content_type": file.content_type,
                "output_format": output_format,
                "extract_tables": True,
                "extract_images": True,
            },
        )
        await job_queue.enqueue(job)
        jobs.append(job)
    return [JobResponse(job_id=j.id, status=JobStatus.PENDING) for j in jobs]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _assert_ready():
    if not model_manager or not job_queue:
        raise HTTPException(503, "Server is still initializing. Try again shortly.")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alchemy Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--documents", action="store_true", help="Load document models")
    parser.add_argument("--media", action="store_true", help="Load media models")
    parser.add_argument("--web", action="store_true", help="Enable web crawler")
    parser.add_argument("--all", action="store_true", help="Load all models")
    parser.add_argument("--reload", action="store_true", help="Dev hot-reload")
    args = parser.parse_args()

    # Inject CLI flags into environment so Settings picks them up
    import os
    if args.all or args.documents:
        os.environ["LOAD_DOCUMENTS"] = "true"
    if args.all or args.media:
        os.environ["LOAD_MEDIA"] = "true"
    if args.all or args.web:
        os.environ["LOAD_WEB"] = "true"
    settings.max_workers = args.workers

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
