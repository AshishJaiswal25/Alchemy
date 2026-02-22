"""
Async Job Queue
---------------
A lightweight in-process job queue backed by asyncio.
- Jobs are enqueued and processed by a pool of async workers.
- Workers stay alive and share warm model references from ModelManager.
- Supports both fire-and-forget (async_mode) and blocking (run_sync) usage.
"""

from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from alchemy.schemas import JobStatus

if TYPE_CHECKING:
    from alchemy.models.manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    task: str                            # parse_document | parse_image | parse_audio | parse_video | parse_web
    payload: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class JobQueue:
    def __init__(self, model_manager: "ModelManager", max_workers: int = 2):
        self._manager = model_manager
        self._max_workers = max_workers
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: dict[str, Job] = {}        # job_id → Job (in-memory store)
        self._workers: list[asyncio.Task] = []
        self._running = False

    async def start(self):
        self._running = True
        for i in range(self._max_workers):
            task = asyncio.create_task(self._worker(i), name=f"alchemy-worker-{i}")
            self._workers.append(task)
        logger.info(f"Job queue started with {self._max_workers} workers.")

    async def stop(self):
        self._running = False
        # Drain the queue
        for _ in self._workers:
            await self._queue.put(None)   # sentinel
        await asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("Job queue stopped.")

    async def enqueue(self, job: Job):
        self._jobs[job.id] = job
        await self._queue.put(job)

    async def run_sync(self, job: Job) -> Any:
        """Run a job immediately in the current event loop without queuing."""
        self._jobs[job.id] = job
        await self._execute(job)
        if job.error:
            raise RuntimeError(job.error)
        return job.result

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def queue_size(self) -> int:
        return self._queue.qsize()

    # ── Internal ──────────────────────────────────────────────────────────────
    async def _worker(self, worker_id: int):
        logger.debug(f"Worker {worker_id} started.")
        while self._running:
            try:
                job = await self._queue.get()
                if job is None:  # sentinel
                    break
                logger.info(f"[Worker {worker_id}] Processing job {job.id} ({job.task})")
                await self._execute(job)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Worker {worker_id} crashed: {e}")

    async def _execute(self, job: Job):
        job.status = JobStatus.RUNNING
        try:
            result = await self._dispatch(job)
            job.result = result
            job.status = JobStatus.DONE
        except Exception as e:
            logger.exception(f"Job {job.id} failed: {e}")
            job.error = str(e)
            job.status = JobStatus.FAILED
        finally:
            job.completed_at = time.time()

    async def _dispatch(self, job: Job) -> Any:
        p = job.payload
        match job.task:
            case "parse_document":
                return await self._manager.document_parser.parse(
                    content=p["content"],
                    filename=p["filename"],
                    extract_tables=p.get("extract_tables", True),
                    extract_images=p.get("extract_images", True),
                    output_format=p.get("output_format", "markdown"),
                )
            case "parse_image":
                return await self._manager.image_parser.parse(
                    content=p["content"],
                    filename=p["filename"],
                    task=p.get("task", "detailed_caption"),
                    prompt=p.get("prompt"),
                )
            case "parse_audio":
                return await self._manager.media_parser.parse_audio(
                    content=p["content"],
                    filename=p["filename"],
                    language=p.get("language"),
                    diarize=p.get("diarize", False),
                )
            case "parse_video":
                return await self._manager.media_parser.parse_video(
                    content=p["content"],
                    filename=p["filename"],
                    language=p.get("language"),
                    diarize=p.get("diarize", False),
                    extract_frames=p.get("extract_frames", False),
                )
            case "parse_web":
                return await self._manager.web_parser.parse(
                    url=p["url"],
                    max_depth=p.get("max_depth", 1),
                    css_selector=p.get("css_selector"),
                    extraction_schema=p.get("extraction_schema"),
                    headers=p.get("headers"),
                )
            case _:
                raise ValueError(f"Unknown task: {job.task}")
