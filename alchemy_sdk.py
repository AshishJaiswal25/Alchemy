"""
Alchemy Python SDK
------------------
A simple async client for the Alchemy API.

Usage:
    from alchemy_sdk import AlchemyClient

    async with AlchemyClient("http://localhost:8000") as client:
        result = await client.parse_document("report.pdf")
        print(result.markdown)

        result = await client.parse_audio("interview.mp3", diarize=True)
        print(result.markdown)

        result = await client.parse_web("https://example.com")
        print(result.chunks)
"""

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

import httpx


class AlchemyResult:
    def __init__(self, data: dict):
        self.job_id: str = data.get("job_id", "")
        self.status: str = data.get("status", "")
        self._result: Optional[dict] = data.get("result")
        self.error: Optional[str] = data.get("error")

    @property
    def markdown(self) -> Optional[str]:
        return self._result.get("markdown") if self._result else None

    @property
    def chunks(self) -> list[dict]:
        return self._result.get("chunks", []) if self._result else []

    @property
    def tables(self) -> list[dict]:
        return self._result.get("tables", []) if self._result else []

    @property
    def metadata(self) -> dict:
        return self._result.get("metadata", {}) if self._result else {}

    @property
    def raw(self) -> Any:
        return self._result.get("raw") if self._result else None

    def __repr__(self):
        snippet = (self.markdown[:80] + "...") if self.markdown and len(self.markdown) > 80 else self.markdown
        return f"AlchemyResult(status={self.status!r}, markdown={snippet!r})"


class AlchemyClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        await self._client.aclose()

    async def health(self) -> dict:
        r = await self._client.get("/health")
        r.raise_for_status()
        return r.json()

    # ── Documents ─────────────────────────────────────────────────────────────
    async def parse_document(
        self,
        path: str | Path,
        extract_tables: bool = True,
        extract_images: bool = True,
        output_format: str = "markdown",
        async_mode: bool = False,
    ) -> AlchemyResult:
        path = Path(path)
        with open(path, "rb") as f:
            content = f.read()
        files = {"file": (path.name, content, "application/octet-stream")}
        data = {
            "extract_tables": str(extract_tables).lower(),
            "extract_images": str(extract_images).lower(),
            "output_format": output_format,
            "async_mode": str(async_mode).lower(),
        }
        r = await self._client.post("/parse/document", files=files, data=data)
        r.raise_for_status()
        result = AlchemyResult(r.json())
        if async_mode:
            result = await self._poll(result.job_id)
        return result

    # ── Images ────────────────────────────────────────────────────────────────
    async def parse_image(
        self,
        path: str | Path,
        task: str = "detailed_caption",
        prompt: Optional[str] = None,
    ) -> AlchemyResult:
        path = Path(path)
        with open(path, "rb") as f:
            content = f.read()
        files = {"file": (path.name, content, "image/jpeg")}
        data = {"task": task}
        if prompt:
            data["prompt"] = prompt
        r = await self._client.post("/parse/image", files=files, data=data)
        r.raise_for_status()
        return AlchemyResult(r.json())

    # ── Audio ─────────────────────────────────────────────────────────────────
    async def parse_audio(
        self,
        path: str | Path,
        language: Optional[str] = None,
        diarize: bool = False,
    ) -> AlchemyResult:
        path = Path(path)
        with open(path, "rb") as f:
            content = f.read()
        files = {"file": (path.name, content, "audio/mpeg")}
        data = {"diarize": str(diarize).lower()}
        if language:
            data["language"] = language
        r = await self._client.post("/parse/audio", files=files, data=data)
        r.raise_for_status()
        return AlchemyResult(r.json())

    # ── Video ─────────────────────────────────────────────────────────────────
    async def parse_video(
        self,
        path: str | Path,
        language: Optional[str] = None,
        diarize: bool = False,
        extract_frames: bool = False,
    ) -> AlchemyResult:
        path = Path(path)
        with open(path, "rb") as f:
            content = f.read()
        files = {"file": (path.name, content, "video/mp4")}
        data = {
            "diarize": str(diarize).lower(),
            "extract_frames": str(extract_frames).lower(),
            "async_mode": "true",
        }
        if language:
            data["language"] = language
        r = await self._client.post("/parse/video", files=files, data=data)
        r.raise_for_status()
        result = AlchemyResult(r.json())
        return await self._poll(result.job_id)

    # ── Web ───────────────────────────────────────────────────────────────────
    async def parse_web(
        self,
        url: str,
        max_depth: int = 1,
        css_selector: Optional[str] = None,
    ) -> AlchemyResult:
        payload = {"url": url, "max_depth": max_depth}
        if css_selector:
            payload["css_selector"] = css_selector
        r = await self._client.post("/parse/web", json=payload)
        r.raise_for_status()
        return AlchemyResult(r.json())

    # ── Batch ─────────────────────────────────────────────────────────────────
    async def parse_batch(
        self,
        paths: list[str | Path],
        output_format: str = "markdown",
    ) -> list[AlchemyResult]:
        files = []
        for path in paths:
            path = Path(path)
            with open(path, "rb") as f:
                files.append(("files", (path.name, f.read(), "application/octet-stream")))
        data = {"output_format": output_format}
        r = await self._client.post("/parse/batch", files=files, data=data)
        r.raise_for_status()
        job_responses = r.json()
        # Poll all jobs concurrently
        results = await asyncio.gather(
            *[self._poll(j["job_id"]) for j in job_responses]
        )
        return list(results)

    # ── Polling ───────────────────────────────────────────────────────────────
    async def _poll(
        self,
        job_id: str,
        interval: float = 1.0,
        timeout: float = 300.0,
    ) -> AlchemyResult:
        start = time.time()
        while time.time() - start < timeout:
            r = await self._client.get(f"/job/{job_id}")
            r.raise_for_status()
            data = r.json()
            if data["status"] in ("done", "failed"):
                return AlchemyResult(data)
            await asyncio.sleep(interval)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
