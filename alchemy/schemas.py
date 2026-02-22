"""Shared Pydantic schemas for requests and responses."""

from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, HttpUrl


# ─── Job ──────────────────────────────────────────────────────────────────────
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None


# ─── Parse Results ────────────────────────────────────────────────────────────
class TableData(BaseModel):
    caption: Optional[str] = None
    headers: list[str] = []
    rows: list[list[str]] = []
    markdown: str = ""


class ImageData(BaseModel):
    index: int
    caption: Optional[str] = None
    page: Optional[int] = None
    base64: Optional[str] = None   # only if extract_images=True


class DocumentChunk(BaseModel):
    index: int
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    tokens: Optional[int] = None


class ParseResponse(BaseModel):
    """Unified structured output for all parsed content."""
    source: str                          # filename or URL
    content_type: str                    # document | image | audio | video | web
    markdown: Optional[str] = None       # full markdown string
    chunks: list[DocumentChunk] = []     # semantic chunks ready for RAG
    tables: list[TableData] = []         # extracted tables
    images: list[ImageData] = []         # extracted images/figures
    metadata: dict[str, Any] = {}        # page count, language, duration, etc.
    raw: Optional[Any] = None            # full structured JSON (if output_format=json)


# ─── Web ──────────────────────────────────────────────────────────────────────
class WebParseRequest(BaseModel):
    url: str = Field(..., description="URL to parse")
    max_depth: int = Field(1, ge=1, le=5, description="Crawl depth for following links")
    include_links: bool = False
    css_selector: Optional[str] = None          # scope extraction to a selector
    extraction_schema: Optional[dict] = None    # JSON schema for structured extraction
    headers: Optional[dict[str, str]] = None    # custom HTTP headers
    async_mode: bool = False


# ─── Image Tasks ──────────────────────────────────────────────────────────────
class ImageTask(str, Enum):
    OCR = "ocr"
    CAPTION = "caption"
    DETAILED_CAPTION = "detailed_caption"
    OBJECT_DETECTION = "object_detection"
    TABLE_EXTRACTION = "table_extraction"
    QA = "qa"


class ProcessImageRequest(BaseModel):
    task: ImageTask = ImageTask.DETAILED_CAPTION
    prompt: Optional[str] = None
