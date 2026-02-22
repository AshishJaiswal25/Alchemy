"""
Centralised configuration via pydantic-settings.
All values can be overridden with environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    max_workers: int = 2          # async job-queue concurrency

    # ── Feature Flags ─────────────────────────────────────────────────────────
    load_documents: bool = True
    load_media: bool = True
    load_web: bool = True

    # ── Document Parser (Docling) ──────────────────────────────────────────────
    docling_device: str = "auto"  # auto | cpu | cuda | mps
    docling_ocr_enabled: bool = True
    docling_table_mode: str = "accurate"  # fast | accurate

    # ── Image Parser (Qwen2-VL) ───────────────────────────────────────────────
    vision_model: str = "Qwen/Qwen2-VL-7B-Instruct-AWQ"   # AWQ-quantised, ~6 GB VRAM
    vision_device: str = "auto"
    vision_max_new_tokens: int = 1024

    # ── Audio/Video (Distil-Whisper) ──────────────────────────────────────────
    whisper_model: str = "distil-whisper/distil-large-v3"
    whisper_device: str = "auto"
    whisper_compute_type: str = "float16"   # float16 | int8
    whisper_batch_size: int = 16

    # ── Diarization (pyannote) ────────────────────────────────────────────────
    diarization_enabled: bool = False       # requires HuggingFace token
    huggingface_token: Optional[str] = None

    # ── Web Crawler (Crawl4AI) ────────────────────────────────────────────────
    crawler_max_concurrent: int = 5
    crawler_timeout: int = 30
    crawler_user_agent: str = (
        "Mozilla/5.0 (compatible; AlchemyBot/2.0; +https://github.com/your-username/alchemy)"
    )

    # ── Output / Chunking ─────────────────────────────────────────────────────
    default_output_format: str = "markdown"    # markdown | json | chunks
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_chunking: bool = True

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
