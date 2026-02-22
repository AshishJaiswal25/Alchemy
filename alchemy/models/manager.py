"""
ModelManager — singleton that owns all model instances.
Models are loaded once at startup and kept warm in memory.
Supports lazy loading per modality.
"""

from __future__ import annotations
import logging
from typing import Optional, TYPE_CHECKING

from alchemy.config import Settings

if TYPE_CHECKING:
    from alchemy.parsers.document import DocumentParser
    from alchemy.parsers.image import ImageParser
    from alchemy.parsers.media import MediaParser
    from alchemy.parsers.web import WebParser

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._document_parser: Optional["DocumentParser"] = None
        self._image_parser: Optional["ImageParser"] = None
        self._media_parser: Optional["MediaParser"] = None
        self._web_parser: Optional["WebParser"] = None
        self._loaded: set[str] = set()

    async def initialize(self):
        """Load models according to feature flags."""
        if self.settings.load_documents:
            await self._load_document_parser()
        if self.settings.load_media:
            await self._load_media_parser()
        # Image parser shares the VLM loaded by document parser
        if self.settings.load_documents:
            await self._load_image_parser()
        if self.settings.load_web:
            await self._load_web_parser()

    # ── Lazy accessors ────────────────────────────────────────────────────────
    @property
    def document_parser(self) -> "DocumentParser":
        if not self._document_parser:
            raise RuntimeError("Document parser not loaded. Start server with --documents flag.")
        return self._document_parser

    @property
    def image_parser(self) -> "ImageParser":
        if not self._image_parser:
            raise RuntimeError("Image parser not loaded.")
        return self._image_parser

    @property
    def media_parser(self) -> "MediaParser":
        if not self._media_parser:
            raise RuntimeError("Media parser not loaded. Start server with --media flag.")
        return self._media_parser

    @property
    def web_parser(self) -> "WebParser":
        if not self._web_parser:
            raise RuntimeError("Web parser not loaded. Start server with --web flag.")
        return self._web_parser

    def loaded_models(self) -> list[str]:
        return list(self._loaded)

    # ── Loaders ───────────────────────────────────────────────────────────────
    async def _load_document_parser(self):
        logger.info("📄 Loading document parser (Docling)...")
        from alchemy.parsers.document import DocumentParser
        self._document_parser = DocumentParser(self.settings)
        await self._document_parser.initialize()
        self._loaded.add("docling")
        logger.info("✅ Document parser ready.")

    async def _load_image_parser(self):
        logger.info("🖼️  Loading image parser (Qwen2-VL)...")
        from alchemy.parsers.image import ImageParser
        self._image_parser = ImageParser(self.settings)
        await self._image_parser.initialize()
        self._loaded.add("qwen2-vl")
        logger.info("✅ Image parser ready.")

    async def _load_media_parser(self):
        logger.info("🎙️  Loading media parser (Distil-Whisper)...")
        from alchemy.parsers.media import MediaParser
        self._media_parser = MediaParser(self.settings)
        await self._media_parser.initialize()
        self._loaded.add("distil-whisper")
        logger.info("✅ Media parser ready.")

    async def _load_web_parser(self):
        logger.info("🌐 Loading web parser (Crawl4AI)...")
        from alchemy.parsers.web import WebParser
        self._web_parser = WebParser(self.settings)
        await self._web_parser.initialize()
        self._loaded.add("crawl4ai")
        logger.info("✅ Web parser ready.")

    async def cleanup(self):
        """Release GPU memory and close connections."""
        logger.info("Cleaning up model resources...")
        if self._document_parser:
            await self._document_parser.cleanup()
        if self._image_parser:
            await self._image_parser.cleanup()
        if self._media_parser:
            await self._media_parser.cleanup()
        if self._web_parser:
            await self._web_parser.cleanup()
