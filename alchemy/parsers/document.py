"""
DocumentParser — powered by Docling (IBM Research)
---------------------------------------------------
Replaces the old Marker + Surya OCR pipeline with a single, unified
document understanding model that natively handles:
  - Multi-column PDFs
  - Tables (with proper row/column structure)
  - LaTeX equations
  - Figures with position metadata
  - DOCX, PPTX, HTML in addition to PDF

Output formats:
  - markdown  : clean, LLM-friendly markdown
  - json      : full structured DoclingDocument JSON
  - chunks    : semantic chunks ready to embed for RAG
"""

from __future__ import annotations
import asyncio
import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from alchemy.config import Settings
from alchemy.schemas import ParseResponse, TableData, DocumentChunk
from alchemy.utils.chunker import SemanticChunker

logger = logging.getLogger(__name__)


class DocumentParser:
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".html", ".htm", ".xlsx"}

    def __init__(self, settings: Settings):
        self.settings = settings
        self._converter = None
        self._chunker: Optional[SemanticChunker] = None

    async def initialize(self):
        """Load Docling converter in a thread to avoid blocking the event loop."""
        await asyncio.get_event_loop().run_in_executor(None, self._load_models)

    def _load_models(self):
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
        from docling.datamodel.base_models import InputFormat

        pdf_options = PdfPipelineOptions(
            do_ocr=self.settings.docling_ocr_enabled,
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,
                mode=self.settings.docling_table_mode,
            ),
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            }
        )

        if self.settings.semantic_chunking:
            self._chunker = SemanticChunker(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )

        logger.info("Docling converter loaded.")

    async def parse(
        self,
        content: bytes,
        filename: str,
        extract_tables: bool = True,
        extract_images: bool = True,
        output_format: str = "markdown",
    ) -> ParseResponse:
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._parse_sync,
            content,
            filename,
            extract_tables,
            extract_images,
            output_format,
        )

    def _parse_sync(
        self,
        content: bytes,
        filename: str,
        extract_tables: bool,
        extract_images: bool,
        output_format: str,
    ) -> ParseResponse:
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            result = self._converter.convert(tmp_path)
            doc = result.document

            # ── Markdown ──────────────────────────────────────────────────────
            markdown = doc.export_to_markdown()

            # ── Tables ────────────────────────────────────────────────────────
            tables: list[TableData] = []
            if extract_tables:
                for tbl in doc.tables:
                    try:
                        df = tbl.export_to_dataframe()
                        tables.append(TableData(
                            caption=tbl.caption_text(doc) if hasattr(tbl, "caption_text") else None,
                            headers=list(df.columns),
                            rows=df.values.tolist(),
                            markdown=df.to_markdown(index=False),
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to extract table: {e}")

            # ── Metadata ──────────────────────────────────────────────────────
            num_pages = None
            if hasattr(doc, 'num_pages'):
                np = doc.num_pages
                num_pages = np() if callable(np) else np
            metadata = {
                "num_pages": num_pages,
                "num_tables": len(tables),
                "filename": filename,
            }

            # ── Chunks ────────────────────────────────────────────────────────
            chunks: list[DocumentChunk] = []
            if self._chunker:
                chunks = self._chunker.chunk(markdown)

            # ── Build Response ────────────────────────────────────────────────
            raw = None
            if output_format == "json":
                raw = json.loads(doc.export_to_dict() if hasattr(doc, "export_to_dict") else "{}")

            return ParseResponse(
                source=filename,
                content_type="document",
                markdown=markdown,
                chunks=chunks,
                tables=tables,
                metadata=metadata,
                raw=raw,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    async def parse_streaming(
        self,
        content: bytes,
        filename: str,
        extract_tables: bool = True,
    ) -> AsyncIterator[str]:
        """Yield parsed page chunks as JSON strings for SSE streaming."""
        import json

        result = await asyncio.get_event_loop().run_in_executor(
            None, self._parse_sync, content, filename, extract_tables, False, "markdown"
        )

        # Yield chunk-by-chunk
        if result.chunks:
            for chunk in result.chunks:
                yield json.dumps({"type": "chunk", "data": chunk.model_dump()})
        else:
            # Fall back to yielding the full markdown in one shot
            yield json.dumps({"type": "markdown", "data": result.markdown})

        # Yield metadata at the end
        yield json.dumps({"type": "metadata", "data": result.metadata})

    async def cleanup(self):
        self._converter = None
        self._chunker = None
