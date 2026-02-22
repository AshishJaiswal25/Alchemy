"""
SemanticChunker
---------------
Splits markdown/text into semantically coherent chunks for RAG embedding.

Strategy:
  1. Split on markdown headings to preserve document structure.
  2. If a section exceeds chunk_size tokens, further split on paragraphs.
  3. Apply sliding window overlap so context isn't cut off at boundaries.
  4. Attach section heading as metadata for better retrieval context.
"""

from __future__ import annotations
import re
from typing import Optional

from alchemy.schemas import DocumentChunk


class SemanticChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[DocumentChunk]:
        if not text or not text.strip():
            return []

        sections = self._split_by_headings(text)
        chunks: list[DocumentChunk] = []
        index = 0

        for section_title, section_text in sections:
            token_count = self._estimate_tokens(section_text)

            if token_count <= self.chunk_size:
                chunks.append(DocumentChunk(
                    index=index,
                    text=section_text.strip(),
                    section=section_title,
                    tokens=token_count,
                ))
                index += 1
            else:
                # Split large sections into overlapping sub-chunks
                sub_chunks = self._sliding_window(section_text)
                for sub in sub_chunks:
                    chunks.append(DocumentChunk(
                        index=index,
                        text=sub.strip(),
                        section=section_title,
                        tokens=self._estimate_tokens(sub),
                    ))
                    index += 1

        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[Optional[str], str]]:
        """Split text on markdown headings (# / ## / ###)."""
        heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [(None, text)]

        sections: list[tuple[Optional[str], str]] = []

        # Text before the first heading
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

        for i, match in enumerate(matches):
            heading = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((heading, body))

        return sections

    def _sliding_window(self, text: str) -> list[str]:
        """Split long text into overlapping windows at paragraph boundaries."""
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        windows: list[str] = []
        current_tokens = 0
        current_paras: list[str] = []

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)

            if current_tokens + para_tokens > self.chunk_size and current_paras:
                windows.append("\n\n".join(current_paras))
                # Overlap: keep last N tokens worth of paragraphs
                overlap_paras: list[str] = []
                overlap_tokens = 0
                for p in reversed(current_paras):
                    t = self._estimate_tokens(p)
                    if overlap_tokens + t > self.chunk_overlap:
                        break
                    overlap_paras.insert(0, p)
                    overlap_tokens += t
                current_paras = overlap_paras
                current_tokens = overlap_tokens

            current_paras.append(para)
            current_tokens += para_tokens

        if current_paras:
            windows.append("\n\n".join(current_paras))

        return windows

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token (GPT-family heuristic)."""
        return max(1, len(text) // 4)
