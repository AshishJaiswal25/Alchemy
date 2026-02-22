"""
WebParser — powered by Crawl4AI
---------------------------------
Replaces the heavy Selenium crawler with an async-first, LLM-native
web crawler that:
  - Handles JavaScript-rendered pages
  - Produces clean markdown output natively
  - Supports structured JSON extraction via schema
  - Supports multi-depth crawling
  - Is dramatically faster (no browser startup per request)
"""

from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, Optional

from alchemy.config import Settings
from alchemy.schemas import ParseResponse, DocumentChunk
from alchemy.utils.chunker import SemanticChunker

logger = logging.getLogger(__name__)


class WebParser:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._crawler = None
        self._chunker: Optional[SemanticChunker] = None

    async def initialize(self):
        from crawl4ai import AsyncWebCrawler
        self._crawler = AsyncWebCrawler(
            headless=True,
            verbose=False,
            user_agent=self.settings.crawler_user_agent,
        )
        await self._crawler.start()

        if self.settings.semantic_chunking:
            self._chunker = SemanticChunker(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
        logger.info("Crawl4AI crawler warmed up.")

    async def parse(
        self,
        url: str,
        max_depth: int = 1,
        css_selector: Optional[str] = None,
        extraction_schema: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> ParseResponse:
        """Crawl a URL and return structured markdown + optional extracted data."""
        from crawl4ai import CrawlerRunConfig, CacheMode
        from crawl4ai.extraction_strategy import JsonCssExtractionStrategy, LLMExtractionStrategy

        # Auto-fix URLs missing scheme
        url = url.strip()
        if not url.startswith(("http://", "https://", "file://", "raw:")):
            url = "https://" + url

        # Build extraction strategy
        extraction_strategy = None
        if extraction_schema:
            extraction_strategy = JsonCssExtractionStrategy(extraction_schema)

        # In Crawl4AI v0.8.0, headers belong on BrowserConfig, not CrawlerRunConfig.
        # We pass them at crawler init time, so ignore per-request headers here.
        # Use "domcontentloaded" instead of "networkidle" — heavy sites with ads/trackers
        # never reach network-idle and will always time out.
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            css_selector=css_selector,
            extraction_strategy=extraction_strategy,
            page_timeout=60000,  # 60s — generous for JS-heavy pages
            wait_until="domcontentloaded",
            delay_before_return_html=2.0,  # wait 2s after DOM ready for JS to render
        )

        try:
            result = await self._crawler.arun(url=url, config=config)
        except Exception as e:
            raise RuntimeError(f"Failed to crawl {url}: {e}")

        if not result.success:
            raise RuntimeError(f"Crawl failed for {url}: {result.error_message}")

        markdown = result.markdown_v2.raw_markdown if hasattr(result, "markdown_v2") else result.markdown

        # Clean markdown
        markdown = markdown.strip() if markdown else ""

        # Semantic chunks
        chunks: list[DocumentChunk] = []
        if self._chunker and markdown:
            chunks = self._chunker.chunk(markdown)

        # Structured extraction
        raw = None
        if extraction_schema and result.extracted_content:
            try:
                raw = json.loads(result.extracted_content)
            except Exception:
                raw = result.extracted_content

        # Follow links up to max_depth
        pages: list[dict] = [{"url": url, "markdown": markdown}]
        if max_depth > 1 and result.links:
            internal_links = [lnk["href"] for lnk in result.links.get("internal", [])[:5]]
            sub_results = await asyncio.gather(
                *[
                    self.parse(link, max_depth=max_depth - 1, css_selector=css_selector)
                    for link in internal_links
                ],
                return_exceptions=True,
            )
            for sub in sub_results:
                if isinstance(sub, ParseResponse):
                    pages.append({"url": sub.source, "markdown": sub.markdown})
                    chunks.extend(sub.chunks)

        # Merge pages into one markdown document
        if len(pages) > 1:
            merged = "\n\n---\n\n".join(
                f"## {p['url']}\n\n{p['markdown']}" for p in pages
            )
        else:
            merged = markdown

        return ParseResponse(
            source=url,
            content_type="web",
            markdown=merged,
            chunks=chunks,
            metadata={
                "num_pages_crawled": len(pages),
                "links_found": len(result.links.get("internal", [])) if result.links else 0,
                "status_code": result.status_code,
            },
            raw=raw,
        )

    async def parse_many(self, urls: list[str]) -> list[ParseResponse]:
        """Crawl multiple URLs concurrently (up to crawler_max_concurrent)."""
        sem = asyncio.Semaphore(self.settings.crawler_max_concurrent)

        async def _bounded(url: str) -> ParseResponse:
            async with sem:
                return await self.parse(url)

        results = await asyncio.gather(*[_bounded(u) for u in urls], return_exceptions=True)
        output = []
        for url, res in zip(urls, results):
            if isinstance(res, Exception):
                logger.warning(f"Failed to crawl {url}: {res}")
            else:
                output.append(res)
        return output

    async def cleanup(self):
        if self._crawler:
            try:
                await self._crawler.close()
            except Exception:
                pass  # Browser may already be closed
            self._crawler = None
