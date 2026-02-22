"""
Alchemy — Quickstart Examples
------------------------------------
Run the server first:
    python server.py --all

Then run this script:
    python examples/quickstart.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alchemy_sdk import AlchemyClient


async def main():
    async with AlchemyClient("http://localhost:8000") as client:

        # ── 1. Health check ───────────────────────────────────────────────────
        health = await client.health()
        print("Server status:", health)

        # ── 2. Parse a PDF ────────────────────────────────────────────────────
        # result = await client.parse_document("your_file.pdf", extract_tables=True)
        # print("\n📄 Document markdown (first 500 chars):")
        # print(result.markdown[:500])
        # print(f"\n📊 Tables found: {len(result.tables)}")
        # print(f"🧩 Chunks for RAG: {len(result.chunks)}")

        # ── 3. Parse an image ─────────────────────────────────────────────────
        # result = await client.parse_image("chart.png", task="table_extraction")
        # print("\n🖼️  Extracted table:")
        # print(result.markdown)

        # ── 4. Transcribe audio with speaker labels ───────────────────────────
        # result = await client.parse_audio("interview.mp3", diarize=True)
        # print("\n🎙️  Diarized transcript:")
        # print(result.markdown[:500])

        # ── 5. Parse a web page ───────────────────────────────────────────────
        result = await client.parse_web("https://example.com")
        print("\n🌐 Web page markdown:")
        print(result.markdown[:500] if result.markdown else "No content")
        print(f"\nMetadata: {result.metadata}")

        # ── 6. Batch document processing ──────────────────────────────────────
        # results = await client.parse_batch(["doc1.pdf", "doc2.pdf", "doc3.pdf"])
        # for r in results:
        #     print(f"Parsed {r.metadata.get('filename')}: {len(r.chunks)} chunks")


if __name__ == "__main__":
    asyncio.run(main())
