"""
Pre-download all model weights before starting the server.
Run this once before deploying to avoid cold-start delays.

Usage:
    python download_models.py --all
    python download_models.py --documents --media
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download")


def download_documents():
    logger.info("📄 Downloading Docling models...")
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        logger.info("✅ Docling models downloaded.")
    except Exception as e:
        logger.error(f"Failed to download Docling models: {e}")

    logger.info("🖼️  Downloading Qwen2-VL-7B-Instruct-AWQ...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen2-VL-7B-Instruct-AWQ")
        logger.info("✅ Qwen2-VL downloaded.")
    except Exception as e:
        logger.error(f"Failed: {e}")


def download_media():
    logger.info("🎙️  Downloading Distil-Whisper Large-v3...")
    try:
        from faster_whisper import WhisperModel
        WhisperModel("distil-whisper/distil-large-v3", device="cpu", compute_type="int8")
        logger.info("✅ Distil-Whisper downloaded.")
    except Exception as e:
        logger.error(f"Failed: {e}")


def download_web():
    logger.info("🌐 Installing Crawl4AI browser...")
    try:
        import subprocess
        subprocess.run(["crawl4ai-setup"], check=True)
        logger.info("✅ Crawl4AI browser ready.")
    except Exception as e:
        logger.error(f"Failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download Alchemy model weights")
    parser.add_argument("--documents", action="store_true")
    parser.add_argument("--media", action="store_true")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.documents:
        download_documents()
    if args.all or args.media:
        download_media()
    if args.all or args.web:
        download_web()

    if not any(vars(args).values()):
        parser.print_help()
