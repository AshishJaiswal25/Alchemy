"""
ImageParser — powered by Qwen2-VL (AWQ-quantised, ~6 GB VRAM)
-------------------------------------------------------------
Replaces Florence-2 base + Surya with a single, much more capable
vision-language model.

Supported tasks:
  ocr                — extract all text from the image
  caption            — one-sentence description
  detailed_caption   — multi-paragraph description with structure
  object_detection   — list objects with bounding boxes (JSON)
  table_extraction   — extract table data as markdown
  qa                 — answer a custom question (provide prompt=)
"""

from __future__ import annotations
import asyncio
import base64
import io
import json
import logging
from typing import Any, Optional

from alchemy.config import Settings
from alchemy.schemas import ParseResponse

logger = logging.getLogger(__name__)

# Task-specific system prompts
TASK_PROMPTS = {
    "ocr": (
        "Extract ALL text from this image exactly as it appears, preserving "
        "structure, line breaks, and formatting. Output only the extracted text."
    ),
    "caption": (
        "Describe this image in one concise sentence."
    ),
    "detailed_caption": (
        "Provide a detailed description of this image. Include: main subjects, "
        "setting, visible text, colors, spatial relationships, and any notable details. "
        "Use clear structured paragraphs."
    ),
    "object_detection": (
        "List every distinct object in this image. For each object output a JSON array "
        "entry with keys: label, confidence (0-1), description. "
        "Output ONLY valid JSON. Example: [{\"label\": \"cat\", \"confidence\": 0.95, \"description\": \"orange tabby cat sitting on a chair\"}]"
    ),
    "table_extraction": (
        "Extract the table(s) from this image. Output each table as GitHub-flavored "
        "markdown. If no table is present, reply with: NO_TABLE_FOUND"
    ),
}


class ImageParser:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._processor = None
        self._device = None

    async def initialize(self):
        await asyncio.get_event_loop().run_in_executor(None, self._load_model)

    def _load_model(self):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        device = self.settings.vision_device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        logger.info(f"Loading {self.settings.vision_model} on {device}...")

        # AWQ quantised model loads directly — no manual quantisation needed
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.settings.vision_model,
            torch_dtype="auto",
            device_map=device,
        )

        # Enable Flash Attention 2 if CUDA is available (free ~20% speedup)
        try:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.settings.vision_model,
                torch_dtype="auto",
                device_map=device,
                attn_implementation="flash_attention_2",
            )
            logger.info("Flash Attention 2 enabled.")
        except Exception:
            logger.info("Flash Attention 2 not available, using default attention.")

        self._processor = AutoProcessor.from_pretrained(self.settings.vision_model)
        logger.info("Qwen2-VL loaded.")

    async def parse(
        self,
        content: bytes,
        filename: str,
        task: str = "detailed_caption",
        prompt: Optional[str] = None,
    ) -> ParseResponse:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._parse_sync, content, filename, task, prompt
        )

    # Max tokens per task — keep OCR/caption fast, allow more for detailed tasks
    TASK_MAX_TOKENS = {
        "ocr": 512,
        "caption": 128,
        "detailed_caption": 1024,
        "object_detection": 512,
        "table_extraction": 768,
        "qa": 512,
    }

    # Max image dimension (longer side) — resize to keep VLM inference fast on CPU
    MAX_IMAGE_DIM = 1024

    def _resize_if_needed(self, image):
        """Downscale large images to keep VLM inference tractable on CPU."""
        from PIL import Image as PILImage
        w, h = image.size
        longest = max(w, h)
        if longest > self.MAX_IMAGE_DIM:
            scale = self.MAX_IMAGE_DIM / longest
            new_w, new_h = int(w * scale), int(h * scale)
            logger.info(f"Resizing image {w}x{h} → {new_w}x{new_h} for faster inference")
            image = image.resize((new_w, new_h), PILImage.LANCZOS)
        return image

    def _parse_sync(
        self,
        content: bytes,
        filename: str,
        task: str,
        prompt: Optional[str],
    ) -> ParseResponse:
        import time
        from PIL import Image

        t0 = time.perf_counter()

        image = Image.open(io.BytesIO(content)).convert("RGB")
        orig_w, orig_h = image.size
        image = self._resize_if_needed(image)

        # Build system prompt
        system_prompt = TASK_PROMPTS.get(task, TASK_PROMPTS["detailed_caption"])
        user_content = prompt if prompt else "Process this image."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{system_prompt}\n\n{user_content}"},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        max_tokens = min(
            self.TASK_MAX_TOKENS.get(task, 512),
            self.settings.vision_max_new_tokens,
        )
        logger.info(f"Generating up to {max_tokens} tokens for task={task} on {self._device}...")

        import torch
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        elapsed = time.perf_counter() - t0
        logger.info(f"Image inference done in {elapsed:.1f}s")

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        # Parse structured output for applicable tasks
        raw = None
        if task == "object_detection":
            try:
                raw = json.loads(output_text)
            except json.JSONDecodeError:
                raw = output_text

        return ParseResponse(
            source=filename,
            content_type="image",
            markdown=output_text,
            metadata={
                "task": task,
                "model": self.settings.vision_model,
                "original_size": f"{orig_w}x{orig_h}",
                "processed_size": f"{image.width}x{image.height}",
                "device": self._device,
                "inference_seconds": round(elapsed, 1),
            },
            raw=raw,
        )

    async def cleanup(self):
        if self._model:
            import torch
            del self._model
            del self._processor
            torch.cuda.empty_cache()
            self._model = None
            self._processor = None
