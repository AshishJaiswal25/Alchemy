"""
MediaParser — powered by Distil-Whisper Large-v3 + pyannote diarization
------------------------------------------------------------------------
Replaces Whisper Small with:
  - Distil-Whisper Large-v3: ~6× faster than Whisper Large, near-identical WER
  - faster-whisper backend: CTranslate2-optimised, lower VRAM, int8 support
  - Optional pyannote-audio 3.x speaker diarization
  - Keyframe extraction + VLM captioning for video files
"""

from __future__ import annotations
import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from alchemy.config import Settings
from alchemy.schemas import ParseResponse

logger = logging.getLogger(__name__)


class MediaParser:
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"}
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

    def __init__(self, settings: Settings):
        self.settings = settings
        self._whisper = None
        self._diarizer = None

    async def initialize(self):
        await asyncio.get_event_loop().run_in_executor(None, self._load_models)

    def _load_models(self):
        from faster_whisper import WhisperModel
        import torch

        device = self.settings.whisper_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_type = self.settings.whisper_compute_type
        if device == "cpu":
            compute_type = "int8"   # float16 not supported on CPU

        logger.info(f"Loading {self.settings.whisper_model} on {device} ({compute_type})...")
        self._whisper = WhisperModel(
            self.settings.whisper_model,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Distil-Whisper loaded.")

        # Optional diarization
        if self.settings.diarization_enabled and self.settings.huggingface_token:
            try:
                from pyannote.audio import Pipeline
                self._diarizer = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.settings.huggingface_token,
                )
                if device == "cuda":
                    import torch
                    self._diarizer = self._diarizer.to(torch.device("cuda"))
                logger.info("Speaker diarization pipeline loaded.")
            except Exception as e:
                logger.warning(f"Diarization not available: {e}")

    # ── Audio ─────────────────────────────────────────────────────────────────
    async def parse_audio(
        self,
        content: bytes,
        filename: str,
        language: Optional[str] = None,
        diarize: bool = False,
    ) -> ParseResponse:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._transcribe, content, filename, language, diarize
        )

    # ── Video ─────────────────────────────────────────────────────────────────
    async def parse_video(
        self,
        content: bytes,
        filename: str,
        language: Optional[str] = None,
        diarize: bool = False,
        extract_frames: bool = False,
    ) -> ParseResponse:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._parse_video_sync, content, filename, language, diarize, extract_frames
        )

    # ── Internals ─────────────────────────────────────────────────────────────
    def _transcribe(
        self,
        content: bytes,
        filename: str,
        language: Optional[str],
        diarize: bool,
    ) -> ParseResponse:
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            segments, info = self._whisper.transcribe(
                tmp_path,
                language=language,
                beam_size=5,
                vad_filter=True,                    # skip silence
                vad_parameters={"min_silence_duration_ms": 500},
                batch_size=self.settings.whisper_batch_size,
            )

            segment_list = list(segments)   # materialise generator

            if diarize and self._diarizer:
                transcript = self._diarize_transcript(tmp_path, segment_list)
            else:
                transcript = self._plain_transcript(segment_list)

            # Full plain text
            full_text = " ".join(s.text.strip() for s in segment_list)

            return ParseResponse(
                source=filename,
                content_type="audio",
                markdown=transcript,
                metadata={
                    "language": info.language,
                    "language_probability": round(info.language_probability, 3),
                    "duration_seconds": round(info.duration, 2),
                    "num_segments": len(segment_list),
                    "model": self.settings.whisper_model,
                    "diarized": diarize and self._diarizer is not None,
                },
                raw={"full_text": full_text, "segments": [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in segment_list
                ]},
            )
        finally:
            os.unlink(tmp_path)

    def _plain_transcript(self, segments) -> str:
        lines = []
        for seg in segments:
            start = self._fmt_time(seg.start)
            end = self._fmt_time(seg.end)
            lines.append(f"[{start} → {end}] {seg.text.strip()}")
        return "\n".join(lines)

    def _diarize_transcript(self, audio_path: str, segments) -> str:
        """Merge Whisper segments with pyannote speaker labels."""
        diarization = self._diarizer(audio_path)
        lines = []
        for seg in segments:
            mid = (seg.start + seg.end) / 2
            speaker = "UNKNOWN"
            for turn, _, spk in diarization.itertracks(yield_label=True):
                if turn.start <= mid <= turn.end:
                    speaker = spk
                    break
            start = self._fmt_time(seg.start)
            end = self._fmt_time(seg.end)
            lines.append(f"[{start} → {end}] **{speaker}**: {seg.text.strip()}")
        return "\n".join(lines)

    def _parse_video_sync(
        self,
        content: bytes,
        filename: str,
        language: Optional[str],
        diarize: bool,
        extract_frames: bool,
    ) -> ParseResponse:
        """Extract audio from video, transcribe, and optionally caption keyframes."""
        import subprocess

        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            video_path = tmp.name

        audio_path = video_path + ".wav"
        try:
            # Extract audio with ffmpeg
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", audio_path],
                check=True,
                capture_output=True,
            )

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            result = self._transcribe(audio_bytes, filename, language, diarize)
            result.content_type = "video"

            # Keyframe extraction (uses ffmpeg + VLM caption via image parser if loaded)
            if extract_frames:
                frames_info = self._extract_keyframes(video_path)
                result.metadata["keyframes"] = frames_info

            return result
        finally:
            os.unlink(video_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def _extract_keyframes(self, video_path: str, fps: float = 0.1) -> list[dict]:
        """Extract one frame every 10 seconds using ffmpeg."""
        import subprocess, glob

        frame_dir = tempfile.mkdtemp()
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps={fps}",
                os.path.join(frame_dir, "frame_%04d.jpg"),
            ],
            capture_output=True,
        )

        frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        results = []
        for i, fpath in enumerate(frames):
            results.append({
                "index": i,
                "timestamp_approx_s": round(i / fps),
                "path": fpath,   # downstream can caption these via /parse/image
            })
        return results

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    async def cleanup(self):
        self._whisper = None
        self._diarizer = None
