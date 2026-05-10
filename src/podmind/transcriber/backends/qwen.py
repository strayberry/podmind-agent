"""Qwen3-ASR backend via qwen-asr + torch MPS."""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ...config import ASR_DEVICE, PodmindError
from .._shared import (
    TranscribeProfile,
    TranscriptResult,
    _get_duration,
)
from .base import ASRBackend


def _split_audio(audio_path: str, tmp_dir: str, chunk_seconds: int = 30) -> list[Path]:
    """Split audio into chunk_seconds segments, return list of WAV paths."""
    pattern = Path(tmp_dir) / "chunk_%03d.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", audio_path,
                "-f", "segment", "-segment_time", str(chunk_seconds),
                "-ar", "16000", "-ac", "1",
                str(pattern),
            ],
            check=True,
        )
    except FileNotFoundError:
        raise PodmindError("ffmpeg not found — required for audio splitting") from None
    return sorted(Path(tmp_dir).glob("chunk_*.wav"))


def _default_max_new_tokens(chunk_seconds: int) -> int:
    if chunk_seconds <= 60:
        return 512
    if chunk_seconds <= 120:
        return 1024
    if chunk_seconds <= 300:
        return 2048
    return 3072


class QwenBackend(ASRBackend):
    """Qwen3-ASR backend via qwen-asr + torch MPS."""

    name = "qwen-asr"

    def __init__(self) -> None:
        self._model: Any = None
        self._model_id: str = ""
        self._chunk_seconds: int = 30
        self._batch_size: int = 1
        self._max_new_tokens: int = 0
        self._dtype: str = "float16"

    # ------------------------------------------------------------------
    # ASRBackend interface
    # ------------------------------------------------------------------

    def load_model(self, model_id: str, **kwargs: object) -> None:
        self._model_id = model_id
        self._chunk_seconds = int(kwargs.get("chunk_seconds", 30))  # type: ignore[call-overload]
        self._batch_size = int(kwargs.get("batch_size", 1))  # type: ignore[call-overload]
        self._max_new_tokens = int(kwargs.get("max_new_tokens", 0) or 0)  # type: ignore[call-overload]
        raw_dtype = kwargs.get("dtype", "float16")
        self._dtype = str(raw_dtype) if raw_dtype else "float16"

        if ASR_DEVICE != "mps":
            raise PodmindError(
                f"Unsupported ASR_DEVICE={ASR_DEVICE!r}. "
                "Qwen backend only supports Apple Silicon (MPS)."
            )

        try:
            import torch
        except ImportError as e:
            raise PodmindError(
                "torch is not installed. Run: pip install torch\n"
                f"Original error: {e}"
            ) from e

        if not torch.backends.mps.is_available():
            raise PodmindError("MPS is required — Qwen backend only runs on Apple Silicon")

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            raise PodmindError(
                "qwen-asr is not installed. Run: pip install qwen-asr\n"
                f"Original error: {e}"
            ) from e

        dt = getattr(torch, self._dtype, torch.bfloat16)
        tokens = self._max_new_tokens or _default_max_new_tokens(self._chunk_seconds)
        print(f"Loading Qwen3-ASR model (device={ASR_DEVICE}, dtype={self._dtype}, "
              f"max_new_tokens={tokens})...")
        t0 = time.perf_counter()
        self._model = Qwen3ASRModel.from_pretrained(
            self._model_id,
            dtype=dt,
            device_map={"": ASR_DEVICE},
            max_new_tokens=tokens,
            max_inference_batch_size=self._batch_size,
        )
        self._load_seconds = time.perf_counter() - t0

    @property
    def model_load_seconds(self) -> float:
        return getattr(self, "_load_seconds", 0.0)

    def transcribe_raw(
        self,
        audio_path: str,
        language: str | None = None,
        *,
        profile: bool = False,
        **kwargs: object,
    ) -> TranscriptResult:
        """Run Qwen3-ASR inference. No caching, no file I/O."""
        chunk_seconds = int(kwargs.get("chunk_seconds", self._chunk_seconds))  # type: ignore[call-overload]
        batch_size = int(kwargs.get("batch_size", self._batch_size))  # type: ignore[call-overload]
        max_new_tokens = int(kwargs.get("max_new_tokens", 0) or 0)  # type: ignore[call-overload]
        tokens_limit = max_new_tokens or _default_max_new_tokens(chunk_seconds)

        prof = TranscribeProfile(
            settings={
                "chunk_seconds": chunk_seconds,
                "batch_size": batch_size,
                "max_new_tokens": tokens_limit,
                "dtype": self._dtype,
            },
        ) if profile else None

        # Duration via ffprobe
        t0 = time.perf_counter()
        duration = _get_duration(audio_path)
        if prof:
            prof.stages["ffprobe"] = time.perf_counter() - t0
            prof.total_audio_duration = duration
        print(f"Audio duration: {duration:.0f}s ({duration / 60:.1f} min)")

        # Split + batch transcribe
        with tempfile.TemporaryDirectory(prefix="podmind_") as tmp_dir:
            t0 = time.perf_counter()
            chunks = _split_audio(audio_path, tmp_dir, chunk_seconds)
            if prof:
                prof.stages["ffmpeg_split"] = time.perf_counter() - t0
                prof.chunk_count = len(chunks)
            print(f"Split into {len(chunks)} chunks ({chunk_seconds}s each)")

            full_text = ""
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                start_idx = i + 1
                end_idx = min(i + len(batch), len(chunks))
                print(f"Chunks {start_idx}-{end_idx}/{len(chunks)}...")

                t0 = time.perf_counter()
                results = self._model.transcribe(
                    audio=[str(c) for c in batch],
                    language=language,
                )
                elapsed = time.perf_counter() - t0

                if prof:
                    prof.chunk_transcribe_seconds.append(elapsed)

                for r in results:
                    full_text += r.text + "\n"
                    # Truncation detection: warn if output char count
                    # approaches max_new_tokens (rough proxy for token count).
                    if len(r.text) > tokens_limit * 0.8:
                        print(
                            f"WARNING: output ({len(r.text)} chars) may be truncated "
                            f"(max_new_tokens={tokens_limit}). "
                            f"Consider increasing --qwen-max-new-tokens."
                        )

        full_text = full_text.strip()

        if prof:
            prof.model_load_seconds = self.model_load_seconds

        return TranscriptResult(
            text=full_text,
            segments=[],
            backend=self.name,
            model=self._model_id,
            language=language,
            audio_duration=duration,
            profile=prof,
        )

    @staticmethod
    def normalize_language(lang: str | None) -> str | None:
        """Map ISO codes to full names for Qwen (e.g. 'zh' → 'Chinese')."""
        if lang is None:
            return None
        from ...config import get_language_full
        mapped = get_language_full(lang)
        return mapped if mapped is not None else lang

    @staticmethod
    def cache_extra_meta(**kwargs: object) -> dict:
        """Include Qwen generation settings in the transcript cache key."""
        chunk_seconds = int(kwargs.get("chunk_seconds", 30))  # type: ignore[call-overload]
        batch_size = int(kwargs.get("batch_size", 1))  # type: ignore[call-overload]
        max_new_tokens = int(kwargs.get("max_new_tokens", 0) or 0)  # type: ignore[call-overload]
        dtype = str(kwargs.get("dtype", "") or "float16")
        return {
            "chunk_seconds": chunk_seconds,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens or _default_max_new_tokens(chunk_seconds),
            "dtype": dtype,
        }
