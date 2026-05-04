"""Qwen3-ASR backend via qwen-asr + torch MPS."""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ...config import ASR_DEVICE, ASR_MODEL_ID, PodmindError
from .._shared import (
    TranscribeProfile,
    TranscriptResult,
    _file_sha256,
    _get_duration,
    _transcript_meta_path,
    transcript_path,
)
from .base import ASRBackend


def _split_audio(audio_path: str, tmp_dir: str, chunk_seconds: int = 600) -> list[Path]:
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


class QwenBackend(ASRBackend):
    """Qwen3-ASR backend via qwen-asr + torch MPS."""

    name = "qwen"

    def __init__(self) -> None:
        self._model: Any = None
        self._model_id: str = ""
        self._chunk_seconds: int = 600
        self._batch_size: int = 1

    # ------------------------------------------------------------------
    # ASRBackend interface
    # ------------------------------------------------------------------

    def load_model(self, model_id: str, **kwargs: object) -> None:
        self._model_id = model_id
        self._chunk_seconds = int(kwargs.get("chunk_seconds", 600))  # type: ignore[call-overload]
        self._batch_size = int(kwargs.get("batch_size", 1))  # type: ignore[call-overload]

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

        print(f"Loading Qwen3-ASR model (device={ASR_DEVICE})...")
        t0 = time.perf_counter()
        self._model = Qwen3ASRModel.from_pretrained(
            self._model_id,
            dtype=torch.bfloat16,
            device_map={"": ASR_DEVICE},
            max_new_tokens=4096,
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

        prof = TranscribeProfile(
            chunk_seconds_used=chunk_seconds,
            batch_size_used=batch_size,
        ) if profile else None

        # Duration via ffprobe
        t0 = time.perf_counter()
        duration = _get_duration(audio_path)
        if prof:
            prof.ffprobe_seconds = time.perf_counter() - t0
            prof.total_audio_duration = duration
        print(f"Audio duration: {duration:.0f}s ({duration / 60:.1f} min)")

        # Split + batch transcribe
        with tempfile.TemporaryDirectory(prefix="podmind_") as tmp_dir:
            t0 = time.perf_counter()
            chunks = _split_audio(audio_path, tmp_dir, chunk_seconds)
            if prof:
                prof.ffmpeg_split_seconds = time.perf_counter() - t0
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
                    per_chunk = elapsed / len(batch)
                    prof.chunk_transcribe_seconds.extend([per_chunk] * len(batch))

                for r in results:
                    full_text += r.text + "\n"

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
        """Qwen accepts full language names (e.g. 'Chinese') — pass through."""
        return lang


# ------------------------------------------------------------------
# Qwen-specific cache helpers (re-exported for backward compatibility).
# New code should use the public _check_transcript_cache / _build_meta
# in the transcriber package instead.
# ------------------------------------------------------------------


def _check_qwen_cache(
    episode_id: str,
    audio_path: str,
    language: str | None = None,
    force: bool = False,
    chunk_seconds: int = 600,
    batch_size: int = 1,
) -> str | None:
    if force:
        return None

    out_path = transcript_path(episode_id, backend="qwen")
    if not (out_path.exists() and out_path.stat().st_size > 0):
        return None

    meta_path = _transcript_meta_path(episode_id, backend="qwen")
    current_meta = {
        "audio_sha256": _file_sha256(audio_path),
        "language": language,
        "asr_model": ASR_MODEL_ID,
        "chunk_seconds": chunk_seconds,
        "batch_size": batch_size,
    }

    if meta_path.exists():
        try:
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            if saved == current_meta:
                print(f"Transcript already exists: {out_path}")
                return out_path.read_text(encoding="utf-8")
        except (json.JSONDecodeError, KeyError):
            return None
    else:
        print(
            f"Transcript already exists: {out_path} "
            "(no meta; re-run with --force to refresh)"
        )
        return out_path.read_text(encoding="utf-8")

    return None


def _build_qwen_meta(
    audio_path: str,
    language: str | None,
    chunk_seconds: int,
    batch_size: int,
) -> dict:
    return {
        "audio_sha256": _file_sha256(audio_path),
        "language": language,
        "asr_model": ASR_MODEL_ID,
        "chunk_seconds": chunk_seconds,
        "batch_size": batch_size,
    }
