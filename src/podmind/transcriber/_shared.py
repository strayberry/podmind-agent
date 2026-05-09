"""Shared types and utilities for the transcriber package.

Separated to avoid circular imports between __init__.py and backends/.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..config import TRANSCRIPTS_DIR, PodmindError, validate_episode_id
from ..io_utils import atomic_write as _atomic_write  # noqa: F401 — re-exported

# ------------------------------------------------------------------


def transcript_path(episode_id: str, *, backend: str | None = None) -> Path:
    """Return the transcript file path for an episode.

    When *backend* is None or ``"qwen"``, uses the plain path for backward
    compatibility.  Other backends get a suffix: ``{id}.{backend}.txt``.
    """
    validate_episode_id(episode_id)
    if backend and backend != "qwen":
        return TRANSCRIPTS_DIR / f"{episode_id}.{backend}.txt"
    return TRANSCRIPTS_DIR / f"{episode_id}.txt"


def _transcript_meta_path(episode_id: str, *, backend: str | None = None) -> Path:
    validate_episode_id(episode_id)
    if backend and backend != "qwen":
        return TRANSCRIPTS_DIR / f"{episode_id}.{backend}.meta.json"
    return TRANSCRIPTS_DIR / f"{episode_id}.meta.json"


def _file_sha256(path: str | Path) -> str:
    """Return hex digest of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _get_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", audio_path,
            ],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        raise PodmindError(
            "ffprobe not found — required for audio duration detection"
        ) from None
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------


@dataclass
class TranscribeProfile:
    """Timing profile for a transcription run.

    *settings* carries backend-agnostic configuration (chunk_seconds,
    batch_size, dtype, model, etc.).  *stages* carries per-stage wall-clock
    times (ffprobe, ffmpeg_split, etc.) — any backend can add its own keys.
    """

    model_load_seconds: float = 0.0
    total_audio_duration: float = 0.0
    chunk_count: int = 0
    chunk_transcribe_seconds: list[float] = field(default_factory=list)
    settings: dict[str, object] = field(default_factory=dict)
    stages: dict[str, float] = field(default_factory=dict)

    @property
    def total_transcribe_seconds(self) -> float:
        return sum(self.chunk_transcribe_seconds)

    @property
    def rtf(self) -> float:
        """Real-Time Factor: transcribe_time / audio_duration."""
        if self.total_audio_duration <= 0:
            return 0.0
        return self.total_transcribe_seconds / self.total_audio_duration

    def format(self) -> str:
        lines = ["--- Transcription Profile ---"]
        if self.model_load_seconds:
            lines.append(f"  Model load:     {self.model_load_seconds:.1f}s")
        for name, secs in self.stages.items():
            lines.append(f"  {name + ':':<15} {secs:.1f}s")
        if self.settings:
            parts = [f"{k}={v}" for k, v in self.settings.items()]
            lines.append(f"  Settings:       {', '.join(parts)}")
        if self.chunk_count:
            lines.append(f"  Chunks:         {self.chunk_count}")
        lines += [
            f"  Total transcribe: {self.total_transcribe_seconds:.1f}s",
            f"  Audio duration:   {self.total_audio_duration:.0f}s",
            f"  RTF:              {self.rtf:.3f}",
        ]
        if self.chunk_transcribe_seconds:
            chunks = self.chunk_transcribe_seconds
            lines.append(
                f"  Batch times:   min={min(chunks):.1f}s  max={max(chunks):.1f}s  "
                f"avg={sum(chunks) / len(chunks):.1f}s"
            )
            for i, t in enumerate(chunks):
                lines.append(f"    Batch {i + 1}/{len(chunks)}: {t:.1f}s")
        return "\n".join(lines)


@dataclass
class TranscriptResult:
    """Unified ASR result across all backends."""

    text: str
    segments: list[dict] = field(default_factory=list)
    backend: str = ""
    model: str = ""
    language: str | None = None
    audio_duration: float = 0.0
    profile: TranscribeProfile | None = None
    from_cache: bool = False
