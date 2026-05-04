"""Shared types and utilities for the transcriber package.

Separated to avoid circular imports between __init__.py and backends/.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..config import TRANSCRIPTS_DIR, PodmindError, ensure_dirs, validate_episode_id

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


def _atomic_write(path: Path, content: str) -> None:
    """Write content atomically via a temp file + rename."""
    ensure_dirs()
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp.")
    try:
        try:
            os.write(fd, content.encode("utf-8"))
        finally:
            os.close(fd)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------


@dataclass
class TranscribeProfile:
    """Timing profile for a transcription run."""

    model_load_seconds: float = 0.0
    ffprobe_seconds: float = 0.0
    ffmpeg_split_seconds: float = 0.0
    chunk_seconds_used: int = 600
    batch_size_used: int = 1
    chunk_count: int = 0
    chunk_transcribe_seconds: list[float] = field(default_factory=list)
    total_audio_duration: float = 0.0

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
        lines = [
            "--- Transcription Profile ---",
            f"  Model load:     {self.model_load_seconds:.1f}s",
            f"  ffprobe:        {self.ffprobe_seconds:.1f}s",
            f"  ffmpeg split:   {self.ffmpeg_split_seconds:.1f}s",
            f"  Chunk settings: {self.chunk_seconds_used}s x {self.chunk_count}"
            f" (batch={self.batch_size_used})",
            f"  Total transcribe: {self.total_transcribe_seconds:.1f}s",
            f"  Audio duration: {self.total_audio_duration:.0f}s",
            f"  RTF:            {self.rtf:.3f}",
        ]
        for i, t in enumerate(self.chunk_transcribe_seconds):
            lines.append(f"    Chunk {i + 1}/{self.chunk_count}: {t:.1f}s")
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
