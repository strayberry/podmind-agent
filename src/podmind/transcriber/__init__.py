"""PodMind ASR transcription — pluggable backend architecture.

The public layer owns all caching and file I/O.  Backends only do
pure inference via ``transcribe_raw()``.
"""

import json
import warnings
from pathlib import Path

from ..config import PodmindError
from ._shared import (  # noqa: F401 — re-exported for backward compat
    TranscribeProfile,
    TranscriptResult,
    _atomic_write,
    _file_sha256,
    _get_duration,
    _transcript_meta_path,
    transcript_path,
)
from .backends.base import ASRBackend

warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_DEFAULT_BACKEND = "mlx-whisper"
_DEFAULT_MODELS: dict[str, str] = {
    "qwen": "Qwen/Qwen3-ASR-0.6B",
    "mlx-whisper": "mlx-community/whisper-turbo",
}


def _get_backend_class(name: str):
    """Import and return a backend class by name (lazy)."""
    if name == "qwen":
        from .backends.qwen import QwenBackend
        return QwenBackend
    if name == "mlx-whisper":
        from .backends.mlx_whisper import MLXWhisperBackend
        return MLXWhisperBackend
    raise PodmindError(f"Unknown ASR backend: {name!r}")


# ---------------------------------------------------------------------------
# Shared cache helpers (the ONLY place that reads/writes cache files)
# ---------------------------------------------------------------------------


def _build_meta(
    audio_path: str,
    language: str | None,
    *,
    backend: str,
    backend_model: str,
    extra_meta: dict | None = None,
) -> dict:
    """Build a cache-meta dict.  *extra_meta* carries backend-specific fields."""
    meta: dict = {
        "backend": backend,
        "backend_model": backend_model,
        "language": language,
        "audio_sha256": _file_sha256(audio_path),
    }
    if extra_meta:
        meta.update(extra_meta)
    return meta


def _qwen_extra_meta_match(saved: dict, extra_meta: dict) -> bool:
    """Check that old-format Qwen meta params match the current request.

    Only compares keys present in BOTH dicts — a missing key in old meta
    means it predates that feature, so we accept the default.
    """
    for key in ("chunk_seconds", "batch_size", "max_new_tokens", "dtype"):
        if key in saved and key in extra_meta and saved[key] != extra_meta[key]:
            return False
    return True


def _check_transcript_cache(
    episode_id: str,
    audio_path: str,
    *,
    backend: str,
    backend_model: str,
    language: str | None = None,
    force: bool = False,
    extra_meta: dict | None = None,
) -> str | None:
    """Return cached transcript text if valid, or None if cache miss."""
    if force:
        return None

    out_path = transcript_path(episode_id, backend=backend)
    if not (out_path.exists() and out_path.stat().st_size > 0):
        return None

    meta_path = _transcript_meta_path(episode_id, backend=backend)
    current_meta = _build_meta(
        audio_path, language,
        backend=backend, backend_model=backend_model,
        extra_meta=extra_meta,
    )

    if meta_path.exists():
        try:
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            if saved == current_meta:
                print(f"Transcript already exists: {out_path}")
                return out_path.read_text(encoding="utf-8")
            # Old-format Qwen cache (pre-pluggable-backend): meta has
            # "asr_model" instead of "backend"/"backend_model".
            if (backend == "qwen"
                    and "backend" not in saved
                    and saved.get("audio_sha256") == current_meta.get("audio_sha256")
                    and saved.get("language") == current_meta.get("language")
                    and saved.get("asr_model") == backend_model
                    and _qwen_extra_meta_match(saved, extra_meta or {})):
                # Migrate to new format in-place
                _atomic_write(
                    meta_path,
                    json.dumps(current_meta, ensure_ascii=False),
                )
                print(f"Transcript already exists: {out_path}")
                return out_path.read_text(encoding="utf-8")
        except (json.JSONDecodeError, KeyError):
            return None
    else:
        # Only accept missing meta for legacy plain-path (qwen / no backend).
        # Suffixed backends (mlx-whisper) must have a matching meta to hit cache.
        if not backend or backend == "qwen":
            print(
                f"Transcript already exists: {out_path} "
                "(no meta; re-run with --force to refresh)"
            )
            return out_path.read_text(encoding="utf-8")
        return None

    return None


def _write_transcript_cache(
    episode_id: str,
    result: TranscriptResult,
    audio_path: str,
    language: str | None,
    extra_meta: dict | None = None,
) -> None:
    """Persist transcription result and its cache metadata."""
    out_path = transcript_path(episode_id, backend=result.backend)
    meta_path = _transcript_meta_path(episode_id, backend=result.backend)
    _atomic_write(out_path, result.text)
    current_meta = _build_meta(
        audio_path, language,
        backend=result.backend, backend_model=result.model,
        extra_meta=extra_meta,
    )
    _atomic_write(meta_path, json.dumps(current_meta, ensure_ascii=False))
    print(f"Transcript saved: {out_path} ({len(result.text)} chars)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transcribe(
    episode_id: str,
    audio_path: str | Path,
    language: str | None = None,
    force: bool = False,
    chunk_seconds: int = 30,
    batch_size: int = 1,
    profile: bool = False,
    *,
    backend: str = _DEFAULT_BACKEND,
    backend_model: str | None = None,
    max_new_tokens: int = 0,
    dtype: str = "",
) -> str:
    """Transcribe audio using the selected ASR backend.

    Cache is checked **before** model loading — cache hits return
    immediately without importing heavy dependencies.
    """
    audio_path = str(audio_path)
    model_id = backend_model or _DEFAULT_MODELS.get(backend, "")
    backend_cls = _get_backend_class(backend)

    # Normalize language for this backend
    lang = backend_cls.normalize_language(language)

    # Build backend-specific extra meta for cache key
    if backend == "qwen":
        from .backends.qwen import _default_max_new_tokens
        resolved_tokens = max_new_tokens or _default_max_new_tokens(chunk_seconds)
        extra_meta: dict | None = {
            "chunk_seconds": chunk_seconds,
            "batch_size": batch_size,
            "max_new_tokens": resolved_tokens,
            "dtype": dtype or "float16",
        }
    else:
        extra_meta = None

    # --- cache check (no model loaded yet) ---
    cached = _check_transcript_cache(
        episode_id, audio_path,
        backend=backend, backend_model=model_id,
        language=lang, force=force,
        extra_meta=extra_meta,
    )
    if cached is not None:
        return cached

    # --- load model ---
    be = _instantiate_backend(
        backend, model_id, chunk_seconds, batch_size,
        max_new_tokens=max_new_tokens, dtype=dtype,
    )

    # --- transcribe (pure inference, no cache / no I/O) ---
    result = be.transcribe_raw(
        audio_path,
        language=lang,
        profile=profile,
        chunk_seconds=chunk_seconds,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    # --- cache write (single place for all persistence) ---
    _write_transcript_cache(episode_id, result, audio_path, lang, extra_meta)

    if profile and result.profile:
        print(result.profile.format())

    return result.text


class ASRSession:
    """A lightweight session that loads the model once and processes episodes.

    The model is loaded lazily on first use — creating a session is cheap
    so batch-transcribe can check caches before spending time on model load.
    """

    def __init__(
        self,
        backend: str,
        model_id: str,
        language: str | None,
        *,
        chunk_seconds: int = 30,
        batch_size: int = 1,
        max_new_tokens: int = 0,
        dtype: str = "",
    ) -> None:
        self._backend = backend
        self._model_id = model_id
        self._language = _get_backend_class(backend).normalize_language(language)
        self._chunk_seconds = chunk_seconds
        self._batch_size = batch_size
        self._max_new_tokens = max_new_tokens
        self._dtype = dtype or "float16"
        self._be: ASRBackend | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def transcribe_episode(
        self,
        episode_id: str,
        audio_path: str,
        *,
        force: bool = False,
        profile: bool = False,
    ) -> TranscriptResult:
        """Transcribe one episode (cache check → transcribe_raw → write)."""
        extra_meta = self._extra_meta()
        cached = _check_transcript_cache(
            episode_id, audio_path,
            backend=self._backend, backend_model=self._model_id,
            language=self._language, force=force,
            extra_meta=extra_meta,
        )
        if cached is not None:
            return TranscriptResult(
                text=cached, backend=self._backend, model=self._model_id,
                language=self._language, from_cache=True,
            )

        self._ensure_loaded()
        result = self.transcribe_raw(audio_path, profile=profile)
        _write_transcript_cache(episode_id, result, audio_path,
                                self._language, extra_meta)
        return result

    def transcribe_raw(
        self, audio_path: str, *, profile: bool = False,
    ) -> TranscriptResult:
        """Run inference (no caching)."""
        be = self._ensure_loaded()
        return be.transcribe_raw(
            audio_path,
            language=self._language,
            profile=profile,
            chunk_seconds=self._chunk_seconds,
            batch_size=self._batch_size,
            max_new_tokens=self._max_new_tokens,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> ASRBackend:
        if self._be is None:
            self._be = _instantiate_backend(
                self._backend, self._model_id,
                self._chunk_seconds, self._batch_size,
                max_new_tokens=self._max_new_tokens, dtype=self._dtype,
            )
        return self._be

    def _extra_meta(self) -> dict | None:
        if self._backend == "qwen":
            from .backends.qwen import _default_max_new_tokens
            resolved = self._max_new_tokens or _default_max_new_tokens(self._chunk_seconds)
            return {
                "chunk_seconds": self._chunk_seconds,
                "batch_size": self._batch_size,
                "max_new_tokens": resolved,
                "dtype": self._dtype,
            }
        return None


def _instantiate_backend(
    backend: str,
    model_id: str,
    chunk_seconds: int = 30,
    batch_size: int = 1,
    max_new_tokens: int = 0,
    dtype: str = "",
):
    """Create and load a backend instance.  Factored out for batch use."""
    backend_cls = _get_backend_class(backend)
    be = backend_cls()
    be.load_model(
        model_id,
        chunk_seconds=chunk_seconds,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
    )
    return be
