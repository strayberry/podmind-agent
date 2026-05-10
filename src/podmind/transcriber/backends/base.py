"""Abstract ASR backend interface.

Backends are pure inference adapters — they do NOT handle caching,
file I/O, or episode metadata. That is the public layer's job.
"""

from abc import ABC, abstractmethod

from .._shared import TranscriptResult


class ASRBackend(ABC):
    """Abstract ASR backend.

    Subclasses implement model loading, language normalization,
    and raw transcription.  The public ``transcribe()`` function
    handles caching and file output.
    """

    name: str  # set by subclasses, e.g. "qwen-asr", "mlx-whisper"

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self, model_id: str, **kwargs: object) -> None:
        """Load/prepare the ASR model. Called once per session."""
        ...

    @abstractmethod
    def transcribe_raw(
        self,
        audio_path: str,
        language: str | None = None,
        *,
        profile: bool = False,
        **kwargs: object,
    ) -> TranscriptResult:
        """Run ASR on *audio_path* and return the result.

        No caching, no file I/O — pure model inference.  The caller is
        responsible for any caching or persistence.
        """
        ...

    @staticmethod
    @abstractmethod
    def normalize_language(lang: str | None) -> str | None:
        """Convert a unified language code to the backend's expected format."""
        ...

    @staticmethod
    def cache_extra_meta(**kwargs: object) -> dict | None:
        """Return backend-specific cache keys for the current settings."""
        return None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def model_load_seconds(self) -> float:
        """Time spent loading the model in the last ``load_model`` call."""
        return getattr(self, "_load_seconds", 0.0)
