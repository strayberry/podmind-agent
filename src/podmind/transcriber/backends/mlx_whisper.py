"""mlx-whisper ASR backend."""

import time

from ...config import get_language_iso
from .._shared import (
    TranscribeProfile,
    TranscriptResult,
    _get_duration,
)
from .base import ASRBackend


class MLXWhisperBackend(ASRBackend):
    """mlx-whisper backend via Apple MLX framework."""

    name = "mlx-whisper"

    def __init__(self) -> None:
        self._model_id: str = ""

    # ------------------------------------------------------------------
    # ASRBackend interface
    # ------------------------------------------------------------------

    def load_model(self, model_id: str, **kwargs: object) -> None:
        # mlx-whisper has no public preload API — it uses an internal
        # ModelHolder that caches the model in-process after the first
        # transcribe() call, so repeated calls with the same model_id
        # will reuse the cached model without reloading.
        self._model_id = model_id

    def transcribe_raw(
        self,
        audio_path: str,
        language: str | None = None,
        *,
        profile: bool = False,
        **kwargs: object,
    ) -> TranscriptResult:
        """Run mlx-whisper inference. No caching, no file I/O."""
        import mlx_whisper

        lang_code = self.normalize_language(language)
        t0 = time.perf_counter()
        raw = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self._model_id,
            language=lang_code,
            verbose=False,
            word_timestamps=False,
            hallucination_silence_threshold=2.0,
        )
        elapsed = time.perf_counter() - t0

        duration = _get_duration(audio_path)
        prof = TranscribeProfile(
            total_audio_duration=duration,
            chunk_count=1,
            settings={"model": self._model_id},
        ) if profile else None
        if prof:
            prof.chunk_transcribe_seconds = [elapsed]

        full_text = raw["text"].strip()

        return TranscriptResult(
            text=full_text,
            segments=raw.get("segments", []),
            backend=self.name,
            model=self._model_id,
            language=raw.get("language"),
            audio_duration=duration,
            profile=prof,
        )

    @staticmethod
    def normalize_language(lang: str | None) -> str | None:
        """Map full language names (e.g. 'Chinese') to ISO codes (e.g. 'zh')."""
        if lang is None:
            return None
        mapped = get_language_iso(lang)
        if mapped is not None:
            return mapped
        if len(lang) <= 3:
            return lang
        return None
