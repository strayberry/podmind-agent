"""MLX Qwen3-ASR backend via mlx-audio (4-bit quantized)."""

import time

from ...config import PodmindError, get_language_full
from .._shared import (
    TranscribeProfile,
    TranscriptResult,
    _get_duration,
)
from .base import ASRBackend


class MLXQwenBackend(ASRBackend):
    """Qwen3-ASR 4-bit via mlx-audio — Apple MLX framework."""

    name = "mlx-qwen"

    def __init__(self) -> None:
        self._model: object = None
        self._model_id: str = ""

    # ------------------------------------------------------------------
    # ASRBackend interface
    # ------------------------------------------------------------------

    def load_model(self, model_id: str, **kwargs: object) -> None:
        self._model_id = model_id

        try:
            from mlx_audio.stt.utils import load_model
        except ImportError as e:
            raise PodmindError(
                "mlx-audio is not installed. Run: pip install mlx-audio\n"
                f"Original error: {e}"
            ) from e

        print(f"Loading MLX Qwen3-ASR model ({model_id})...")
        t0 = time.perf_counter()
        self._model = load_model(model_id)
        self._load_seconds = time.perf_counter() - t0
        print(f"Model loaded in {self._load_seconds:.1f}s")

    def transcribe_raw(
        self,
        audio_path: str,
        language: str | None = None,
        *,
        profile: bool = False,
        **kwargs: object,
    ) -> TranscriptResult:
        """Run MLX Qwen3-ASR inference. No caching, no file I/O."""
        from mlx_audio.stt.generate import generate_transcription

        lang = self.normalize_language(language)

        duration = _get_duration(audio_path)
        print(f"Audio duration: {duration:.0f}s ({duration / 60:.1f} min)")

        import os as _os
        import tempfile as _tempfile
        with _tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as _tmp:
            _tmp_path = _tmp.name
        t0 = time.perf_counter()
        result = generate_transcription(
            model=self._model,
            audio=audio_path,
            output_path=_tmp_path,
            language=lang,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        _os.unlink(_tmp_path)

        prof = TranscribeProfile(
            total_audio_duration=duration,
            chunk_count=1,
            settings={"model": self._model_id},
        ) if profile else None
        if prof:
            prof.model_load_seconds = self.model_load_seconds
            prof.chunk_transcribe_seconds = [elapsed]

        text = getattr(result, "text", "") or ""
        text = text.strip()

        return TranscriptResult(
            text=text,
            segments=[],
            backend=self.name,
            model=self._model_id,
            language=lang,
            audio_duration=duration,
            profile=prof,
        )

    @staticmethod
    def normalize_language(lang: str | None) -> str | None:
        """Map ISO codes to full names (e.g. 'zh' → 'Chinese') for mlx-qwen."""
        if lang is None:
            return None
        mapped = get_language_full(lang)
        return mapped if mapped is not None else lang
