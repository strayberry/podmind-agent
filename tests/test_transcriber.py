from unittest.mock import MagicMock, patch

import pytest

from podmind.transcriber import (
    TranscribeProfile,
    TranscriptResult,
    _split_audio,
    _transcribe_episode,
)
from podmind.transcriber._shared import _transcript_meta_path, transcript_path


class TestTranscribeProfile:
    def test_rtf_calculates_correctly(self):
        prof = TranscribeProfile(
            chunk_transcribe_seconds=[30.0, 25.0, 20.0],
            total_audio_duration=300.0,
        )
        assert prof.total_transcribe_seconds == 75.0
        assert prof.rtf == pytest.approx(0.25)

    def test_rtf_zero_duration_avoids_division_by_zero(self):
        prof = TranscribeProfile(
            chunk_transcribe_seconds=[10.0],
            total_audio_duration=0.0,
        )
        assert prof.rtf == 0.0

    def test_format_includes_key_fields(self):
        prof = TranscribeProfile(
            model_load_seconds=5.0,
            ffprobe_seconds=0.1,
            ffmpeg_split_seconds=2.0,
            chunk_seconds_used=300,
            batch_size_used=2,
            chunk_count=4,
            chunk_transcribe_seconds=[30.0, 30.0, 25.0, 20.0],
            total_audio_duration=1200.0,
        )
        out = prof.format()
        assert "Model load:" in out
        assert "RTF:" in out
        assert "batch=2" in out
        assert "Chunk 1/4" in out
        assert "Chunk 4/4" in out

    def test_default_values(self):
        prof = TranscribeProfile()
        assert prof.total_transcribe_seconds == 0.0
        assert prof.rtf == 0.0
        assert prof.chunk_seconds_used == 600
        assert prof.batch_size_used == 1


class TestSplitAudio:
    def test_passes_chunk_seconds_to_ffmpeg(self, tmp_path):
        """_split_audio should forward chunk_seconds to ffmpeg -segment_time."""
        with patch(
            "podmind.transcriber.backends.qwen.subprocess.run"
        ) as mock_run, patch.object(
            _split_audio, "__defaults__", (600,)
        ):  # ensure default
            _split_audio("/fake/audio.m4a", str(tmp_path), chunk_seconds=120)
            args_str = " ".join(str(a) for a in mock_run.call_args[0][0])
            assert "120" in args_str

    def test_default_chunk_seconds(self, tmp_path):
        with patch(
            "podmind.transcriber.backends.qwen.subprocess.run"
        ) as mock_run:
            with patch("podmind.transcriber.backends.qwen.Path.glob", return_value=[]):
                _split_audio("/fake/audio.m4a", str(tmp_path))
                args_str = " ".join(str(a) for a in mock_run.call_args[0][0])
                assert "600" in args_str


class TestTranscribeEpisode:
    def test_returns_none_profile_when_profile_false(self, tmp_path):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = [
            MagicMock(text="hello"),
        ]
        mock_chunk = tmp_path / "chunk_001.wav"
        mock_chunk.touch()

        with patch("podmind.transcriber.backends.qwen._get_duration", return_value=120.0):
            with patch(
                "podmind.transcriber.backends.qwen._split_audio",
                return_value=[mock_chunk],
            ):
                text, prof = _transcribe_episode(
                    mock_model,
                    "69f441cd5390b7cc928acdcc",
                    "/fake/audio.m4a",
                    profile=False,
                )

        assert prof is None
        assert text == "hello"

    def test_returns_profile_when_profile_true(self, tmp_path):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = [
            MagicMock(text="hello"),
        ]
        mock_chunks = [
            tmp_path / "chunk_001.wav",
            tmp_path / "chunk_002.wav",
            tmp_path / "chunk_003.wav",
        ]
        for c in mock_chunks:
            c.touch()

        with patch("podmind.transcriber.backends.qwen._get_duration", return_value=600.0):
            with patch(
                "podmind.transcriber.backends.qwen._split_audio",
                return_value=mock_chunks,
            ):
                text, prof = _transcribe_episode(
                    mock_model,
                    "69f441cd5390b7cc928acdcc",
                    "/fake/audio.m4a",
                    profile=True,
                    chunk_seconds=300,
                    batch_size=2,
                )

        assert prof is not None
        assert prof.chunk_seconds_used == 300
        assert prof.batch_size_used == 2
        assert prof.total_audio_duration == 600.0

    def test_batch_grouping(self, tmp_path):
        """5 chunks with batch_size=2 should make 3 calls to model.transcribe."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = [
            MagicMock(text="a"),
            MagicMock(text="b"),
        ]

        chunks = [tmp_path / f"chunk_{i:03d}.wav" for i in range(5)]
        for c in chunks:
            c.touch()

        with patch(
            "podmind.transcriber.backends.qwen._get_duration", return_value=1000.0
        ):
            with patch(
                "podmind.transcriber.backends.qwen._split_audio",
                return_value=chunks,
            ):
                _transcribe_episode(
                    mock_model,
                    "69f441cd5390b7cc928acdcc",
                    "/fake/audio.m4a",
                    batch_size=2,
                )

        assert mock_model.transcribe.call_count == 3
        # First call: 2 chunks
        assert len(mock_model.transcribe.call_args_list[0][1]["audio"]) == 2
        # Second call: 2 chunks
        assert len(mock_model.transcribe.call_args_list[1][1]["audio"]) == 2
        # Third call: 1 chunk
        assert len(mock_model.transcribe.call_args_list[2][1]["audio"]) == 1

    def test_batch_size_in_meta(self, tmp_path):
        """_write_transcript_cache includes backend-specific extra_meta."""
        from podmind.transcriber import _write_transcript_cache

        out_path = tmp_path / "test.txt"
        meta_path = tmp_path / "test.meta.json"
        result = TranscriptResult(
            text="x",
            backend="qwen",
            model="Qwen/Qwen3-ASR-0.6B",
        )

        with patch("podmind.transcriber._file_sha256", return_value="abc123"):
            with patch(
                "podmind.transcriber.transcript_path", return_value=out_path,
            ):
                with patch(
                    "podmind.transcriber._transcript_meta_path", return_value=meta_path,
                ):
                    with patch("podmind.transcriber._atomic_write"):
                        _write_transcript_cache(
                            "69f441cd5390b7cc928acdcc",
                            result,
                            "/fake/audio.m4a",
                            "Chinese",
                            extra_meta={"chunk_seconds": 300, "batch_size": 2},
                        )

        import json
        meta = json.loads(meta_path.read_text())
        assert meta["chunk_seconds"] == 300
        assert meta["batch_size"] == 2
        assert meta["backend"] == "qwen"

    def test_cache_hit_returns_same_text(self, tmp_path):
        """_check_transcript_cache returns text on meta match."""
        from podmind.transcriber import _check_transcript_cache

        out_path = tmp_path / "test.txt"
        out_path.write_text("cached text", encoding="utf-8")
        meta_path = tmp_path / "test.meta.json"
        meta_path.write_text(
            '{"backend":"qwen","backend_model":"test",'
            '"language":null,"audio_sha256":"abc123",'
            '"chunk_seconds":600,"batch_size":1}',
            encoding="utf-8",
        )

        with patch("podmind.transcriber._file_sha256", return_value="abc123"):
            with patch(
                "podmind.transcriber.transcript_path", return_value=out_path,
            ):
                with patch(
                    "podmind.transcriber._transcript_meta_path", return_value=meta_path,
                ):
                    text = _check_transcript_cache(
                        "69f441cd5390b7cc928acdcc",
                        "/fake/audio.m4a",
                        backend="qwen",
                        backend_model="test",
                        extra_meta={"chunk_seconds": 600, "batch_size": 1},
                    )

        assert text == "cached text"


class TestTranscriptResult:
    def test_default_values(self):
        r = TranscriptResult(text="hello")
        assert r.text == "hello"
        assert r.segments == []
        assert r.backend == ""
        assert r.model == ""
        assert r.language is None
        assert r.audio_duration == 0.0
        assert r.profile is None


class TestTranscriptPath:
    def test_no_backend_uses_plain_path(self, tmp_path):
        with patch("podmind.transcriber._shared.TRANSCRIPTS_DIR", tmp_path):
            p = transcript_path("69f441cd5390b7cc928acdcc")
            assert p == tmp_path / "69f441cd5390b7cc928acdcc.txt"

    def test_qwen_backend_uses_plain_path(self, tmp_path):
        with patch("podmind.transcriber._shared.TRANSCRIPTS_DIR", tmp_path):
            p = transcript_path("69f441cd5390b7cc928acdcc", backend="qwen")
            assert p == tmp_path / "69f441cd5390b7cc928acdcc.txt"

    def test_mlx_whisper_backend_uses_suffixed_path(self, tmp_path):
        with patch("podmind.transcriber._shared.TRANSCRIPTS_DIR", tmp_path):
            p = transcript_path("69f441cd5390b7cc928acdcc", backend="mlx-whisper")
            assert p == tmp_path / "69f441cd5390b7cc928acdcc.mlx-whisper.txt"


class TestTranscriptMetaPath:
    def test_no_backend_uses_plain_path(self, tmp_path):
        with patch("podmind.transcriber._shared.TRANSCRIPTS_DIR", tmp_path):
            p = _transcript_meta_path("69f441cd5390b7cc928acdcc")
            assert p == tmp_path / "69f441cd5390b7cc928acdcc.meta.json"

    def test_mlx_whisper_backend_uses_suffixed_path(self, tmp_path):
        with patch("podmind.transcriber._shared.TRANSCRIPTS_DIR", tmp_path):
            p = _transcript_meta_path("69f441cd5390b7cc928acdcc", backend="mlx-whisper")
            assert p == tmp_path / "69f441cd5390b7cc928acdcc.mlx-whisper.meta.json"


class TestLanguageNormalization:
    def test_qwen_passes_through_full_name(self):
        from podmind.transcriber.backends.qwen import QwenBackend
        assert QwenBackend.normalize_language("Chinese") == "Chinese"
        assert QwenBackend.normalize_language("English") == "English"
        assert QwenBackend.normalize_language(None) is None

    def test_mlx_whisper_maps_full_to_iso(self):
        from podmind.transcriber.backends.mlx_whisper import MLXWhisperBackend
        assert MLXWhisperBackend.normalize_language("Chinese") == "zh"
        assert MLXWhisperBackend.normalize_language("English") == "en"
        assert MLXWhisperBackend.normalize_language("Japanese") == "ja"

    def test_mlx_whisper_passes_through_iso(self):
        from podmind.transcriber.backends.mlx_whisper import MLXWhisperBackend
        assert MLXWhisperBackend.normalize_language("zh") == "zh"
        assert MLXWhisperBackend.normalize_language("en") == "en"

    def test_mlx_whisper_unknown_returns_none(self):
        from podmind.transcriber.backends.mlx_whisper import MLXWhisperBackend
        assert MLXWhisperBackend.normalize_language("Klingon") is None

    def test_mlx_whisper_none_returns_none(self):
        from podmind.transcriber.backends.mlx_whisper import MLXWhisperBackend
        assert MLXWhisperBackend.normalize_language(None) is None


class TestBuildMeta:
    def test_includes_backend_and_model(self):
        from podmind.transcriber import _build_meta
        with patch(
            "podmind.transcriber._file_sha256", return_value="abc123"
        ):
            meta = _build_meta(
                "/fake/audio.m4a", "Chinese",
                backend="mlx-whisper", backend_model="mlx-community/whisper-turbo",
            )
        assert meta["backend"] == "mlx-whisper"
        assert meta["backend_model"] == "mlx-community/whisper-turbo"
        assert meta["language"] == "Chinese"
        assert meta["audio_sha256"] == "abc123"

    def test_extra_meta_merged(self):
        from podmind.transcriber import _build_meta
        with patch(
            "podmind.transcriber._file_sha256", return_value="abc123"
        ):
            meta = _build_meta(
                "/fake/audio.m4a", None,
                backend="qwen", backend_model="Qwen/Qwen3-ASR-0.6B",
                extra_meta={"chunk_seconds": 300, "batch_size": 2},
            )
        assert meta["chunk_seconds"] == 300
        assert meta["batch_size"] == 2


class TestBackendRegistry:
    def test_get_backend_class_qwen(self):
        from podmind.transcriber import _get_backend_class
        from podmind.transcriber.backends.qwen import QwenBackend
        assert _get_backend_class("qwen") is QwenBackend

    def test_get_backend_class_mlx_whisper(self):
        from podmind.transcriber import _get_backend_class
        from podmind.transcriber.backends.mlx_whisper import MLXWhisperBackend
        assert _get_backend_class("mlx-whisper") is MLXWhisperBackend

    def test_get_backend_class_unknown_raises(self):
        from podmind.config import PodmindError
        from podmind.transcriber import _get_backend_class
        with pytest.raises(PodmindError, match="Unknown ASR backend"):
            _get_backend_class("nonexistent")
