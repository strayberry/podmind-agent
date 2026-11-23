import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from podmind.config import PodmindError
from podmind.downloader import download_audio


class TestDownloadAudio:
    def test_empty_response_raises(self, tmp_path: Path):
        """Server returns 0 bytes — should raise RuntimeError, not cache empty file."""
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "0"}
        mock_resp.iter_content.return_value = [b""]
        mock_resp.raise_for_status.return_value = None

        with (
            patch("podmind.downloader.requests.Session.get", return_value=mock_resp),
            patch("podmind.downloader.AUDIO_DIR", tmp_path),
            pytest.raises(RuntimeError, match="empty"),
        ):
            download_audio("69f441cd5390b7cc928acdcc", "https://example.com/audio.m4a")

    def test_short_download_raises(self, tmp_path: Path):
        """Server sends fewer bytes than content-length — should raise RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "1000"}
        mock_resp.iter_content.return_value = [b"short"]
        mock_resp.raise_for_status.return_value = None

        with (
            patch("podmind.downloader.requests.Session.get", return_value=mock_resp),
            patch("podmind.downloader.AUDIO_DIR", tmp_path),
            pytest.raises(RuntimeError, match="Download incomplete"),
        ):
            download_audio("69f441cd5390b7cc928acdcc", "https://example.com/audio.m4a")

    def test_existing_nonempty_file_skips(self, tmp_path: Path):
        """Existing non-empty file should be reused without downloading."""
        audio_path = tmp_path / "69f441cd5390b7cc928acdcc.m4a"
        audio_path.write_bytes(b"cached audio")

        with patch("podmind.downloader.AUDIO_DIR", tmp_path):
            result = download_audio("69f441cd5390b7cc928acdcc", "https://example.com/audio.m4a")
            assert result == audio_path

    def test_crash_during_download_cleans_temp(self, tmp_path: Path):
        """If download crashes mid-stream, the .part file is removed."""
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": "10000"}
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock(return_value=None)
        mock_resp.iter_content.side_effect = RuntimeError("connection lost")

        mock_tqdm = MagicMock()
        mock_tqdm.__enter__ = MagicMock(return_value=MagicMock())

        with (
            patch("podmind.downloader.AUDIO_DIR", tmp_path),
            patch("podmind.downloader.tqdm", return_value=mock_tqdm),
            patch("podmind.downloader.requests.Session") as mock_session_class,
            pytest.raises(RuntimeError, match="connection lost"),
        ):
            mock_session_class.return_value.get.return_value = mock_resp
            download_audio("69f441cd5390b7cc928acdcc", "https://example.com/audio.m4a")

        # No .part files left behind
        part_files = list(tmp_path.glob("*.part.*"))
        assert len(part_files) == 0


class TestYouTubeDownloadAudio:
    def test_youtube_existing_nonempty_file_skips(self, tmp_path: Path):
        audio_path = tmp_path / "yt_dQw4w9WgXcQ.m4a"
        audio_path.write_bytes(b"cached youtube audio")

        with patch("podmind.downloader.AUDIO_DIR", tmp_path):
            result = download_audio(
                "yt_dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )
            assert result == audio_path

    def test_youtube_download_success_moves_atomically(self, tmp_path: Path):
        class FakeDownloadError(Exception):
            pass

        def _make_ydl(opts: dict):
            ydl = MagicMock()
            ydl.__enter__.return_value = ydl

            def _download(urls: list[str]):
                out_dir = Path(opts["outtmpl"]).parent
                (out_dir / "audio.m4a").write_bytes(b"fake m4a audio data")
                for hook in opts.get("progress_hooks", []):
                    hook({"status": "downloading", "downloaded_bytes": 19, "total_bytes": 19})
                    hook({"status": "finished"})

            ydl.download.side_effect = _download
            return ydl

        fake_yt_dlp = SimpleNamespace(
            YoutubeDL=MagicMock(side_effect=_make_ydl),
            utils=SimpleNamespace(DownloadError=FakeDownloadError),
        )

        with (
            patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}),
            patch("podmind.downloader.AUDIO_DIR", tmp_path),
            patch("podmind.downloader.ensure_dirs"),
            patch("podmind.downloader.tqdm", return_value=MagicMock(total=19)),
        ):
            result = download_audio(
                "yt_dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )

        assert result == tmp_path / "yt_dQw4w9WgXcQ.m4a"
        assert result.read_bytes() == b"fake m4a audio data"
        assert list(tmp_path.glob("*.part.*")) == []

    def test_youtube_download_error_raises_podmind_error_and_cleans_temp(self, tmp_path: Path):
        class FakeDownloadError(Exception):
            pass

        mock_ydl = MagicMock()
        mock_ydl.__enter__.return_value = mock_ydl
        mock_ydl.download.side_effect = FakeDownloadError("video unavailable")

        fake_yt_dlp = SimpleNamespace(
            YoutubeDL=MagicMock(return_value=mock_ydl),
            utils=SimpleNamespace(DownloadError=FakeDownloadError),
        )

        with (
            patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}),
            patch("podmind.downloader.AUDIO_DIR", tmp_path),
            patch("podmind.downloader.ensure_dirs"),
            pytest.raises(PodmindError, match="YouTube audio download failed"),
        ):
            download_audio("yt_dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert list(tmp_path.glob("*.part.*")) == []

    def test_youtube_empty_output_raises(self, tmp_path: Path):
        class FakeDownloadError(Exception):
            pass

        mock_ydl = MagicMock()
        mock_ydl.__enter__.return_value = mock_ydl
        mock_ydl.download.return_value = None

        fake_yt_dlp = SimpleNamespace(
            YoutubeDL=MagicMock(return_value=mock_ydl),
            utils=SimpleNamespace(DownloadError=FakeDownloadError),
        )

        with (
            patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}),
            patch("podmind.downloader.AUDIO_DIR", tmp_path),
            patch("podmind.downloader.ensure_dirs"),
            pytest.raises(RuntimeError, match="empty"),
        ):
            download_audio("yt_dQw4w9WgXcQ", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert list(tmp_path.glob("*.part.*")) == []
