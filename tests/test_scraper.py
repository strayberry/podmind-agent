import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from podmind.config import PodmindError
from podmind.scraper import (
    extract_episode_id,
    extract_youtube_id,
    fetch_episode,
    is_youtube_url,
    load_episode_info,
)


class TestEpisodeInfoPath:
    def test_valid_id_resolves_inside_episodes_dir(self, tmp_path):
        from podmind.scraper import _episode_info_path
        with patch("podmind.scraper.EPISODES_DIR", tmp_path):
            path = _episode_info_path("69f441cd5390b7cc928acdcc")
            assert path.parent == tmp_path
            assert path.name == "69f441cd5390b7cc928acdcc.json"

    def test_valid_youtube_id_resolves_inside_episodes_dir(self, tmp_path):
        from podmind.scraper import _episode_info_path
        with patch("podmind.scraper.EPISODES_DIR", tmp_path):
            path = _episode_info_path("yt_dQw4w9WgXcQ")
            assert path.parent == tmp_path
            assert path.name == "yt_dQw4w9WgXcQ.json"

    def test_path_escape_rejected(self):
        from podmind.scraper import _episode_info_path
        with pytest.raises(PodmindError):
            _episode_info_path("../../etc/passwd")

    def test_short_id_rejected(self):
        from podmind.scraper import _episode_info_path
        with pytest.raises(PodmindError):
            _episode_info_path("abc123")


class TestYouTubeUrlParsing:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "yt_dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "yt_dQw4w9WgXcQ"),
            ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "yt_dQw4w9WgXcQ"),
            ("https://music.youtube.com/watch?v=dQw4w9WgXcQ", "yt_dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "yt_dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ?si=abc", "yt_dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "yt_dQw4w9WgXcQ"),
            ("https://www.youtube.com/live/dQw4w9WgXcQ", "yt_dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "yt_dQw4w9WgXcQ"),
        ],
    )
    def test_extract_youtube_id_formats(self, url: str, expected: str):
        assert is_youtube_url(url) is True
        assert extract_youtube_id(url) == expected
        assert extract_episode_id(url) == expected

    def test_non_youtube_url(self):
        url = "https://www.xiaoyuzhoufm.com/episode/69f441cd5390b7cc928acdcc"
        assert is_youtube_url(url) is False
        assert extract_episode_id(url) == "69f441cd5390b7cc928acdcc"

    def test_invalid_youtube_url_raises(self):
        with pytest.raises(PodmindError):
            extract_youtube_id("https://www.youtube.com/watch?v=short")
        with pytest.raises(PodmindError):
            extract_youtube_id("https://www.youtube.com/channel/UC12345")


class TestFetchYouTubeEpisode:
    def test_fetch_youtube_metadata_and_persist(self, tmp_path):
        class FakeDownloadError(Exception):
            pass

        mock_ydl = MagicMock()
        mock_ydl.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {
            "title": "Demo Talk",
            "channel": "AI Channel",
            "duration": 1805,
        }
        fake_yt_dlp = SimpleNamespace(
            YoutubeDL=MagicMock(return_value=mock_ydl),
            utils=SimpleNamespace(DownloadError=FakeDownloadError),
        )

        with (
            patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}),
            patch("podmind.scraper.EPISODES_DIR", tmp_path),
            patch("podmind.scraper.ensure_dirs"),
            patch("podmind.io_utils.ensure_dirs"),
        ):
            info = fetch_episode("https://youtu.be/dQw4w9WgXcQ")

        assert info.episode_id == "yt_dQw4w9WgXcQ"
        assert info.title == "Demo Talk"
        assert info.podcast_name == "AI Channel"
        assert info.duration_sec == 1805.0
        assert info.audio_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert info.source == "youtube"

        saved = json.loads((tmp_path / "yt_dQw4w9WgXcQ.json").read_text(encoding="utf-8"))
        assert saved["source"] == "youtube"
        assert saved["title"] == "Demo Talk"

    def test_load_legacy_episode_info_without_source(self, tmp_path):
        eid = "69f441cd5390b7cc928acdcc"
        legacy = {
            "episode_id": eid,
            "title": "Legacy",
            "podcast_name": "Show",
            "audio_url": "https://example.com/a.m4a",
            "duration_sec": 120.0,
        }
        (tmp_path / f"{eid}.json").write_text(json.dumps(legacy), encoding="utf-8")

        with patch("podmind.scraper.EPISODES_DIR", tmp_path):
            info = load_episode_info(eid)

        assert info is not None
        assert info.source == "xiaoyuzhou"
        assert info.title == "Legacy"
