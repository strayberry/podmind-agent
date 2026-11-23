import pytest

from podmind.config import PodmindError, is_youtube_episode_id, validate_episode_id


class TestValidateEpisodeId:
    def test_valid_hex_returns_same_string(self):
        assert validate_episode_id("69f441cd5390b7cc928acdcc") == "69f441cd5390b7cc928acdcc"

    def test_valid_youtube_id_returns_same_string(self):
        assert validate_episode_id("yt_dQw4w9WgXcQ") == "yt_dQw4w9WgXcQ"
        assert validate_episode_id("yt_abc-123_XYZ") == "yt_abc-123_XYZ"

    def test_is_youtube_episode_id(self):
        assert is_youtube_episode_id("yt_dQw4w9WgXcQ") is True
        assert is_youtube_episode_id("69f441cd5390b7cc928acdcc") is False

    def test_invalid_youtube_id_rejected(self):
        with pytest.raises(PodmindError):
            validate_episode_id("yt_short")
        with pytest.raises(PodmindError):
            validate_episode_id("yt_toolong12345")
        with pytest.raises(PodmindError):
            validate_episode_id("yt_bad/char123")

    def test_path_escape_rejected(self):
        with pytest.raises(PodmindError):
            validate_episode_id("../../etc/passwd")

    def test_short_string_rejected(self):
        with pytest.raises(PodmindError):
            validate_episode_id("abc123")

    def test_non_hex_rejected(self):
        with pytest.raises(PodmindError):
            validate_episode_id("gggg441cd5390b7cc928acdcc")
