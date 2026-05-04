import pytest

from podmind.config import PodmindError, validate_episode_id


class TestValidateEpisodeId:
    def test_valid_hex_returns_same_string(self):
        assert validate_episode_id("69f441cd5390b7cc928acdcc") == "69f441cd5390b7cc928acdcc"

    def test_path_escape_rejected(self):
        with pytest.raises(PodmindError):
            validate_episode_id("../../etc/passwd")

    def test_short_string_rejected(self):
        with pytest.raises(PodmindError):
            validate_episode_id("abc123")

    def test_non_hex_rejected(self):
        with pytest.raises(PodmindError):
            validate_episode_id("gggg441cd5390b7cc928acdcc")
