from unittest.mock import patch

import pytest

from podmind.config import PodmindError


class TestEpisodeInfoPath:
    def test_valid_id_resolves_inside_episodes_dir(self, tmp_path):
        from podmind.scraper import _episode_info_path
        with patch("podmind.scraper.EPISODES_DIR", tmp_path):
            path = _episode_info_path("69f441cd5390b7cc928acdcc")
            assert path.parent == tmp_path
            assert path.name == "69f441cd5390b7cc928acdcc.json"

    def test_path_escape_rejected(self):
        from podmind.scraper import _episode_info_path
        with pytest.raises(PodmindError):
            _episode_info_path("../../etc/passwd")

    def test_short_id_rejected(self):
        from podmind.scraper import _episode_info_path
        with pytest.raises(PodmindError):
            _episode_info_path("abc123")
