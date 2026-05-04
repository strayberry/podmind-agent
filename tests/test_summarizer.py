from unittest.mock import MagicMock, patch

import pytest

from podmind.summarizer import summarize


class TestSummarize:
    def test_empty_response_raises(self, tmp_path):
        """DeepSeek returns None content — should raise RuntimeError."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_resp

        with (
            patch("podmind.summarizer.OpenAI", return_value=mock_client),
            patch("podmind.summarizer.OUTPUTS_DIR", tmp_path),
            patch("podmind.summarizer.DEEPSEEK_API_KEY", "sk-test"),
            pytest.raises(RuntimeError, match="empty response"),
        ):
            summarize(
                "69f441cd5390b7cc928acdcc",
                "test transcript",
                title="Test",
            )
