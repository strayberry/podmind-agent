from unittest.mock import MagicMock, patch

import pytest

from podmind.summarizer import _reduce_notes, _split_transcript, summarize


def _chat_response(content: str | None) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestSummarize:
    def test_empty_response_raises(self, tmp_path):
        """DeepSeek returns None content — should raise RuntimeError."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _chat_response(None)

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

    def test_split_transcript_prefers_paragraph_boundaries(self):
        chunks = _split_transcript("alpha\n\nbeta beta\n\ngamma", chunk_chars=12)

        assert chunks == ["alpha", "beta beta", "gamma"]

    def test_long_summary_extracts_chunks_then_renders_with_shared_prompt(self, tmp_path):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _chat_response("## Topic A\n- point from first chunk"),
            _chat_response("## Topic B\n- point from second chunk"),
            _chat_response("# Test\n\nfinal summary"),
        ]

        with (
            patch("podmind.summarizer.OpenAI", return_value=mock_client),
            patch("podmind.summarizer.OUTPUTS_DIR", tmp_path),
            patch("podmind.summarizer.DEEPSEEK_API_KEY", "sk-test"),
            patch("podmind.summarizer._MAX_DIRECT_CHARS", 10),
            patch("podmind.summarizer._CHUNK_CHARS", 20),
        ):
            result = summarize(
                "69f441cd5390b7cc928acdcc",
                "first paragraph\n\nsecond paragraph",
                title="Test",
            )

        assert result == "# Test\n\nfinal summary"
        assert mock_client.chat.completions.create.call_count == 3

        first_call = mock_client.chat.completions.create.call_args_list[0].kwargs
        final_call = mock_client.chat.completions.create.call_args_list[-1].kwargs

        assert first_call["messages"][0]["role"] == "system"
        assert final_call["messages"][0]["role"] == "system"
        assert "point from first chunk" in final_call["messages"][1]["content"]
        assert "point from second chunk" in final_call["messages"][1]["content"]

    def test_reduce_notes_compacts_long_extracted_notes(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _chat_response("short group one"),
            _chat_response("short group two"),
        ]

        with patch("podmind.summarizer._REDUCE_NOTES_MAX_CHARS", 30):
            result = _reduce_notes(
                mock_client,
                "first extracted topic\n\nsecond extracted topic",
                "Test",
                "deepseek-v4-pro",
                "system prompt",
            )

        assert "short group one" in result
        assert "short group two" in result
        assert mock_client.chat.completions.create.call_count == 2
