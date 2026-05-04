from importlib import resources


def test_prompt_is_readable():
    """The summary prompt template must be packaged and readable."""
    path = resources.files("podmind.prompts") / "summary.txt"
    content = path.read_text(encoding="utf-8")

    assert len(content) > 100
    assert "核心总结" in content
    assert "一句话总结" in content
