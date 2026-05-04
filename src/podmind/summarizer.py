import hashlib
import json
import os
import tempfile
from importlib import resources
from pathlib import Path

from openai import OpenAI

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    OUTPUTS_DIR,
    ensure_dirs,
    validate_episode_id,
)


def summary_path(episode_id: str, *, backend: str | None = None) -> Path:
    validate_episode_id(episode_id)
    if backend and backend not in ("qwen", ""):
        return OUTPUTS_DIR / f"{episode_id}_summary.{backend}.md"
    return OUTPUTS_DIR / f"{episode_id}_summary.md"


def _summary_meta_path(episode_id: str, *, backend: str | None = None) -> Path:
    validate_episode_id(episode_id)
    if backend and backend not in ("qwen", ""):
        return OUTPUTS_DIR / f"{episode_id}_summary.{backend}.meta.json"
    return OUTPUTS_DIR / f"{episode_id}_summary.meta.json"


def _load_prompt() -> str:
    return (resources.files("podmind.prompts") / "summary.txt").read_text(encoding="utf-8")


def _atomic_write(path: Path, content: str) -> None:
    """Write content atomically via a temp file + rename."""
    ensure_dirs()
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp.")
    try:
        try:
            os.write(fd, content.encode("utf-8"))
        finally:
            os.close(fd)
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# Maximum chars for direct summarization (~25K tokens).
_MAX_DIRECT_CHARS = 100_000
# Chunk size for map step (leaves headroom for prompt + response).
_CHUNK_CHARS = 80_000

_CHUNK_EXTRACT_PROMPT = (
    "你是一个专业的知识整理助手。请从以下播客文字稿片段中提取核心观点和关键信息。\n"
    "\n"
    "要求：\n"
    "1. 提取所有重要的论点、框架、模型、公式和结论\n"
    "2. 保留关键数据和事实\n"
    "3. 用简洁的要点形式输出，每个要点1-2行\n"
    "4. 不要遗漏重要内容\n"
    "5. 不要加入原文没有的内容\n"
    "\n"
    "文字稿片段：\n"
    "\n"
    "{chunk}"
)


def _extract_key_points(client: OpenAI, chunk: str, model: str) -> str:
    """Extract key points from a single transcript chunk."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": _CHUNK_EXTRACT_PROMPT.format(chunk=chunk)},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("DeepSeek returned empty response for chunk extraction")
    return content


def _map_reduce_summarize(
    client: OpenAI,
    transcript: str,
    title: str,
    model: str,
    system_prompt: str,
) -> str:
    """Summarize a long transcript via map-reduce.

    Split transcript into chunks, extract key points from each, then
    generate a final structured summary from the combined key points.
    """
    chunks = [
        transcript[i : i + _CHUNK_CHARS]
        for i in range(0, len(transcript), _CHUNK_CHARS)
    ]
    print(f"Transcript is {len(transcript):,} chars — "
          f"splitting into {len(chunks)} chunks for map-reduce")

    key_points_parts: list[str] = []
    for i, chunk in enumerate(chunks):
        print(f"  Extracting key points from chunk {i + 1}/{len(chunks)}...")
        key_points_parts.append(_extract_key_points(client, chunk, model))

    combined = "\n\n---\n\n".join(key_points_parts)
    print(f"  Key points extracted: {len(combined):,} chars — generating final summary")

    user_message = (
        f"播客名称: {title}\n\n"
        f"以下是从播客文字稿中提取的核心观点：\n\n{combined}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=8192,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("DeepSeek returned empty response")
    return content


def summarize(
    episode_id: str,
    transcript: str,
    title: str = "",
    model: str = "deepseek-v4-pro",
    force: bool = False,
    *,
    backend: str | None = None,
) -> str:
    """Summarize a transcript using DeepSeek API.

    Cache is invalidated when transcript content, model, or prompt changes
    (tracked via sidecar .meta.json).
    """
    out_path = summary_path(episode_id, backend=backend)
    meta_path = _summary_meta_path(episode_id, backend=backend)
    system_prompt = _load_prompt().strip()

    current_meta = {
        "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
        "model": model,
        "prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "chunk_extract_prompt_sha256": hashlib.sha256(
            _CHUNK_EXTRACT_PROMPT.encode()
        ).hexdigest(),
        "chunk_chars": _CHUNK_CHARS,
        "max_direct_chars": _MAX_DIRECT_CHARS,
        "title_sha256": hashlib.sha256(title.encode()).hexdigest(),
    }

    if out_path.exists() and not force:
        if meta_path.exists():
            try:
                saved = json.loads(meta_path.read_text(encoding="utf-8"))
                if saved == current_meta:
                    print(f"Summary already exists: {out_path}")
                    return out_path.read_text(encoding="utf-8")
            except (json.JSONDecodeError, KeyError):
                pass
        else:
            print(f"Summary already exists: {out_path} (no meta; re-run with --force to refresh)")
            return out_path.read_text(encoding="utf-8")

    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Create a .env file or set the environment variable."
        )

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    if len(transcript) > _MAX_DIRECT_CHARS:
        content = _map_reduce_summarize(
            client, transcript, title, model, system_prompt
        )
    else:
        user_message = f"播客名称: {title}\n\n以下是播客文字稿:\n\n{transcript}"
        print(f"Summarizing with DeepSeek (model={model})...")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=8192,
        )
        _content = resp.choices[0].message.content
        if _content is None:
            raise RuntimeError("DeepSeek returned empty response")
        content = _content

    _atomic_write(out_path, content)
    meta_path.write_text(json.dumps(current_meta, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved: {out_path}")

    return content
