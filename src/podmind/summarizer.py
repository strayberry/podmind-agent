import hashlib
import json
from importlib import resources
from pathlib import Path

from openai import OpenAI

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    OUTPUTS_DIR,
    validate_episode_id,
)
from .io_utils import atomic_write


def summary_path(episode_id: str, *, backend: str | None = None) -> Path:
    validate_episode_id(episode_id)
    if backend and backend not in ("qwen-asr", ""):
        return OUTPUTS_DIR / f"{episode_id}_summary.{backend}.md"
    return OUTPUTS_DIR / f"{episode_id}_summary.md"


def _summary_meta_path(episode_id: str, *, backend: str | None = None) -> Path:
    validate_episode_id(episode_id)
    if backend and backend not in ("qwen-asr", ""):
        return OUTPUTS_DIR / f"{episode_id}_summary.{backend}.meta.json"
    return OUTPUTS_DIR / f"{episode_id}_summary.meta.json"


def _load_prompt() -> str:
    return (resources.files("podmind.prompts") / "summary.txt").read_text(encoding="utf-8")


# Maximum chars for direct summarization. Above this, use multi-pass.
_MAX_DIRECT_CHARS = 30_000
# Chunk size for multi-pass extraction.
_CHUNK_CHARS = 25_000
# Max tokens per chunk extraction.
_CHUNK_EXTRACT_MAX_TOKENS = 4096
# Maximum extracted notes passed into the final renderer. Above this, reduce first.
_REDUCE_NOTES_MAX_CHARS = 30_000
# Max tokens for final merge summary.
_FINAL_MAX_TOKENS = 24576

_CHUNK_EXTRACT_PROMPT = (
    "请从以下播客文字稿片段中提取后续生成长期复习笔记所需的核心素材。\n"
    "\n"
    "要求：\n"
    "1. 只提取文字稿中明确出现的信息，不要补充外部知识。\n"
    "2. 按主题归纳核心观点、关键概念、方法、框架、例子或判断。\n"
    "3. 内容要简洁，服务于最终生成不超过10个 section 的结构化摘要。\n"
    "4. 如果片段只是承接上下文，也要保留可用于最终合并的事实。\n"
    "\n"
    "输出格式：\n"
    "\n"
    "## 主题名称\n"
    "- 核心观点\n"
    "- 关键事实/例子/框架\n"
    "\n"
    "如果该片段信息很少，也要输出最有价值的要点。\n"
    "\n"
    "文字稿片段：\n"
    "\n"
    "{chunk}"
)


def _split_transcript(transcript: str, chunk_chars: int = _CHUNK_CHARS) -> list[str]:
    """Split transcript on paragraph boundaries when possible."""
    paragraphs = [p.strip() for p in transcript.splitlines() if p.strip()]
    if not paragraphs:
        return [
            transcript[i : i + chunk_chars]
            for i in range(0, len(transcript), chunk_chars)
        ]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            chunks.extend(
                paragraph[i : i + chunk_chars]
                for i in range(0, len(paragraph), chunk_chars)
            )
            continue

        next_len = current_len + len(paragraph) + (2 if current else 0)
        if current and next_len > chunk_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len = next_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _chunk_extract(
    client: OpenAI,
    chunk: str,
    title: str,
    model: str,
    system_prompt: str,
    chunk_idx: int,
    total: int,
) -> str:
    """Extract concise candidate notes from a single transcript chunk."""
    user_message = (
        f"播客名称: {title}\n"
        f"片段: {chunk_idx + 1}/{total}\n\n"
        f"{_CHUNK_EXTRACT_PROMPT.format(chunk=chunk)}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=_CHUNK_EXTRACT_MAX_TOKENS,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError(f"DeepSeek returned empty response for chunk {chunk_idx + 1}")
    return content


def _render_final_summary(
    client: OpenAI,
    notes: str,
    title: str,
    model: str,
    system_prompt: str,
) -> str:
    """Render the final note with the shared summary prompt."""
    user_message = (
        f"播客名称: {title}\n\n"
        "以下是从完整文字稿各片段中提取出的核心素材。"
        "请去重、合并相近主题，并严格按照系统提示的格式输出最终摘要。\n\n"
        f"{notes}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=_FINAL_MAX_TOKENS,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("DeepSeek returned empty response for final summary")
    return content


def _reduce_notes(
    client: OpenAI,
    notes: str,
    title: str,
    model: str,
    system_prompt: str,
) -> str:
    """Reduce long extracted notes before final rendering."""
    if len(notes) <= _REDUCE_NOTES_MAX_CHARS:
        return notes

    note_chunks = _split_transcript(notes, _REDUCE_NOTES_MAX_CHARS)
    print(f"  Extracted notes are {len(notes):,} chars — reducing "
          f"{len(note_chunks)} groups before final render")

    reduced: list[str] = []
    for i, chunk in enumerate(note_chunks):
        user_message = (
            f"播客名称: {title}\n"
            f"素材组: {i + 1}/{len(note_chunks)}\n\n"
            "以下是从文字稿片段中提取出的核心素材。请合并重复主题，保留关键观点、"
            "概念、框架、例子和判断，输出用于最终摘要的简洁结构化素材。"
            "不要生成最终 markdown 摘要。\n\n"
            f"{chunk}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=_CHUNK_EXTRACT_MAX_TOKENS,
        )
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError(f"DeepSeek returned empty response for reduce group {i + 1}")
        reduced.append(f"## 合并素材 {i + 1}/{len(note_chunks)}\n\n{content.strip()}")

    reduced_notes = "\n\n---\n\n".join(reduced)
    if len(reduced_notes) < len(notes):
        return _reduce_notes(client, reduced_notes, title, model, system_prompt)
    return reduced_notes


def _multi_pass_summarize(
    client: OpenAI,
    transcript: str,
    title: str,
    model: str,
    system_prompt: str,
) -> str:
    """Multi-pass summarization: chunk extraction → global final rendering."""
    chunks = _split_transcript(transcript, _CHUNK_CHARS)
    print(f"Transcript is {len(transcript):,} chars — "
          f"multi-pass with {len(chunks)} chunks of {_CHUNK_CHARS:,} chars each")

    chunk_notes: list[str] = []
    for i, chunk in enumerate(chunks):
        print(f"  Extracting chunk {i + 1}/{len(chunks)} "
              f"({len(chunk):,} chars)...")
        detail = _chunk_extract(
            client, chunk, title, model, system_prompt, i, len(chunks),
        )
        chunk_notes.append(f"## 片段 {i + 1}/{len(chunks)}\n\n{detail.strip()}")
        print(f"    → chunk {i + 1}: {len(detail):,} chars")

    print("  Rendering final summary from extracted notes...")
    notes = _reduce_notes(
        client,
        "\n\n---\n\n".join(chunk_notes),
        title,
        model,
        system_prompt,
    )
    return _render_final_summary(
        client,
        notes,
        title,
        model,
        system_prompt,
    )


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
        "summary_pipeline_version": 3,
        "chunk_extract_prompt_sha256": hashlib.sha256(
            _CHUNK_EXTRACT_PROMPT.encode()
        ).hexdigest(),
        "chunk_chars": _CHUNK_CHARS,
        "max_direct_chars": _MAX_DIRECT_CHARS,
        "chunk_extract_max_tokens": _CHUNK_EXTRACT_MAX_TOKENS,
        "reduce_notes_max_chars": _REDUCE_NOTES_MAX_CHARS,
        "final_max_tokens": _FINAL_MAX_TOKENS,
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
        content = _multi_pass_summarize(
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
            max_tokens=_FINAL_MAX_TOKENS,
        )
        _content = resp.choices[0].message.content
        if _content is None:
            raise RuntimeError("DeepSeek returned empty response")
        content = _content

    atomic_write(out_path, content)
    atomic_write(meta_path, json.dumps(current_meta, ensure_ascii=False))
    print(f"Summary saved: {out_path}")

    return content
