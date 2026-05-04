import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import EPISODES_DIR, ensure_dirs, validate_episode_id

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}


@dataclass
class EpisodeInfo:
    episode_id: str
    title: str
    podcast_name: str
    audio_url: str
    duration_sec: float


def extract_episode_id(url: str) -> str:
    """Extract and validate the episode ID from a Xiaoyuzhou FM URL."""
    path = urlparse(url).path
    parts = path.rstrip("/").split("/")
    if parts and parts[-1]:
        return validate_episode_id(parts[-1])
    raise ValueError(f"Could not extract episode ID from URL: {url}")


def _episode_info_path(episode_id: str) -> Path:
    validate_episode_id(episode_id)
    return EPISODES_DIR / f"{episode_id}.json"


def save_episode_info(info: EpisodeInfo) -> None:
    """Persist episode metadata to episodes/{id}.json."""
    ensure_dirs()
    path = _episode_info_path(info.episode_id)
    path.write_text(json.dumps(asdict(info), ensure_ascii=False, indent=2), encoding="utf-8")


def load_episode_info(episode_id: str) -> EpisodeInfo | None:
    """Load persisted episode metadata, or None if not found."""
    path = _episode_info_path(episode_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return EpisodeInfo(**data)


def fetch_episode(url: str) -> EpisodeInfo:
    """Scrape episode metadata and audio URL from a Xiaoyuzhou FM page."""
    episode_id = extract_episode_id(url)
    page_url = f"https://www.xiaoyuzhoufm.com/episode/{episode_id}"

    resp = requests.get(page_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    match = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text
    )
    if not match:
        raise RuntimeError(
            "Could not find __NEXT_DATA__ script tag in page. "
            "The Xiaoyuzhou FM page structure may have changed."
        )

    data = json.loads(match.group(1))
    try:
        episode = data["props"]["pageProps"]["episode"]
    except (KeyError, TypeError) as e:
        raise RuntimeError(
            f"Could not parse episode data from page — "
            f"the page structure may have changed: {e}"
        ) from e

    audio_url = episode.get("enclosure", {}).get("url", "")
    if not audio_url:
        media = episode.get("media", {})
        audio_url = media.get("source", {}).get("url", "")

    if not audio_url:
        raise RuntimeError("Could not find audio URL in episode data")

    info = EpisodeInfo(
        episode_id=episode_id,
        title=episode.get("title", ""),
        podcast_name=episode.get("podcast", {}).get("title", ""),
        audio_url=audio_url,
        duration_sec=float(episode.get("duration", 0)),
    )
    save_episode_info(info)
    return info
