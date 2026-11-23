import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import EPISODES_DIR, PodmindError, ensure_dirs, validate_episode_id
from .io_utils import atomic_write

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}


def _session_with_retry(
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (500, 502, 503, 504),
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods={"GET"},
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}
_YOUTUBE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass
class EpisodeInfo:
    episode_id: str
    title: str
    podcast_name: str
    audio_url: str
    duration_sec: float
    source: str = "xiaoyuzhou"


def is_youtube_url(url: str) -> bool:
    """Return True if *url* points to a YouTube host."""
    host = (urlparse(url).hostname or "").lower()
    return host in _YOUTUBE_HOSTS


def extract_youtube_id(url: str) -> str:
    """Extract and validate a YouTube video ID, returning ``yt_<video_id>``."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        raise PodmindError(f"Not a YouTube URL: {url}")

    video_id = ""
    parts = [p for p in parsed.path.split("/") if p]
    if host in {"youtu.be", "www.youtu.be"}:
        if parts:
            video_id = parts[0]
    elif parts and parts[0] == "watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    elif len(parts) >= 2 and parts[0] in {"shorts", "live", "embed", "v"}:
        video_id = parts[1]

    if not _YOUTUBE_VIDEO_ID_RE.match(video_id):
        raise PodmindError(f"Could not extract YouTube video ID from URL: {url}")
    return validate_episode_id(f"yt_{video_id}")


def extract_episode_id(url: str) -> str:
    """Extract and validate the episode ID from a Xiaoyuzhou FM or YouTube URL."""
    if is_youtube_url(url):
        return extract_youtube_id(url)
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
    atomic_write(path, json.dumps(asdict(info), ensure_ascii=False, indent=2))


def load_episode_info(episode_id: str) -> EpisodeInfo | None:
    """Load persisted episode metadata, or None if not found."""
    path = _episode_info_path(episode_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return EpisodeInfo(**data)


def fetch_youtube_episode(url: str) -> EpisodeInfo:
    """Fetch YouTube video metadata via yt-dlp without downloading media."""
    episode_id = extract_youtube_id(url)
    video_id = episode_id.removeprefix("yt_")
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        import yt_dlp
    except ImportError as e:
        raise PodmindError(
            "yt-dlp is not installed. Run: pip install yt-dlp\n"
            f"Original error: {e}"
        ) from e

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            data = ydl.extract_info(canonical_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise PodmindError(f"Failed to fetch YouTube video metadata: {e}") from e

    if not isinstance(data, dict):
        raise PodmindError("Could not parse YouTube video metadata")

    info = EpisodeInfo(
        episode_id=episode_id,
        title=str(data.get("title") or ""),
        podcast_name=str(data.get("channel") or data.get("uploader") or ""),
        audio_url=canonical_url,
        duration_sec=float(data.get("duration") or 0),
        source="youtube",
    )
    save_episode_info(info)
    return info


def _fetch_xiaoyuzhou_episode(url: str) -> EpisodeInfo:
    """Scrape episode metadata and audio URL from a Xiaoyuzhou FM page."""
    episode_id = extract_episode_id(url)
    page_url = f"https://www.xiaoyuzhoufm.com/episode/{episode_id}"

    resp = _session_with_retry().get(page_url, headers=HEADERS, timeout=30)
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
        source="xiaoyuzhou",
    )
    save_episode_info(info)
    return info


def fetch_episode(url: str) -> EpisodeInfo:
    """Fetch metadata for a Xiaoyuzhou FM episode or YouTube video."""
    if is_youtube_url(url):
        return fetch_youtube_episode(url)
    return _fetch_xiaoyuzhou_episode(url)
