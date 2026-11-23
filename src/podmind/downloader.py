import os
import tempfile
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .config import (
    AUDIO_DIR,
    PodmindError,
    ensure_dirs,
    is_youtube_episode_id,
    validate_episode_id,
)
from .scraper import is_youtube_url


def audio_path(episode_id: str) -> Path:
    validate_episode_id(episode_id)
    return AUDIO_DIR / f"{episode_id}.m4a"


def _download_youtube_audio(episode_id: str, video_url: str, path: Path) -> Path:
    """Download best audio from YouTube as .m4a via yt-dlp."""
    try:
        import yt_dlp
    except ImportError as e:
        raise PodmindError(
            "yt-dlp is not installed. Run: pip install yt-dlp\n"
            f"Original error: {e}"
        ) from e

    bar: Any = None
    last_bytes = 0

    def _progress_hook(status: dict[str, Any]) -> None:
        nonlocal bar, last_bytes
        state = status.get("status")
        if state == "downloading":
            total = int(
                status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            )
            downloaded = int(status.get("downloaded_bytes") or 0)
            if bar is None:
                bar = tqdm(
                    desc=episode_id,
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
            elif total and bar.total != total:
                bar.total = total
            delta = downloaded - last_bytes
            if delta > 0:
                bar.update(delta)
                last_bytes = downloaded
        elif state == "finished" and bar is not None:
            bar.close()
            bar = None

    with tempfile.TemporaryDirectory(
        dir=str(path.parent), prefix=path.name + ".part."
    ) as tmp_dir:
        out_template = str(Path(tmp_dir) / "audio.%(ext)s")
        ydl_opts: dict[str, Any] = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": out_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "progress_hooks": [_progress_hook],
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                    "preferredquality": "0",
                }
            ],
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except yt_dlp.utils.DownloadError as e:
            raise PodmindError(f"YouTube audio download failed: {e}") from e
        finally:
            if bar is not None:
                bar.close()

        downloaded_path = Path(tmp_dir) / "audio.m4a"
        if not downloaded_path.exists():
            candidates = sorted(Path(tmp_dir).glob("audio.*"))
            if candidates:
                downloaded_path = candidates[0]

        if not downloaded_path.exists() or downloaded_path.stat().st_size == 0:
            raise RuntimeError("Download produced an empty file")

        downloaded_path.replace(path)

    return path


def download_audio(episode_id: str, audio_url: str, force: bool = False) -> Path:
    """Download podcast or YouTube audio, skipping if already downloaded (non-empty)."""
    path = audio_path(episode_id)

    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"Audio already downloaded: {path}")
        return path

    ensure_dirs()

    if is_youtube_episode_id(episode_id) or is_youtube_url(audio_url):
        return _download_youtube_audio(episode_id, audio_url, path)

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods={"GET"},
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    resp = session.get(audio_url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    block_size = 1024 * 1024

    # Write to .part temp file, then atomic rename
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".part.")
    try:
        with open(fd, "wb") as f, tqdm(
            desc=episode_id,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=block_size):
                f.write(chunk)
                bar.update(len(chunk))
    except BaseException:
        os.unlink(tmp)
        raise

    # Validate download: must be non-empty, and match content-length if provided
    downloaded = Path(tmp).stat().st_size
    if downloaded == 0:
        os.unlink(tmp)
        raise RuntimeError("Download produced an empty file")
    if total > 0 and downloaded < total:
        os.unlink(tmp)
        raise RuntimeError(
            f"Download incomplete: expected {total} bytes, got {downloaded}"
        )

    Path(tmp).replace(path)
    return path
