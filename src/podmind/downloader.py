import os
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

from .config import AUDIO_DIR, ensure_dirs, validate_episode_id


def audio_path(episode_id: str) -> Path:
    validate_episode_id(episode_id)
    return AUDIO_DIR / f"{episode_id}.m4a"


def download_audio(episode_id: str, audio_url: str, force: bool = False) -> Path:
    """Download podcast audio, skipping if already downloaded (non-empty)."""
    path = audio_path(episode_id)

    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"Audio already downloaded: {path}")
        return path

    ensure_dirs()

    resp = requests.get(audio_url, stream=True, timeout=120)
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
