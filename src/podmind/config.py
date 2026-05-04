import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class PodmindError(Exception):
    """Base exception for PodMind library errors.

    Library functions raise this instead of calling sys.exit.
    The CLI layer catches it and translates to an exit code.
    """


# Valid Xiaoyuzhou FM episode ID: 24-char hex string
_EPISODE_ID_RE = re.compile(r"^[0-9a-f]{24}$")


def validate_episode_id(eid: str) -> str:
    """Validate and return episode_id. Raises PodmindError if invalid."""
    if not _EPISODE_ID_RE.match(eid):
        raise PodmindError(f"Invalid episode ID: {eid!r} (expected 24 hex chars)")
    return eid


# Data directory — defaults to ./data relative to CWD, overridable via env
DATA_DIR = Path(os.getenv("PODMIND_DATA_DIR", Path.cwd() / "data"))

AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
EPISODES_DIR = DATA_DIR / "episodes"
OUTPUTS_DIR = DATA_DIR / "outputs"


def ensure_dirs() -> None:
    """Create data directories on demand (no side-effects at import time)."""
    for d in [AUDIO_DIR, TRANSCRIPTS_DIR, EPISODES_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# API config
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ASR config (overridable via env)
ASR_MODEL_ID = os.getenv("PODMIND_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
ASR_DEVICE = os.getenv("PODMIND_ASR_DEVICE", "mps")


# Maps ISO 639-1 codes / lowercase aliases to the full names qwen-asr expects.
_LANGUAGE_MAP: dict[str, str] = {
    "zh": "Chinese", "chinese": "Chinese",
    "en": "English", "english": "English",
    "yue": "Cantonese", "cantonese": "Cantonese",
    "ja": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "korean": "Korean",
    "ar": "Arabic", "arabic": "Arabic",
    "de": "German", "german": "German",
    "fr": "French", "french": "French",
    "es": "Spanish", "spanish": "Spanish",
    "pt": "Portuguese", "portuguese": "Portuguese",
    "id": "Indonesian", "indonesian": "Indonesian",
    "it": "Italian", "italian": "Italian",
    "ru": "Russian", "russian": "Russian",
    "th": "Thai", "thai": "Thai",
    "vi": "Vietnamese", "vietnamese": "Vietnamese",
    "tr": "Turkish", "turkish": "Turkish",
    "hi": "Hindi", "hindi": "Hindi",
    "ms": "Malay", "malay": "Malay",
    "nl": "Dutch", "dutch": "Dutch",
    "sv": "Swedish", "swedish": "Swedish",
    "da": "Danish", "danish": "Danish",
    "fi": "Finnish", "finnish": "Finnish",
    "pl": "Polish", "polish": "Polish",
    "cs": "Czech", "czech": "Czech",
    "fil": "Filipino", "filipino": "Filipino",
    "fa": "Persian", "persian": "Persian",
    "el": "Greek", "greek": "Greek",
    "ro": "Romanian", "romanian": "Romanian",
    "hu": "Hungarian", "hungarian": "Hungarian",
    "mk": "Macedonian", "macedonian": "Macedonian",
}

_SUPPORTED = sorted(set(_LANGUAGE_MAP.values()))


def validate_language(value: str | None) -> str | None:
    """Normalize a language code/name to the form qwen-asr expects.

    Returns the canonical language name (e.g. ``'zh'`` → ``'Chinese'``),
    or ``None`` to let the model auto-detect.
    Raises ``argparse.ArgumentTypeError`` for unsupported languages.
    """
    if value is None:
        return None
    key = value.lower()
    if key in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[key]
    raise argparse.ArgumentTypeError(
        f"Unsupported language: {value!r}. "
        f"Supported codes: {', '.join(sorted(k for k in _LANGUAGE_MAP if len(k) <= 3))}"
    )
