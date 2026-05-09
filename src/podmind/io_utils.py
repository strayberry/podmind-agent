"""Shared I/O utilities."""

import os
import tempfile
from pathlib import Path

from .config import ensure_dirs


def atomic_write(path: Path, content: str) -> None:
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
