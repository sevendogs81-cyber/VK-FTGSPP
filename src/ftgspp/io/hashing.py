"""Streaming hashes for large run artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return a file's SHA256 without materializing it in memory."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()
