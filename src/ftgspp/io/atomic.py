"""Publish file outputs only after the writer has completed successfully."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    """Yield a unique sibling path; preserve the destination on writer failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
