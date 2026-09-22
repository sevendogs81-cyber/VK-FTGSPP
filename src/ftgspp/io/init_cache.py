"""Initialization-cache helpers for atomic L2 publication."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ftgspp.io.hashing import file_sha256

DA3_SCHEMA = "ftgspp-da3-foreground-v1"
DA3_CACHE_POINTER_SCHEMA = "ftgspp-da3-cache-pointer-v1"


def canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint_payload(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@contextmanager
def namespace_lock(directory: Path) -> Iterator[None]:
    """Exclusive lock for one cache namespace publish/check cycle."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".publish.lock"
    handle = lock_path.open("a+b")
    try:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        handle.close()


def resolve_published_cache_root(root: Path) -> Path:
    """Resolve an atomically published cache namespace, with legacy fallback."""

    base = Path(root)
    pointer = base / "current.json"
    if not pointer.is_file():
        return base
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return base
    if not isinstance(payload, dict):
        return base
    directory = payload.get("directory")
    if (
        payload.get("schema") != DA3_CACHE_POINTER_SCHEMA
        or not isinstance(directory, str)
        or not directory
        or directory in {".", ".."}
        or Path(directory).name != directory
    ):
        return base
    candidate = base / directory
    manifest = candidate / "manifest.json"
    completed = candidate / "completed.json"
    if not manifest.is_file() or not completed.is_file():
        return base
    try:
        manifest_sha256 = file_sha256(manifest)
    except OSError:
        return base
    if payload.get("manifest_sha256") != manifest_sha256:
        return base
    return candidate


def publish_cache_pointer(
    base: Path, cache_root: Path, manifest_path: Path,
) -> None:
    pointer = Path(base) / "current.json"
    write_json_atomic(pointer, {
        "schema": DA3_CACHE_POINTER_SCHEMA,
        "directory": cache_root.name,
        "manifest_sha256": file_sha256(manifest_path),
    })


def published_cache_candidates(base: Path) -> list[Path]:
    """Try the active cache first, then completed sibling namespaces only."""

    base = Path(base)
    candidates = [resolve_published_cache_root(base), base]
    if base.is_dir():
        candidates.extend(sorted(
            child for child in base.iterdir()
            if not child.name.startswith(".") and child.is_dir()
            and (child / "manifest.json").is_file()
            and (child / "completed.json").is_file()
        ))
    return list(dict.fromkeys(candidates))


def write_json_atomic(path: Path, payload: dict[str, object]) -> Path:
    path = Path(path)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Different namespaces may publish from threads in the same process.
    # A PID alone is not a unique temporary filename.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
