"""Deterministic identities for runtime-affecting source files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


_SOURCE_SUFFIXES = {
    ".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".slang", ".glsl",
    ".yaml", ".yml", ".toml", ".lock",
}
def repository_root() -> Path:
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "src" / "ftgspp").is_dir():
        return candidate
    # Installed wheels do not have a checkout. The package directory is still
    # fingerprintable and is the recoverable source available to that host.
    return Path(__file__).resolve().parents[1]


def runtime_source_files(root: Path | None = None) -> list[Path]:
    base = repository_root() if root is None else Path(root).resolve()
    roots = [base / "src" / "ftgspp", base / "native" / "src", base / "native" / "slang"]
    paths: set[Path] = set()
    for subtree in roots:
        if subtree.is_dir():
            paths.update(
                path for path in subtree.rglob("*")
                if path.is_file() and path.suffix.lower() in _SOURCE_SUFFIXES
            )
    for relative in (
        "native/CMakeLists.txt", "native/compile_shaders.py",
        "pyproject.toml", "uv.lock",
    ):
        path = base / relative
        if path.is_file():
            paths.add(path)
    if not paths and base.name == "ftgspp":
        paths.update(path for path in base.rglob("*.py") if path.is_file())
    return sorted(paths, key=lambda path: path.relative_to(base).as_posix())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_identity(root: Path | None = None) -> dict[str, object]:
    base = repository_root() if root is None else Path(root).resolve()
    files: dict[str, str] = {}
    for path in runtime_source_files(base):
        relative = path.relative_to(base).as_posix()
        try:
            files[relative] = _sha256_bytes(path.read_bytes())
        except OSError:
            files[relative] = "unknown"
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "ftgspp-source-tree-v1",
        "root": str(base),
        "set_sha256": _sha256_bytes(encoded) if files else "unknown",
        "file_count": len(files),
    }
