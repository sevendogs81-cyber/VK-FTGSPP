"""Deterministic identities for runtime-affecting source files."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Sequence


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


def _git_patch_sha256(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ("git", "-C", str(root), "diff", "--binary", "--no-ext-diff", "HEAD", "--",
             "src", "native/src", "native/slang", "native/CMakeLists.txt",
             "native/compile_shaders.py", "pyproject.toml", "uv.lock"),
            check=True, capture_output=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _sha256_bytes(result.stdout)


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
        "files": files,
        "git_patch_sha256": _git_patch_sha256(base),
    }


def stage_identity(
    relative_files: Sequence[str],
    *,
    root: Path | None = None,
) -> dict[str, object]:
    """Content identity for one cache/init stage, not the whole source tree.

    Audit identity remains ``source_identity``. Stage keys decide whether a
    cached DA3 prediction, fused cloud, or Gaussian init may be reused.
    """

    base = repository_root() if root is None else Path(root).resolve()
    files: dict[str, str] = {}
    for relative in relative_files:
        relative_path = Path(relative)
        # Cache keys use checkout-relative names in both distributions. Wheels
        # contain ftgspp/ directly, without the checkout's src/ftgspp prefix.
        if base.name == "ftgspp" and not (base / "src" / "ftgspp").is_dir():
            try:
                relative_path = relative_path.relative_to("src/ftgspp")
            except ValueError:
                pass
        path = base / relative_path
        try:
            files[str(relative)] = _sha256_bytes(path.read_bytes()) if path.is_file() else "missing"
        except OSError:
            files[str(relative)] = "unknown"
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "ftgspp-stage-identity-v1",
        "set_sha256": _sha256_bytes(encoded) if files else "unknown",
        "file_count": len(files),
        "files": files,
    }
