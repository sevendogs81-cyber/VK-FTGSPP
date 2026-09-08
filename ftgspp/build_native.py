"""Build the bundled shaders and native Python module in-place."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _cache_path(cache: Path, key: str) -> Path | None:
    if not cache.exists():
        return None
    prefix = f"{key}:"
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            _, value = line.split("=", 1)
            return Path(value)
    return None


def _first_cmake_package(candidates: list[Path], config: str) -> Path | None:
    for candidate in candidates:
        if (candidate / config).exists():
            return candidate
    return None


def build(
    *,
    slangc: Path | None = None,
    glslc: Path | None = None,
    skip_shaders: bool = False,
) -> Path:
    root = Path(__file__).resolve().parents[1]
    if not skip_shaders:
        shader_command = [sys.executable, "native/compile_shaders.py", "--force"]
        if slangc is not None:
            shader_command += ["--slangc", str(slangc)]
        if glslc is not None:
            shader_command += ["--glslc", str(glslc)]
        _run(shader_command, root)

    build_dir = root / "build" / "native"
    cache = build_dir / "CMakeCache.txt"
    cached_paths = [
        _cache_path(cache, "CMAKE_MAKE_PROGRAM"),
        _cache_path(cache, "Vulkan_INCLUDE_DIR"),
    ]
    if any(path is not None and not path.exists() for path in cached_paths):
        shutil.rmtree(build_dir, ignore_errors=True)
    configure = [
        "cmake",
        "-S",
        str(root / "native"),
        "-B",
        str(build_dir),
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    ]
    try:
        import pybind11

        configure.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
    except ImportError:
        pybind11_dir = _first_cmake_package(
            sorted(
                (root / ".venv").glob(
                    "lib/python*/site-packages/pybind11/share/cmake/pybind11"
                )
            ),
            "pybind11Config.cmake",
        )
        if pybind11_dir is not None:
            configure.append(f"-Dpybind11_DIR={pybind11_dir}")
        # CMake's pinned FetchContent fallback remains available.
    glm_dir = _first_cmake_package(
        [
            root / "build" / "native" / "_deps" / "glm-src" / "cmake" / "glm",
            root / "build" / "_deps" / "glm-src" / "cmake" / "glm",
        ],
        "glmConfig.cmake",
    )
    if glm_dir is not None:
        configure.append(f"-Dglm_DIR={glm_dir}")
    ninja = shutil.which("ninja")
    if ninja:
        configure += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    vulkan_include = Path("/usr/include")
    vulkan_library = Path("/usr/lib/x86_64-linux-gnu/libvulkan.so")
    if (vulkan_include / "vulkan" / "vulkan.h").exists():
        configure.append(f"-DVulkan_INCLUDE_DIR={vulkan_include}")
    if vulkan_library.exists():
        configure.append(f"-DVulkan_LIBRARY={vulkan_library}")
    _run(configure, root)
    _run(["cmake", "--build", str(build_dir), "--config", "Release"], root)

    candidates = list(build_dir.rglob("_ftgspp_vk*.pyd")) + list(
        build_dir.rglob("_ftgspp_vk*.so")
    )
    if not candidates:
        raise FileNotFoundError("CMake completed but _ftgspp_vk was not produced")
    destination = root / candidates[0].name
    shutil.copy2(candidates[0], destination)
    print(f"installed native module: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slangc", type=Path)
    parser.add_argument("--glslc", type=Path)
    parser.add_argument(
        "--skip-shaders",
        action="store_true",
        help="Rebuild/install only C++; use only when tracked SPIR-V is current",
    )
    args = parser.parse_args()
    build(
        slangc=args.slangc,
        glslc=args.glslc,
        skip_shaders=args.skip_shaders,
    )


if __name__ == "__main__":
    main()
