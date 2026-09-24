from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[2]
SLANG_VERSION = "2026.10"
SLANG_TARBALL = f"slang-{SLANG_VERSION}-linux-x86_64-glibc-2.27.tar.gz"
SLANG_SHA256 = "beddadd9b48151aa2d7b556056ea02b9826bde59d4d4fa827b33c5184e42a983"
UV_VERSION = "0.11.29"


def test_setup_script_pins_match_artifact_workflow():
    setup = (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")
    artifacts = (ROOT / ".github" / "workflows" / "artifacts.yml").read_text(
        encoding="utf-8"
    )
    assert f'SLANG_VERSION="{SLANG_VERSION}"' in setup
    assert "linux-x86_64-glibc-2.27.tar.gz" in setup
    assert SLANG_SHA256 in setup
    assert f'UV_VERSION="{UV_VERSION}"' in setup
    assert SLANG_TARBALL in artifacts
    assert SLANG_SHA256 in artifacts
    assert UV_VERSION in artifacts


def test_setup_separates_training_and_initialization_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    birefnet = {
        re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0]
        for requirement in extras["birefnet"]
    }
    assert {
        "torch", "torchvision", "transformers", "safetensors",
        "timm", "kornia", "einops",
    } <= birefnet
    da3 = {
        re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0]
        for requirement in extras["da3"]
    }
    assert {
        "torch", "torchvision", "safetensors", "einops",
        "depth-anything-3", "addict",
        "evo", "imageio", "moviepy", "pillow", "plyfile", "trimesh",
        "huggingface-hub",
    } <= da3
    assert "foreground" not in extras
    assert "torchmetrics" not in birefnet | da3
    assert extras["remote"] == ["websockets>=15,<16"]
    assert project["tool"]["uv"]["dependency-metadata"] == [{
        "name": "depth-anything-3",
        "version": "0.1.1",
        "requires-dist": [],
    }]

    setup = (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")
    assert "INSTALL_EXTRAS=()" in setup
    assert "sync --locked --no-dev --no-editable" in setup
    assert "--with-init)" in setup
    assert "--with-birefnet)" in setup
    assert "--with-da3)" in setup
    assert "--with-viewer)" in setup
    assert "--with-remote)" in setup
    assert "--with-tensorboard)" in setup
    assert "--core-only" not in setup
    assert '"depth-anything-3": "depth_anything_3.api"' in setup
    assert "resolved outside .venv" in setup


def test_setup_glibc_floor_covers_locked_binary_wheels():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))

    def wheel_floor(package_name: str) -> int:
        package = next(
            item for item in lock["package"] if item["name"] == package_name
        )
        floors = []
        for wheel in package["wheels"]:
            filename = wheel["url"].rsplit("/", 1)[-1]
            if "x86_64" not in filename or not any(
                tag in filename for tag in ("cp312-", "abi3-", "py3-none-")
            ):
                continue
            tags = [
                int(value) for value in re.findall(
                    r"manylinux_2_(\d+)_x86_64", filename
                )
            ]
            if "manylinux2014_x86_64" in filename:
                tags.append(17)
            if tags:
                floors.append(min(tags))
        assert floors, f"no Linux x86_64 wheel for {package_name}"
        return min(floors)

    setup = (ROOT / "scripts" / "setup.sh").read_text(encoding="utf-8")
    core_floor, tensorboard_floor = map(
        int, re.findall(r"required_glibc_minor=(\d+)", setup)
    )
    assert wheel_floor("pycolmap") <= core_floor
    assert wheel_floor("tensorboard-data-server") <= tensorboard_floor
