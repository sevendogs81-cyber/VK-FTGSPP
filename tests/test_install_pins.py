from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
