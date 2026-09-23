#!/usr/bin/env bash
# One-command install for TurboGS. Pins match .github/workflows/artifacts.yml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SLANG_VERSION="2026.10"
SLANG_TARBALL="slang-${SLANG_VERSION}-linux-x86_64-glibc-2.27.tar.gz"
SLANG_URL="https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/${SLANG_TARBALL}"
SLANG_SHA256="beddadd9b48151aa2d7b556056ea02b9826bde59d4d4fa827b33c5184e42a983"
UV_VERSION="0.11.29"
SLANG_PREFIX="${FTGSPP_SLANG_PREFIX:-$HOME/.local/slang-${SLANG_VERSION}}"
WITH_VIEWER=0
SKIP_APT=0
SKIP_DOCTOR=0
DOCTOR_DEVICE=0
INSTALL_EXTRAS=()

add_extra() {
  local candidate="$1"
  local existing
  for existing in "${INSTALL_EXTRAS[@]}"; do
    [[ "$existing" == "$candidate" ]] && return
  done
  INSTALL_EXTRAS+=("$candidate")
}

usage() {
  cat <<'EOF'
Usage: bash scripts/setup.sh [options]

Install system build dependencies, pin Slang 2026.10 and uv 0.11.29, create
.venv with Python 3.12, install the lightweight Vulkan training runtime, build
the native extension, and run ftgspp doctor. Prepared initialization artifacts
(including foreground masks and the point-cloud set required by the config)
are required unless an initialization option is selected.

Options:
  --with-viewer     Build the optional GLFW/ImGui desktop viewer (X11)
  --with-remote     Install the lightweight headless browser viewer server
  --with-init       Install BiRefNet-HR and DA3 for raw-RGB initialization
  --with-birefnet   Install BiRefNet-HR mask generation
  --with-da3        Install DA3 point generation
  --with-lpips      Install optional locked final-test LPIPS dependencies
  --skip-apt        Do not call apt-get (headers must already be present)
  --skip-doctor     Skip the GPU check after install
  --device N        Vulkan device id passed to ftgspp doctor (default: 0)
  -h, --help        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-viewer) WITH_VIEWER=1; shift ;;
    --with-remote) add_extra remote; shift ;;
    --skip-apt) SKIP_APT=1; shift ;;
    --skip-doctor) SKIP_DOCTOR=1; shift ;;
    --with-init)
      add_extra birefnet
      add_extra da3
      shift
      ;;
    --with-birefnet) add_extra birefnet; shift ;;
    --with-da3) add_extra da3; shift ;;
    --with-lpips) add_extra lpips; shift ;;
    --device)
      [[ $# -ge 2 ]] || { echo "error: --device needs an integer" >&2; exit 2; }
      DOCTOR_DEVICE="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "error: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "error: this release targets Linux x86_64" >&2
  exit 1
fi

need_apt=()
if ! command -v g++ >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
  need_apt+=(build-essential)
fi
if ! command -v curl >/dev/null 2>&1; then
  need_apt+=(curl ca-certificates)
fi
if ! command -v git >/dev/null 2>&1; then
  need_apt+=(git)
fi
if ! command -v tar >/dev/null 2>&1; then
  need_apt+=(tar)
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  need_apt+=(coreutils)
fi
if [[ ! -e /usr/include/vulkan/vulkan.h ]]; then
  need_apt+=(libvulkan-dev)
fi
if [[ ! -e /usr/include/glm/glm.hpp ]]; then
  need_apt+=(libglm-dev)
fi
if [[ "$WITH_VIEWER" -eq 1 ]]; then
  for header_pkg in "X11/Xlib.h:libx11-dev" "X11/extensions/Xrandr.h:libxrandr-dev" "X11/extensions/Xinerama.h:libxinerama-dev" "X11/Xcursor/Xcursor.h:libxcursor-dev" "X11/extensions/XInput2.h:libxi-dev"; do
    [[ -f "/usr/include/${header_pkg%%:*}" ]] || need_apt+=("${header_pkg##*:}")
  done
fi
if ((${#need_apt[@]})) && [[ "$SKIP_APT" -eq 0 ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "error: install ${need_apt[*]} (no apt-get on PATH)" >&2
    exit 1
  fi
  apt=(apt-get)
  if [[ "$EUID" -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
      echo "error: install ${need_apt[*]} (root or sudo is required)" >&2
      exit 1
    fi
    apt=(sudo apt-get)
  fi
  "${apt[@]}" update
  "${apt[@]}" install -y "${need_apt[@]}"
elif ((${#need_apt[@]})); then
  echo "error: --skip-apt was set but missing ${need_apt[*]}" >&2
  exit 1
fi

slang_ok() {
  command -v slangc >/dev/null 2>&1 && slangc -v 2>&1 | grep -q "$SLANG_VERSION"
}

if slang_ok; then
  :
elif [[ -x "$SLANG_PREFIX/bin/slangc" ]]; then
  export PATH="$SLANG_PREFIX/bin:$PATH"
else
  tarball="$(mktemp "${TMPDIR:-/tmp}/slang.XXXXXX.tgz")"
  cleanup_tarball() { rm -f "$tarball"; }
  trap cleanup_tarball EXIT
  curl -fsSL "$SLANG_URL" -o "$tarball"
  echo "$SLANG_SHA256  $tarball" | sha256sum -c -
  mkdir -p "$SLANG_PREFIX"
  tar -xzf "$tarball" -C "$SLANG_PREFIX"
  trap - EXIT
  cleanup_tarball
  export PATH="$SLANG_PREFIX/bin:$PATH"
fi
if ! slang_ok; then
  echo "error: slangc ${SLANG_VERSION} is not on PATH (looked in $SLANG_PREFIX/bin)" >&2
  exit 1
fi
SLANG_BIN_DIR="$(cd "$(dirname "$(command -v slangc)")" && pwd)"
echo "using slangc: $(command -v slangc)"
slangc -v

uv_ok() {
  command -v uv >/dev/null 2>&1 && \
    [[ "$(uv --version | awk '{print $2}')" == "$UV_VERSION" ]]
}

if ! uv_ok; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! uv_ok; then
  echo "error: uv ${UV_VERSION} is not on PATH after install" >&2
  exit 1
fi

uv python install 3.12
if [[ -d "$ROOT/.venv" ]]; then
  if [[ ! -x "$ROOT/.venv/bin/python" ]] || \
     [[ "$("$ROOT/.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
    echo "error: $ROOT/.venv exists but is not a Python 3.12 environment" >&2
    echo "error: remove it or choose a different checkout before retrying" >&2
    exit 1
  fi
else
  uv venv --python 3.12
fi

activate="$ROOT/.venv/bin/activate"
marker="# TurboGS: Slang compiler for native rebuilds"
legacy_marker="# FastGS: Slang compiler for native rebuilds"
if ! grep -qF "$marker" "$activate" && ! grep -qF "$legacy_marker" "$activate"; then
  cat >> "$activate" <<EOF

$marker
export PATH="$SLANG_BIN_DIR:\$PATH"
EOF
fi

echo "building and installing ftgspp into $ROOT/.venv"
sync_args=(sync --locked --no-dev --no-editable --python "$ROOT/.venv/bin/python")
for extra in "${INSTALL_EXTRAS[@]}"; do
  sync_args+=(--extra "$extra")
done
sync_args+=(--reinstall-package ftgspp-vulkan-community)
if [[ "$WITH_VIEWER" -eq 1 ]]; then
  sync_args+=(--config-setting cmake.define.FTGSPP_BUILD_VIEWER=ON)
else
  sync_args+=(--config-setting cmake.define.FTGSPP_BUILD_VIEWER=OFF)
fi
uv "${sync_args[@]}"

install_extras_csv="$(IFS=,; echo "${INSTALL_EXTRAS[*]}")"
FTGSPP_INSTALL_EXTRAS="$install_extras_csv" "$ROOT/.venv/bin/python" - <<'PY'
import os
import sys
from importlib import import_module
from importlib.metadata import distribution
from pathlib import Path

required = {
    "ftgspp-vulkan-community": "ftgspp",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "opencv-python-headless": "cv2",
    "pycolmap": "pycolmap",
    "scipy": "scipy",
    "tqdm": "tqdm",
}
extras = {value for value in os.environ["FTGSPP_INSTALL_EXTRAS"].split(",") if value}
if "birefnet" in extras:
    required.update({
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "safetensors": "safetensors",
        "timm": "timm",
        "kornia": "kornia",
        "einops": "einops",
    })
if "da3" in extras:
    required.update({
        "depth-anything-3": "depth_anything_3.api",
        "torch": "torch",
        "torchvision": "torchvision",
        "safetensors": "safetensors",
        "einops": "einops",
        "addict": "addict",
        "evo": "evo",
        "imageio": "imageio",
        "moviepy": "moviepy",
        "pillow": "PIL",
        "plyfile": "plyfile",
        "trimesh": "trimesh",
        "huggingface-hub": "huggingface_hub",
    })
if "lpips" in extras:
    required.update({
        "torch": "torch",
        "torchvision": "torchvision",
        "torchmetrics": "torchmetrics",
    })
if "remote" in extras:
    required["websockets"] = "websockets"

venv = Path(sys.prefix).resolve()
versions = []
for package, module in required.items():
    item = distribution(package)
    location = Path(item.locate_file("")).resolve()
    if not location.is_relative_to(venv):
        raise RuntimeError(
            f"{package} resolved outside .venv: {location}; remove foreign "
            ".pth/PYTHONPATH injection and rerun setup"
        )
    import_module(module)
    versions.append(f"{package}={item.version}")
print("verified runtime:", " ".join(versions))

if "torch" in required:
    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
PY

if [[ "$SKIP_DOCTOR" -eq 0 ]]; then
  if ! "$ROOT/.venv/bin/ftgspp" doctor --devices "$DOCTOR_DEVICE"; then
    echo "warning: the wheel is installed, but doctor could not open Vulkan device ${DOCTOR_DEVICE}." >&2
    echo "warning: that is expected on a CPU-only machine. On a GPU host, check the NVIDIA driver and retry ftgspp doctor." >&2
  fi
fi

cat <<EOF

TurboGS training runtime is installed.

  source $ROOT/.venv/bin/activate
EOF

if ((${#INSTALL_EXTRAS[@]})); then
  echo "Optional extras: $install_extras_csv"
fi
if [[ ",$install_extras_csv," != *,birefnet,* && ",$install_extras_csv," != *,da3,* ]]; then
  echo "Use prepared initialization artifacts, or rerun with --with-init."
fi
echo "See docs/usage.md for dataset preparation and training commands."
