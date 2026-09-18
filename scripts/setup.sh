#!/usr/bin/env bash
# One-command install for FastGS. Pins match .github/workflows/artifacts.yml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SLANG_VERSION="2026.10"
SLANG_TARBALL="slang-${SLANG_VERSION}-linux-x86_64-glibc-2.27.tar.gz"
SLANG_URL="https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/${SLANG_TARBALL}"
SLANG_SHA256="beddadd9b48151aa2d7b556056ea02b9826bde59d4d4fa827b33c5184e42a983"
UV_VERSION="0.11.29"
SLANG_PREFIX="${FTGSPP_SLANG_PREFIX:-$HOME/.local/slang-${SLANG_VERSION}}"
SKIP_APT=0
SKIP_DOCTOR=0
DOCTOR_DEVICE=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup.sh [options]

Install system headers if needed, pin Slang 2026.10, create .venv with
Python 3.12, build the native wheel, and run ftgspp doctor.

Options:
  --skip-apt       Do not call apt-get (headers must already be present)
  --skip-doctor    Skip the GPU check after install
  --device N       Vulkan device id passed to ftgspp doctor (default: 0)
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-apt) SKIP_APT=1; shift ;;
    --skip-doctor) SKIP_DOCTOR=1; shift ;;
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
if ! command -v g++ >/dev/null 2>&1; then
  need_apt+=(build-essential)
fi
if [[ ! -e /usr/include/vulkan/vulkan.h ]]; then
  need_apt+=(libvulkan-dev)
fi
if ((${#need_apt[@]})) && [[ "$SKIP_APT" -eq 0 ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "error: install ${need_apt[*]} (no apt-get on PATH)" >&2
    exit 1
  fi
  sudo apt-get update
  sudo apt-get install -y "${need_apt[@]}"
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

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is not on PATH after install" >&2
  exit 1
fi

uv python install 3.12
if [[ ! -d "$ROOT/.venv" ]]; then
  uv venv --python 3.12
fi

activate="$ROOT/.venv/bin/activate"
marker="# FastGS: Slang compiler for native rebuilds"
if ! grep -qF "$marker" "$activate"; then
  cat >> "$activate" <<EOF

$marker
export PATH="$SLANG_BIN_DIR:\$PATH"
EOF
fi

echo "building and installing ftgspp into $ROOT/.venv"
uv pip install --python "$ROOT/.venv/bin/python" "$ROOT"

if [[ "$SKIP_DOCTOR" -eq 0 ]]; then
  if ! "$ROOT/.venv/bin/ftgspp" doctor --devices "$DOCTOR_DEVICE"; then
    echo "warning: the wheel is installed, but doctor could not open Vulkan device ${DOCTOR_DEVICE}." >&2
    echo "warning: that is expected on a CPU-only machine. On a GPU host, check the NVIDIA driver and retry ftgspp doctor." >&2
  fi
fi

cat <<EOF

FastGS is installed.

  source $ROOT/.venv/bin/activate
  ftgspp prepare dataset --images data/my_capture/images --fps 30 --eval-cameras 3
  ftgspp train --config data/my_capture/ftgspp.yaml --devices ${DOCTOR_DEVICE}

Keep this venv activated for later ftgspp commands. See docs/usage.md.
EOF
