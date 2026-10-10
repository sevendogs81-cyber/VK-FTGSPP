#!/usr/bin/env bash
# One-command install for VK-FTGSPP. Pins match .github/workflows/artifacts.yml.
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
RESET_PROFILE=0
PROFILE_PATH="$ROOT/.venv/ftgspp-install-profile"

source "$ROOT/scripts/setup_profile.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup.sh [options]

Install system build and FFmpeg dependencies, pin Slang 2026.10 and uv 0.11.29,
create .venv with Python 3.12, install the lightweight Vulkan training runtime,
build the native extension, and run ftgspp doctor. Prepared initialization artifacts
(including foreground masks and the point-cloud set required by the config)
are required unless an initialization option is selected.

Options:
  --with-viewer     Build the optional GLFW/ImGui desktop viewer (X11)
  --with-remote     Install the lightweight headless browser viewer server
  --with-init       Install BiRefNet-HR and DA3 for raw-RGB initialization
  --with-birefnet   Install BiRefNet-HR mask generation
  --with-da3        Install DA3 point generation
  --with-lpips      Install optional locked final-test LPIPS dependencies
  --with-tensorboard  Install optional TensorBoard tools
  --reset-profile  Replace saved optional components with only this command's flags
  --skip-apt        Do not call apt-get (system tools and Vulkan/X11 headers must be present)
  --skip-doctor     Skip the GPU check after install
  --device N        Vulkan device id passed to ftgspp doctor (default: 0)
  -h, --help        Show this help

Successful installs remember optional components in .venv. Later setup runs
retain them and add the requested options. Use --reset-profile to remove extras.
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
    --with-tensorboard) add_extra tensorboard; shift ;;
    --reset-profile) RESET_PROFILE=1; shift ;;
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

if [[ "$RESET_PROFILE" -eq 0 ]]; then
  load_install_profile "$PROFILE_PATH"
fi
if [[ ! "$DOCTOR_DEVICE" =~ ^[0-9]+$ ]]; then
  echo "error: --device needs a nonnegative integer" >&2
  exit 2
fi

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "error: this release targets Linux x86_64" >&2
  exit 1
fi
required_glibc_minor=28
for extra in "${INSTALL_EXTRAS[@]}"; do
  if [[ "$extra" == "tensorboard" ]]; then
    required_glibc_minor=31
  fi
done
libc_version="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
if [[ ! "$libc_version" =~ ^glibc[[:space:]]+([0-9]+)\.([0-9]+)$ ]] || \
   ((10#${BASH_REMATCH[1]:-0} < 2 || \
     (10#${BASH_REMATCH[1]:-0} == 2 && 10#${BASH_REMATCH[2]:-0} < required_glibc_minor))); then
  echo "error: this install profile requires glibc 2.${required_glibc_minor} or newer (found ${libc_version:-unknown})" >&2
  exit 1
fi

need_apt=()
if ! command -v g++ >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
  need_apt+=(build-essential)
fi
if ! command -v curl >/dev/null 2>&1; then
  need_apt+=(curl)
fi
if [[ ! -s /etc/ssl/certs/ca-certificates.crt ]]; then
  need_apt+=(ca-certificates)
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
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  need_apt+=(ffmpeg)
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

if ! printf '#include <optional>\nint main() { return std::optional<int>{0}.value(); }\n' | \
   g++ -std=c++17 -x c++ -fsyntax-only - >/dev/null 2>&1; then
  echo "error: a C++17-capable g++ and standard library are required" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1 || \
   ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -E 'libx264[[:space:]]' >/dev/null; then
  echo "error: FFmpeg with ffprobe and the libx264 encoder is required for SelfCap preparation and MP4 export" >&2
  echo "error: install the distribution's ffmpeg package or select a complete FFmpeg build on PATH" >&2
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
  curl --retry 3 --retry-delay 2 --connect-timeout 30 -fsSL "$SLANG_URL" -o "$tarball"
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
marker="# VK-FTGSPP: Slang compiler for native rebuilds"
# Recognize earlier project names without duplicating the compiler PATH entry.
if ! grep -qE "^# .*: Slang compiler for native rebuilds$" "$activate"; then
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
FTGSPP_INSTALL_EXTRAS="$install_extras_csv" FTGSPP_INSTALL_VIEWER="$WITH_VIEWER" \
  "$ROOT/.venv/bin/python" "$ROOT/scripts/verify_install.py"

save_install_profile "$PROFILE_PATH"

if [[ "$SKIP_DOCTOR" -eq 0 ]]; then
  if ! "$ROOT/.venv/bin/ftgspp" doctor --devices "$DOCTOR_DEVICE"; then
    echo "warning: the wheel is installed, but doctor could not open Vulkan device ${DOCTOR_DEVICE}." >&2
    echo "warning: that is expected on a CPU-only machine. On a GPU host, check the NVIDIA driver and retry ftgspp doctor." >&2
  fi
fi

cat <<EOF

VK-FTGSPP training runtime is installed.

  source $ROOT/.venv/bin/activate
EOF

if ((${#INSTALL_EXTRAS[@]})); then
  echo "Optional extras: $install_extras_csv"
fi
if [[ ",$install_extras_csv," != *,birefnet,* && ",$install_extras_csv," != *,da3,* ]]; then
  echo "Use prepared initialization artifacts, or rerun with --with-init."
fi
echo "See docs/usage.md for dataset preparation and training commands."
