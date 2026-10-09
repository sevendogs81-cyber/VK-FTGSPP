# Installation

Before running setup, you need Linux x86_64 with glibc 2.28 or newer
(2.31 or newer with `--with-tensorboard`),
`apt-get` and root/sudo access (or all packages already present with
`--skip-apt`), and network access to fetch uv, Python, Slang, and locked Python
packages. Native training and rendering also need an installed NVIDIA Linux
driver with a Vulkan ICD and a GPU in the [runnable range](hardware.md); the
script does not install or replace the driver. A local desktop viewer needs an
X11/XWayland display backed by the selected GPU. Raw-RGB initialization and
LPIPS fetch their model weights on first use unless cached.

The script installs Python 3.12, pinned uv/Slang and the selected dependencies;
you do not need to activate a Conda environment or install those tools manually.
Choose the feature flags in the table below before the first install.

Run from a clone (the following is the core-only profile):

```bash
git clone https://github.com/sevendogs81-cyber/VK-FTGSPP.git
cd VK-FTGSPP
bash scripts/setup.sh
source .venv/bin/activate
```

Building the wheel compiles C++ and every shader. That step does **not** need
a GPU. `ftgspp doctor`, `train`, and `eval` do. CUDA ordinals and Vulkan
device indices can differ; `--devices` is the Vulkan ID from `doctor`, not
`nvidia-smi`.

Not every NVIDIA GPU is supported. Read the
[hardware compatibility matrix](hardware.md) before installation; the full
training path requires a physical NVIDIA device classified as `VIABLE`, rather
than `POSSIBLY VIABLE`, by `ftgspp doctor`.

The default installation is the lightweight Vulkan training runtime. It stays
Torch-free and is intended for users who already have the masks and keyframe
PLYs required by their configuration. Add the complete raw-RGB initialization
stack when needed:

```bash
bash scripts/setup.sh --with-init
```

The two initialization components can also be installed independently:
`--with-birefnet` adds BiRefNet-HR mask generation, while `--with-da3` adds DA3
point generation and CUDA-accelerated carving. The foreground pipeline mode is
a data/training choice; it is not an installation profile.

`--with-lpips` adds the optional locked final-test metric. It is composable with
every initialization profile, for example `bash scripts/setup.sh --with-init
--with-lpips`, and never enables LPIPS during training.
`--with-tensorboard` adds the optional research visualization tools.

Viewer options are independent of training and initialization. `--with-viewer`
builds the GLFW / Dear ImGui X11 window and installs its required X11 headers.
`--with-remote` installs the lightweight `websockets` dependency for the SSH
browser viewer; it needs no server display or desktop build. Headless
`ftgspp render` is part of the core install. The two viewer flags can be
combined with each other and with initialization or LPIPS flags.

| Available inputs | Setup command |
|---|---|
| Required masks and point-cloud set are already prepared | `bash scripts/setup.sh` |
| Point clouds are ready; masks must be generated | `bash scripts/setup.sh --with-birefnet` |
| Masks are ready; DA3 points or automatic carving bounds are needed | `bash scripts/setup.sh --with-da3` |
| Start from raw RGB | `bash scripts/setup.sh --with-init` |
| Add LPIPS to any profile | Add `--with-lpips` to the same setup command |
| Add the local desktop viewer | Add `--with-viewer` to the same setup command |
| Add the SSH browser viewer | Add `--with-remote` to the same setup command |
| Add TensorBoard tools | Add `--with-tensorboard` to the same setup command |

**Rerunning setup replaces the selected profile; it does not append features.**
Always repeat all required flags, including `--with-viewer` for a desktop build.
For example, to add the browser viewer to an initialization environment:

```bash
bash scripts/setup.sh --with-init --with-remote
source .venv/bin/activate
```

Running only `--with-remote` would remove the initialization dependencies, and
omitting `--with-viewer` rebuilds without the desktop module. Stop processes
using `.venv` before any setup or `uv sync`.
To install every public optional component in one pass, use:

```bash
bash scripts/setup.sh --with-init --with-viewer --with-remote \
  --with-lpips --with-tensorboard
```

The command builds the desktop module and installs the full optional Python
stack; model weights are fetched when their feature first runs.

Release wheels are not published to PyPI. YAML examples live in the **clone**;
the wheel only provides the `ftgspp` command.

## Verify installation

```bash
source .venv/bin/activate
ftgspp --help
ftgspp doctor --devices 0
```

Select the intended NVIDIA GPU from the printed Vulkan list. It must be
`VIABLE`; a successful wheel build alone is insufficient. Setup treats doctor
failure as a warning to allow CPU-only build hosts. On a training host, fix that
failure before continuing. Then run the [tiny end-to-end check](../README.md#check-the-complete-workflow).

## Shared servers and offline use

`--skip-apt` skips installation of system packages, not their requirements.
On a shared server without sudo, ask the administrator to provide the listed
compiler, Vulkan headers/loader and FFmpeg packages, then use `--skip-apt`
alongside your feature flags. This script does not bootstrap missing system
headers into a user-local directory.

Installing a wheel builds native code and may fetch toolchains/dependencies;
`--skip-apt` is not an offline switch. For offline inference, install the selected
environment and cache weights beforehand. Set `init.da3.model_path` for local
DA3 weights; BiRefNet and LPIPS also need their respective cached weights.
The core tiny fixture requires no inference-model download.

<details>
<summary>Dependency and build reference</summary>

## What the script does

1. Checks Linux x86_64 and the selected glibc minimum, then installs
   `build-essential`, `git`,
   `curl`, `ca-certificates`, `tar`,
   `coreutils`, `libvulkan-dev`, and FFmpeg with `apt-get` when
   they are missing (unless `--skip-apt`). It verifies C++17, `ffprobe`, and the
   `libx264` encoder for SelfCap video preparation and MP4 export. A
   `--with-viewer` build also installs the required X11 development headers.
   CMake downloads GLM 0.9.9.8 when its system headers are absent.
2. Downloads Slang **2026.10** to `$HOME/.local/slang-2026.10` when `slangc`
   2026.10 is not already on `PATH`, and checks the same tarball checksum as
   CI (`beddadd9b481…`).
3. Installs [uv](https://docs.astral.sh/uv/) **0.11.29** when that exact version
   is not already on `PATH`.
4. Creates `.venv` with Python **3.12.x**, appends Slang to that venv's
   `activate` script, then synchronizes the locked core environment plus the
   optional components selected on the command line, then rebuilds the wheel
   with the selected desktop-viewer setting.
5. Imports every selected component and rejects dependencies resolved from a
   foreign `.pth`/`PYTHONPATH` location outside `.venv`.
6. Runs `ftgspp doctor --devices 0` unless `--skip-doctor`.

`.venv` is gitignored. Keep it activated for every later `ftgspp` and
extra-install command.

The build needs CMake 3.24+; scikit-build-core can provide it when the system
copy is older.

## Installed components

| Layer | Required components | Installed by `setup.sh` |
|---|---|---|
| System/build | Linux x86_64, glibc 2.28+ (2.31+ with TensorBoard), C++17-capable `g++`/`make`, Git, curl/CA certificates, tar/coreutils, Vulkan headers and loader, GLM | System tools and Vulkan headers are installed through `apt-get` when needed; CMake fetches GLM if absent |
| Video input/export | `ffmpeg`, `ffprobe`, and the `libx264` encoder | Yes, through `apt-get` when needed, then checked on `PATH` |
| Shader build | Slang/`slangc` 2026.10 | Yes, pinned with SHA-256 verification |
| Python/build | CPython 3.12.x, uv 0.11.29, scikit-build-core, pybind11 3.0.1, CMake 3.24+ | Yes |
| Core runtime | NumPy 2.x, SciPy, OpenCV headless 4.12, pycolmap 3.12, OmegaConf, tqdm | Yes |
| Native training | NVIDIA Linux driver with Vulkan ICD and a device reported as `VIABLE` by `ftgspp doctor`; see [hardware.md](hardware.md) | Driver/GPU must already exist |
| BiRefNet-HR masks | PyTorch/TorchVision, Transformers, Safetensors, timm, Kornia, einops, and pinned `ZhengPeng7/BiRefNet_HR` | With `--with-birefnet` or `--with-init`; model weights require network or a populated Hugging Face cache |
| DA3 points/carving | PyTorch/TorchVision, pinned `depth-anything-3==0.1.1`, Addict, evo, ImageIO/MoviePy, Pillow, plyfile, and trimesh | With `--with-da3` or `--with-init`; model weights require network or a populated cache |
| Locked final-test LPIPS | PyTorch, TorchMetrics and TorchVision | With `--with-lpips`; only needed for `ftgspp eval --lpips` |
| SSH browser viewer | websockets 15.x | With `--with-remote`; server remains loopback-only and display-free |
| Local desktop viewer | GLFW 3.4, Dear ImGui v1.91.9b and X11 development headers | With `--with-viewer`; archives are fetched by CMake using pinned SHA-256 values |
| TensorBoard | TensorBoard | With `--with-tensorboard` |

The optional inference dependency pins and the DA3 metadata override are
explained in the [maintainer notes](../CONTRIBUTING.md#optional-inference-dependencies).

</details>

<details>
<summary>Manual installation (advanced)</summary>

## Manual install

Use this only if you cannot run `scripts/setup.sh`. The pins must stay the
same.

```bash
git clone https://github.com/sevendogs81-cyber/VK-FTGSPP.git
cd VK-FTGSPP

sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates tar coreutils libvulkan-dev libglm-dev ffmpeg

curl -fsSL https://github.com/shader-slang/slang/releases/download/v2026.10/slang-2026.10-linux-x86_64-glibc-2.27.tar.gz \
  -o /tmp/slang.tgz
echo "beddadd9b48151aa2d7b556056ea02b9826bde59d4d4fa827b33c5184e42a983  /tmp/slang.tgz" | sha256sum -c -
mkdir -p "$HOME/.local/slang-2026.10"
tar -xzf /tmp/slang.tgz -C "$HOME/.local/slang-2026.10"
export PATH="$HOME/.local/slang-2026.10/bin:$PATH"
slangc -v   # expect 2026.10

curl -LsSf https://astral.sh/uv/0.11.29/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv sync --locked --no-dev --no-editable
ftgspp doctor --devices 0
```

Check `ffprobe -version` and `ffmpeg -hide_banner -encoders` for `libx264`
before SelfCap preparation or MP4 export. Debian/Ubuntu's `ffmpeg` package
normally supplies all three.

For the SSH browser viewer, add `--extra remote` to the `uv sync` command.
For the local desktop viewer, install
`libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`, then build
with `-C cmake.define.FTGSPP_BUILD_VIEWER=ON`. See [viewer.md](viewer.md) for
the complete commands and runtime requirements.

For both raw-RGB initialization components, replace the `uv sync` command with:

```bash
uv sync --locked --no-dev --no-editable --extra birefnet --extra da3
```

`uv build --python 3.12` produces the core release wheel. A DA3 initialization
environment should be synchronized from the clone so uv applies the locked DA3
metadata correction; installing a standalone wheel extra cannot carry
project-level uv resolver metadata.

</details>

## Troubleshooting

The selected device from `doctor` must be an NVIDIA GPU classified as `VIABLE`,
not `POSSIBLY VIABLE`, `NOT VIABLE`, or `llvmpipe`. Use that device's JSON
`device_id` as `--devices`. CUDA ordinals from `nvidia-smi` and
Vulkan IDs often differ: on a two-GPU Linux host, Vulkan index `1` can be
`llvmpipe` while `nvidia-smi` GPU `1` is the second NVIDIA card. Doctor opens
a real device; it fails on a CPU-only build machine even when the wheel
compiled. `scripts/setup.sh` treats a doctor failure as a warning so a CPU
host can still finish the install. Passing the feature gate does not guarantee
enough VRAM for a particular run; see [hardware.md](hardware.md).

If `slangc: not found`, Python/CMake cannot find 3.12, or Vulkan headers are
missing, stop and fix that step. Do not pass `--devices` with two ids: the
first release trains on one GPU.

Use the setup flags above to select optional components. Advanced `uv sync`
commands must likewise repeat every required `--extra` and the desktop CMake
setting. Do not copy a single-extra sync command into an environment whose
other capabilities you want to retain.

The first scored LPIPS batch downloads the roughly 528 MiB TorchVision VGG16
weights when they are absent from the Torch cache (normally
`~/.cache/torch/hub/checkpoints`). Training and ordinary validation never load
TorchMetrics or these weights.

See [usage.md](usage.md) for when each extra is required.
