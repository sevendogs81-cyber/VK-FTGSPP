# Install

The usual path is one script from a clone:

```bash
git clone https://github.com/sevendogs81-cyber/TurboGS.git
cd TurboGS
bash scripts/setup.sh
source .venv/bin/activate
```

Building the wheel compiles C++ and every shader. That step does **not** need
a GPU. `ftgspp doctor`, `train`, and `eval` do. CUDA ordinals and Vulkan
device indices can differ; `--devices` is the Vulkan id from `doctor`, not
`nvidia-smi`.

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

Each setup run synchronizes the selected profile. Use `--with-init` or pass
both component flags in the same command when both must remain installed.

Release wheels are not published to PyPI. YAML examples live in the **clone**;
the wheel only provides the `ftgspp` command.

## What the script does

1. Installs `build-essential`, `git`, `curl`, `libvulkan-dev`, and `libglm-dev`
   with `apt-get` when they are missing (unless `--skip-apt`). A
   `--with-viewer` build also installs the required X11 development headers.
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
| System/build | Linux x86_64, `g++`/`make`, Git, curl/CA certificates, tar/coreutils, Vulkan headers and loader, GLM | Yes, through `apt-get` when needed; otherwise root/sudo or `--skip-apt` is required |
| Shader build | Slang/`slangc` 2026.10 | Yes, pinned with SHA-256 verification |
| Python/build | CPython 3.12.x, uv 0.11.29, scikit-build-core, pybind11 3.0.1, CMake 3.24+ | Yes |
| Core runtime | NumPy 2.x, SciPy, OpenCV headless 4.12, pycolmap 3.12, OmegaConf, tqdm | Yes |
| Native training | NVIDIA driver and a Vulkan device accepted by `ftgspp doctor` | Driver/GPU must already exist |
| BiRefNet-HR masks | PyTorch/TorchVision, Transformers, Safetensors, timm, Kornia, einops, and pinned `ZhengPeng7/BiRefNet_HR` | With `--with-birefnet` or `--with-init`; model weights require network or a populated Hugging Face cache |
| DA3 points/carving | PyTorch/TorchVision, pinned `depth-anything-3==0.1.1`, Addict, evo, ImageIO/MoviePy, Pillow, plyfile, and trimesh | With `--with-da3` or `--with-init`; model weights require network or a populated cache |
| Locked final-test LPIPS | PyTorch, TorchMetrics and TorchVision | With `--with-lpips`; only needed for `ftgspp eval --lpips` |
| SSH browser viewer | websockets 15.x | With `--with-remote`; server remains loopback-only and display-free |
| Local desktop viewer | GLFW 3.4, Dear ImGui v1.91.9b and X11 development headers | With `--with-viewer`; archives are fetched by CMake using pinned SHA-256 values |
| TensorBoard | TensorBoard | No; install `.[tensorboard]` |

The pinned BiRefNet-HR repository publishes a broad
[training/application requirements file](https://huggingface.co/ZhengPeng7/BiRefNet_HR/blob/a7a562f6fd16021180f2f4348f4de003a2d3d1e1/requirements.txt)
that includes `numpy<2`. This project does not install that file. Its
`birefnet` extra declares the direct imports used by the pinned inference
module (`torchvision`, timm, Kornia, and einops) while retaining the project's
tested NumPy 2/OpenCV 4.12 core.

The audited upstream `depth-anything-3==0.1.1` distribution declares its full
demo/server/training environment as hard dependencies, including `numpy<2`.
FTGSPP uses only the inference API and is tested with NumPy 2/OpenCV 4.12. The
uv lock therefore supplies reviewed static metadata for that exact DA3 version
and explicitly declares the smaller inference dependency set in the `da3`
extra. This avoids unrelated Open3D, web-server, pre-commit, and
duplicate OpenCV packages while keeping DA3 itself version-pinned.

## Manual install

Use this only if you cannot run `scripts/setup.sh`. The pins must stay the
same.

```bash
git clone https://github.com/sevendogs81-cyber/TurboGS.git
cd TurboGS

sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates tar coreutils libvulkan-dev libglm-dev

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

## Troubleshooting

The JSON `name` from `doctor` should be an NVIDIA GPU, not `llvmpipe`. Use
that object's `device_id` as `--devices`. CUDA ordinals from `nvidia-smi` and
Vulkan ids often differ: on a two-GPU Linux host, Vulkan index `1` can be
`llvmpipe` while `nvidia-smi` GPU `1` is the second NVIDIA card. Doctor opens
a real device; it fails on a CPU-only build machine even when the wheel
compiled. `scripts/setup.sh` treats a doctor failure as a warning so a CPU
host can still finish the install.

If `slangc: not found`, Python/CMake cannot find 3.12, or Vulkan headers are
missing, stop and fix that step. Do not pass `--devices` with two ids: the
first release trains on one GPU.

Optional extras (`[remote]`, `[lpips]`, `[birefnet]`, `[da3]`, and
`[tensorboard]`) install into the same venv. `scripts/setup.sh --with-init` is a
convenience profile that selects both `[birefnet]` and `[da3]` without
duplicating their dependency declarations. The locked `[da3]` extra includes
the pinned upstream package and reviewed inference dependencies.

```bash
uv sync --locked --no-dev --no-editable --extra da3
```

LPIPS alone can be added to the core environment with:

```bash
uv sync --locked --no-dev --no-editable --extra lpips
```

Keep every required extra in the same exact sync command. For raw-RGB
initialization plus final-test LPIPS, use:

```bash
uv sync --locked --no-dev --no-editable \
  --extra birefnet --extra da3 --extra lpips
```

The first scored LPIPS batch downloads the roughly 528 MiB TorchVision VGG16
weights when they are absent from the Torch cache (normally
`~/.cache/torch/hub/checkpoints`). Training and ordinary validation never load
TorchMetrics or these weights.

See [usage.md](usage.md) for when each extra is required.
