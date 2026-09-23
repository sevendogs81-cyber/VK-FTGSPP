# FastGS

A Linux/NVIDIA, Python 3.12 implementation of explicit-velocity 4D Gaussian
training. Clone this repository, run one setup script, then use the `ftgspp`
CLI. YAML examples live in the **clone**; the wheel only provides the command.
It is not a library you `import`.

**First release trains on one GPU.** One view produces one optimizer update.
Two-device synchronous training is **in development**: the pair code,
`dev/tests/gpu/` scripts, and the controlled GPU workflow stay in the tree for
engineering, but they are not a supported user path. Local gradient
accumulation is not part of the interface.

**License:** non-commercial research and educational use only, not an
OSI-approved or commercially permissive open-source license. Read [LICENSE](LICENSE)
and [third-party notices](THIRD_PARTY.md) before use or redistribution.

## Requirements

- Linux x86_64 and an NVIDIA GPU driver with Vulkan compute support
- **Python 3.12.x** (3.11 and 3.13 will fail the native build)
- `build-essential`, `git`, `curl`, `libvulkan-dev`, and `libglm-dev`
- Slang **2026.10** with `slangc` on `PATH` (the setup script pins this)
- [uv](https://docs.astral.sh/uv/) **0.11.29**

`ftgspp doctor` is the authoritative GPU compatibility check. Native training
requires 32-lane subgroups, shader Int16/Int64, timeline semaphores, and Vulkan
float32 buffer atomic add support.

## Installation

```bash
git clone https://github.com/sevendogs81-cyber/FastGS.git
cd FastGS
bash scripts/setup.sh
source .venv/bin/activate
```

The default is the lightweight, Torch-free Vulkan training runtime for users
with prepared masks and PLYs. Starting from raw RGB instead uses:

```bash
bash scripts/setup.sh --with-init
```

That adds BiRefNet-HR and DA3. `--with-birefnet` and `--with-da3` install only
one preparation capability. `--with-lpips` adds the optional locked final-test
metric without enabling it during training. Full component requirements are in
[docs/install.md](docs/install.md).

## First training run

Work from the clone. Index an existing rectangular multi-view grid (pixels
stay in place):

```text
data/my_capture/images/<camera_id>/<numeric-frame>.(png|jpg|...)
```

```bash
ftgspp prepare dataset --images data/my_capture/images --fps 30 --eval-cameras 3
```

`prepare` writes a starter YAML and prints the required initialization step.
Replace
`3` with a hold-out camera that already exists under `images/`, or omit
`--eval-cameras`. `--devices` is the Vulkan id from `ftgspp doctor`, not
`nvidia-smi`.

Scene-mode `init.backend: auto` needs a complete keyframe RGB PLY set under
`data/my_capture/points_stride10/`, or DA3 installed with `--with-da3` or
`--with-init`. Without prepared PLYs or locally/network-accessible DA3 weights,
training stops with an explicit initialization error. Full data rules, resume, eval,
export, foreground, and extras: [docs/usage.md](docs/usage.md).

After the configured point source exists:

```bash
ftgspp train --config data/my_capture/ftgspp.yaml --devices 0
```

## View and render results

Headless PNG and optional FFmpeg video export work with the default install:

```bash
ftgspp render --run /path/to/run --camera 0 --time 1.25 \
  --device 0 --output frames/still
```

For a local GLFW / Dear ImGui window, build the desktop option and open a run:

```bash
bash scripts/setup.sh --with-viewer
ftgspp viewer --run /path/to/run --device 0
```

On an SSH GPU server without a display session, install the browser server and
forward its loopback port:

```bash
bash scripts/setup.sh --with-remote
ftgspp viewer --remote --run /path/to/run --device 0
ssh -L 8765:127.0.0.1:8765 USER@SERVER
```

Open the exact tokenized URL printed by the server. Both viewers use the native
Vulkan 4DGS renderer and support capture/free cameras, playback, screenshots,
camera paths and export. See [viewer setup, controls and limits](docs/viewer.md).

## Next

- [Foreground, DA3, SelfCap, research YAML](docs/usage.md)
- [Local, SSH browser and headless viewers](docs/viewer.md)
- [SelfCap train-only speed](docs/performance.md)
- [Architecture](docs/architecture.md)
- [Migration from older FTGSPP layouts](docs/migration.md)
- [Maintainer notes](CONTRIBUTING.md)

## Repository layout

```text
FastGS/
├── scripts/setup.sh   # one-command install
├── examples/          # YAML templates (copy or generate via prepare)
├── docs/              # install, usage, architecture
├── src/ftgspp/        # Python CLI, installed as `ftgspp`
├── native/            # Vulkan + Slang sources compiled into the wheel
└── dev/               # tests and research tools; not a user path
```

## Development

```bash
uv sync --locked --python 3.12 --extra dev --no-install-project
uv run --no-sync pytest
```

Use this only when changing code. It rewrites `.venv` from the lockfile and
does **not** install the wheel; CPU tests skip `_ftgspp_vk` when it is
missing. Keep the Install venv for running `ftgspp`. Shader or C++ changes
need `slangc` on `PATH` and `uv pip install .` (or `uv build --python 3.12`).
CPU contract tests and native GPU numerical tests are engineering-correctness
gates, not evidence of reconstruction quality.
