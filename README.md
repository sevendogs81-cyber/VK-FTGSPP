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

- Linux x86_64 and an NVIDIA driver that exposes Vulkan compute
- **Python 3.12.x** (3.11 and 3.13 will fail the native build)
- C++17 compiler (`g++`) and Vulkan headers/loader (`libvulkan-dev`)
- Slang **2026.10** with `slangc` on `PATH` (the setup script pins this)
- [uv](https://docs.astral.sh/uv/) (CI uses 0.11.29)

The core wheel is Torch-free. Optional extras add Torch only when you ask.

## Installation

```bash
git clone https://github.com/sevendogs81-cyber/FastGS.git
cd FastGS
bash scripts/setup.sh
source .venv/bin/activate
```

That script installs missing headers, pins Slang 2026.10 and uv 0.11.29 to the
same versions as CI, creates `.venv`, builds the native wheel, and runs
`ftgspp doctor`. Keep the venv activated for every later `ftgspp` command.
Manual steps and troubleshooting are in [docs/install.md](docs/install.md).

## First training run

Work from the clone. Index an existing rectangular multi-view grid (pixels
stay in place):

```text
data/my_capture/images/<camera_id>/<numeric-frame>.(png|jpg|...)
```

```bash
ftgspp prepare dataset --images data/my_capture/images --fps 30 --eval-cameras 3
ftgspp train --config data/my_capture/ftgspp.yaml --devices 0
```

`prepare` writes a starter YAML and prints the next `train` command. Replace
`3` with a hold-out camera that already exists under `images/`, or omit
`--eval-cameras`. `--devices` is the Vulkan id from `ftgspp doctor`, not
`nvidia-smi`.

`init.backend: auto` needs a complete keyframe RGB PLY set under
`data/my_capture/points_stride10/`, or the optional DA3 stack. Without
either, training stops at initialization. Full data rules, resume, eval,
export, foreground, and extras: [docs/usage.md](docs/usage.md).

## Next

- [Foreground, DA3, SelfCap, research YAML](docs/usage.md)
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
uv sync --python 3.12 --extra dev --no-install-project
uv run --no-sync pytest
```

Use this only when changing code. It rewrites `.venv` from the lockfile and
does **not** install the wheel; CPU tests skip `_ftgspp_vk` when it is
missing. Keep the Install venv for running `ftgspp`. Shader or C++ changes
need `slangc` on `PATH` and `uv pip install .` (or `uv build --python 3.12`).
CPU contract tests and native GPU numerical tests are engineering-correctness
gates, not evidence of reconstruction quality.
