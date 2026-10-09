# Installation options

For the usual image-to-model workflow, use the [README installation](../README.md#install).
Setup creates `.venv` with Python 3.12 and installs the required build tools.
It does not install or replace your NVIDIA driver.

## Choose components

Run `bash scripts/setup.sh` with the options you need:

| Option | Adds |
|---|---|
| No flags | Core training and headless rendering; use prepared inputs |
| `--with-init` | BiRefNet masks and DA3 geometry/camera estimation |
| `--with-birefnet` / `--with-da3` | Just one preparation component |
| `--with-remote` | Browser viewer, including over SSH |
| `--with-viewer` | Linux desktop viewer; needs X11/XWayland |
| `--with-lpips` | Optional final-test perceptual metric |

You can add components later. Setup remembers earlier selections:

```bash
bash scripts/setup.sh --with-remote
source .venv/bin/activate
```

Stop jobs using `.venv` before updating it. `--reset-profile` replaces the saved
selection; older installations without a saved profile need their desired flags
specified once. See `bash scripts/setup.sh --help` for all options.

## Shared servers

Without sudo, ask your administrator to provide a C++17 compiler, Vulkan
headers/loader and FFmpeg with `ffprobe` and `libx264`, then use `--skip-apt`.
The supported platform is Linux x86_64 with glibc 2.28+; see
[hardware requirements](hardware.md) for GPU compatibility.

Setup needs network access for tools and packages. BiRefNet and DA3 download
weights when first used; prepare their caches before working offline. For
local DA3 weights, set `init.da3.model_path` in your YAML.

## Check the installation

```bash
source .venv/bin/activate
ftgspp doctor --devices 0
```

The intended NVIDIA GPU should be marked `VIABLE`. If setup finished but doctor
failed, fix the driver/device issue before training.

For a small execution check without data or model downloads, choose a new path:

```bash
python examples/make_tiny_scene.py --output /tmp/ftgspp-tiny
ftgspp train --config /tmp/ftgspp-tiny/ftgspp.yaml --devices 0
ftgspp render --run /tmp/ftgspp-tiny/run --camera 0 --time 0 \
  --device 0 --output /tmp/ftgspp-tiny/render
```

Success creates `run/gaussians.npz` and `render/000000.png`, a small synthetic
colored target. This checks execution, not real-scene reconstruction quality.

## Common problems

| Problem | Next step |
|---|---|
| No sudo / missing Vulkan headers | Have system dependencies installed, then use `--skip-apt` |
| Existing `.venv` uses another Python version | Use a fresh checkout; setup requires Python 3.12 |
| Wrong GPU or `llvmpipe` selected | Choose the NVIDIA Vulkan ID shown by doctor, not the `nvidia-smi` index |
| No `VIABLE` GPU | Check the driver and [supported hardware](hardware.md) |
| Missing optional package | Add its setup flag and reactivate `.venv` |

For manual builds and dependency pins, see [maintainer notes](../dev/CONTRIBUTING.md).
