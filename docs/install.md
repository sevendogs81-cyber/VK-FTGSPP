# Install

## Requirements

- Linux x86_64, glibc 2.28+, and a supported NVIDIA Vulkan GPU.
- An NVIDIA driver exposing Vulkan 1.2; setup does not install the driver.
- Network access for packages and, when first used, optional model weights.
- Sudo for missing system packages, or use the shared-server option below.

The maintained GPU acceptance tests use RTX 4090 24 GB. For the default model,
12 GB or more is a starting memory estimate; actual use depends on your capture.
Select a GPU marked **VIABLE** by `ftgspp doctor`. Windows, non-NVIDIA GPUs and
multi-GPU training are unsupported. Compute-only data-center GPUs such as
A100/H100 are outside the documented supported range.

## Recommended installation

```bash
git clone https://github.com/sevendogs81-cyber/VK-FTGSPP.git
cd VK-FTGSPP
bash scripts/setup.sh --with-init --with-remote
source .venv/bin/activate
ftgspp doctor --devices 0
```

Setup creates a Python 3.12 environment. This command includes image preparation
and browser viewing. Use the NVIDIA Vulkan ID shown by doctor; it may differ
from the `nvidia-smi` index. Continue with [your first capture](first-run.md).

## Optional components

| Setup option | Use it for |
|---|---|
| No flags | Training/rendering with prepared inputs |
| `--with-init` | BiRefNet masks and DA3 geometry/camera estimation |
| `--with-birefnet` / `--with-da3` | Just one preparation component |
| `--with-remote` | Browser viewer, including over SSH |
| `--with-viewer` | Linux desktop viewer with X11/XWayland |
| `--with-lpips` | Optional perceptual metric for final-test evaluation |

Add a component by rerunning setup with its flag. Earlier selections are kept.
Stop jobs using `.venv` before updating it. Older installations without a saved
profile need all desired flags specified once; `--reset-profile` replaces the
selection. Use `bash scripts/setup.sh --help` for additional options.

## Shared servers and offline use

Without sudo, have a C++17 compiler, Vulkan headers/loader and FFmpeg with
`ffprobe` and `libx264` installed, then add `--skip-apt` to setup.

Download packages and populate model caches before working offline. For local
DA3 weights, set `init.da3.model_path` in your YAML. Optional preparation uses
Torch/CUDA separately from Vulkan training. On GPUs unsupported by the pinned
Torch build, use CPU preparation (`--device cpu`, `init.da3.device: cpu`).

## Check the installation

For a small execution check without data or model downloads, choose a new path:

```bash
python examples/make_tiny_scene.py --output /tmp/ftgspp-tiny
ftgspp train --config /tmp/ftgspp-tiny/ftgspp.yaml --devices 0
ftgspp render --run /tmp/ftgspp-tiny/run --camera 0 --time 0 \
  --device 0 --output /tmp/ftgspp-tiny/render
```

Success creates `run/gaussians.npz` and `render/000000.png`. This synthetic
example checks execution; use your own capture to judge reconstruction quality.

## Common problems

| Problem | Action |
|---|---|
| Missing system packages / no sudo | Install the requirements above, then use `--skip-apt` |
| Existing `.venv` uses another Python version | Use a fresh checkout; setup requires Python 3.12 |
| Wrong GPU or `llvmpipe` selected | Use the NVIDIA Vulkan ID printed by doctor |
| No `VIABLE` GPU | Check the NVIDIA driver; include doctor output when reporting a problem |
| Missing optional package | Add its setup flag and reactivate `.venv` |
