# VK-FTGSPP (FastGS)

![VK-FTGSPP DNA-Rendering dynamic scene showcase](docs/assets/vk-ftgspp_dna_teaser.gif)

Train and view dynamic 3D scenes from multi-view image sequences with a native
Vulkan 4D Gaussian renderer. Reconstruct the full scene, or use masks to
reconstruct a foreground subject.

- **Platform:** Linux x86_64, a supported NVIDIA GPU, and one GPU per run.
- **Inputs:** synchronized multi-view images; existing camera calibration and
  point clouds, or optional Depth Anything 3 (DA3) initialization. Foreground
  reconstruction also needs masks, supplied by you or generated with BiRefNet.
- **Outputs:** a native dynamic model, evaluation results, an optional browser
  or desktop viewer, and PNG/MP4 rendering.
- **License:** non-commercial research and educational use only. This is not
  an OSI-approved or commercially permissive license. See [LICENSE](LICENSE)
  and [third-party notices](THIRD_PARTY.md).

## Install

You need Linux x86_64 with glibc 2.28+, an installed NVIDIA driver with Vulkan
support, and network access. The installer provides Python 3.12, uv, Slang and
build dependencies; missing system packages require apt-get and root/sudo.
Check [GPU compatibility](docs/hardware.md) before installing.

Choose your optional components **before running setup**. The following installs
the lightweight, Torch-free core and the SSH browser viewer:

```bash
git clone https://github.com/sevendogs81-cyber/VK-FTGSPP.git
cd VK-FTGSPP
bash scripts/setup.sh --with-remote
source .venv/bin/activate
ftgspp doctor --devices 0
```

Use a NVIDIA device reported as `VIABLE`. Vulkan IDs may differ from
`nvidia-smi`; replace `0` in subsequent commands with the selected Vulkan ID.
The browser computer does not need a GPU; rendering runs on the server.

| Your initialization inputs | Setup flags, including browser access |
|---|---|
| Calibration and prepared point clouds; masks if using foreground mode | `--with-remote` |
| Masks are ready; generate points or estimate carving bounds with DA3 | `--with-da3 --with-remote` |
| Generate both foreground masks and geometry from RGB | `--with-init --with-remote` |

**Setup reruns replace the selected optional components.** Repeat every flag
you want to keep; running only `--with-remote` after `--with-init` removes the
initialization extras. Stop jobs using `.venv` before changing it.
See [installation](docs/install.md) for desktop, offline, manual and no-sudo options.

## Check the complete workflow

This small generated fixture requires no data or model download. It checks that
training and rendering work; it is **not a reconstruction-quality benchmark**.
Use a new output directory:

```bash
python examples/make_tiny_scene.py --output /tmp/ftgspp-tiny
ftgspp train --config /tmp/ftgspp-tiny/ftgspp.yaml --devices 0
ftgspp eval --run /tmp/ftgspp-tiny/run --device 0
ftgspp render --run /tmp/ftgspp-tiny/run --camera 0 --time 0 \
  --device 0 --output /tmp/ftgspp-tiny/render
```

Success produces `run/gaussians.npz`, `run/config.resolved.yaml`,
`run/viewer_manifest.json`, and `render/000000.png`. The image is deliberately
a tiny synthetic colored target; a successful exit does not establish real-scene
quality. The fixture uses 32×32 images, two frames and 64 Gaussians.

To inspect it interactively, on the GPU server run:

```bash
ftgspp viewer --remote --run /tmp/ftgspp-tiny/run --device 0
```

On your computer, keep this tunnel open and visit the exact tokenized URL
printed by the viewer:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

On the same machine, open the printed URL directly. Only one browser client
can connect at a time. See [viewer controls and export](docs/viewer.md).

## Train your own capture

Follow [the training guide](docs/usage.md) from start to finish. Choose
**scene** for full-frame RGB, or **foreground** for a masked subject. The guide
keeps the same generated config through preparation, training and viewing.

| Task | Guide |
|---|---|
| Arrange images, masks, calibration and timestamps | [Data requirements](docs/data.md) |
| Understand defaults and change training settings | [Configuration](docs/configuration.md) |
| Resume, evaluate and inspect output quality | [Usage and troubleshooting](docs/usage.md) |
| View, capture or export a trained model | [Viewer](docs/viewer.md) |
| Understand measured speed and its limits | [Performance](docs/performance.md) |
| Develop or rebuild native code | [Maintainer notes](CONTRIBUTING.md) |

The YAML files in `examples/` are starter templates, not recipes that reproduce
the showcase. They require your capture's FPS, frame range and camera split.
A downloadable real-scene quality reproduction bundle is not included in this
release; the showcase and timing tables are not acceptance targets for arbitrary
captures. See [how to review reconstruction quality](docs/usage.md#check-results).

Historical SelfCap timings report a 2.22× aggregate train-only speedup over the
original CUDA + gsplat implementation, with different relocation accounting.
See [definitions and results](docs/performance.md) before comparing performance.

Source code lives in `src/ftgspp/` and `native/`; user examples in `examples/`;
engineering tests and unsupported tools in `dev/`. Two-device training remains
an engineering path outside the supported user workflow.
