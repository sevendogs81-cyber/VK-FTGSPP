# VK-FTGSPP (FastGS)

![Dynamic reconstruction example](docs/assets/vk-ftgspp_dna_teaser.gif)

Reconstruct and explore moving subjects and scenes from synchronized multi-view
images. VK-FTGSPP trains dynamic 3D Gaussians with Vulkan and includes a browser
viewer for playback, free-view navigation and video export.

## Install

You need **Linux x86_64 and a supported NVIDIA GPU**. The installer sets up
Python 3.12 and the build tools; missing system packages require sudo.
See [requirements and installation options](docs/install.md#requirements).

```bash
git clone https://github.com/sevendogs81-cyber/VK-FTGSPP.git
cd VK-FTGSPP
bash scripts/setup.sh --with-init --with-remote
source .venv/bin/activate
ftgspp doctor --devices 0
```

This includes image preparation and browser viewing. Use the NVIDIA device ID
marked `VIABLE` by `doctor`; it may differ from the `nvidia-smi` index.
For a smaller install or a shared server, see [installation options](docs/install.md).

## Train your capture

Start with images from synchronized cameras viewing the same scene:

```text
my_capture/
└── images/
    ├── 0/000000.jpg, 000001.jpg, ...
    ├── 1/000000.jpg, 000001.jpg, ...
    └── 2/000000.jpg, 000001.jpg, ...
```

Follow **[Train your first capture](docs/first-run.md)** to prepare the images,
train a model and open the result. The guide starts from RGB images; you can
reuse existing masks, camera calibration or point clouds along the way.

## View the result

On the GPU machine, open the completed run:

```bash
ftgspp viewer --remote --run /path/to/run --device 0
```

Open the link printed in the terminal. If the GPU is on another machine, first
run this on your computer and leave it running:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

Drag to rotate, scroll to zoom, and press Play to see the motion.
See [viewer controls and export](docs/viewer.md) for screenshots and videos.

## Performance

SelfCap · RTX 4090 · reported train-only throughput:

| Scene RGB, six-capture mean | FTGSPP (CUDA) | VK-FTGSPP (Vulkan) | Ratio |
|---|---:|---:|---:|
| Reported training-step rate | 49.2 steps/s | **114.5 steps/s** | **2.33×** |

These reported intervals exclude evaluation and hard separation; the original
includes relocation while VK-FTGSPP excludes it, so this is not an end-to-end
speedup. See the [comparison results](docs/performance.md).

## More

- [Resume, evaluate and troubleshoot](docs/usage.md)
- [Data formats](docs/data.md) · [Common settings](docs/usage.md#change-settings)
- [FTGSPP comparison](docs/performance.md) · [Development](dev/README.md)

The showcase illustrates a real reconstruction; a downloadable reproduction
bundle is not currently included. A small [installation check](docs/install.md#check-the-installation)
is available without downloading a dataset.

**License:** non-commercial research and educational use only. See
[LICENSE](LICENSE) and [third-party notices](THIRD_PARTY.md).
