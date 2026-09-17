# FTGSPP Vulkan Community

A Linux/NVIDIA, Python 3.12 implementation of explicit-velocity 4D Gaussian
training. One view per device produces one optimizer update. Single-device and
two-device synchronous training are supported; local gradient accumulation is
not part of the interface.

**License:** non-commercial research and educational use only, not an
OSI-approved or commercially permissive open-source license. Read [LICENSE](LICENSE)
and [third-party notices](THIRD_PARTY.md) before use or redistribution.

## Install

Release artifacts are not published automatically. Install a locally built wheel:

```bash
python3.12 -m pip install dist/ftgspp_vulkan_community-*.whl
ftgspp doctor --devices 0
```

To build from source, install a C++17 toolchain, Vulkan development headers/loader,
CMake 3.24+, and Slang 2026.10 (`slangc` on PATH), then:

```bash
uv build
python3.12 -m pip install dist/ftgspp_vulkan_community-*.whl
```

The build compiles C++ and every shader; neither requires a GPU at build time.
Wheels include their native extension and shader resources. Runtime requires a
compatible NVIDIA Vulkan driver; CUDA ordinals and Vulkan device indices can differ.
There is no source-tree or Torch dependency for ordinary training.

## Use

Bring an existing rectangular multi-view image grid without copying it:

```bash
# images/<camera>/<numeric-frame>.(png|jpg|...) -> manifest + ftgspp.yaml
ftgspp prepare dataset --images data/my_capture/images --fps 30 \
  --pipeline foreground --mask-source alpha --eval-cameras 7
ftgspp train --config data/my_capture/ftgspp.yaml --devices 0
```

The command writes `manifests/dataset.json` and a starter config. Camera IDs,
frame range and train/eval coverage are validated. For arbitrary filenames or
images spread across directories, author an `ftgspp-dataset-v1` JSON with
`frame`, `camera_id`, `rgb_path` and optional per-view `timestamp`, then run
`ftgspp prepare dataset --manifest views.json`. Timestamps must cover all
selected views or none; optional `rgb_sha256` values are verified. SelfCap
layouts use the dedicated adapter: `ftgspp prepare selfcap --scene-root ...`.

YAML paths are relative to the YAML file; CLI path arguments are relative to
the current directory.

```bash
ftgspp prepare --help
ftgspp train --config examples/scene.yaml --output runs/my_scene --devices 0
ftgspp train --config examples/scene.yaml --output runs/my_pair --devices 0 2
ftgspp train --resume runs/my_scene
ftgspp eval --run runs/my_scene --device 0
ftgspp export --model runs/my_scene/gaussians.npz --output model.ply
```

Every new run selects one public task profile. The image objective and
evaluation protocol are derived from `pipeline.mode`; they are not separate
user settings.

```yaml
pipeline:
  mode: scene       # mask-free full-frame RGB reconstruction
# mode: foreground  # masked target reconstruction
```

`ftgspp train --config ... --pipeline foreground` is the equivalent CLI
override. Foreground mode expects `${data.root}/manifests/masks.json`, uses a
separate foreground DA3 cache, enables foreground-only DA3 filtering, trains
`composite_v1` with random solid backgrounds, and ranks checkpoints by
foreground PSNR. Scene mode uses the mask-free `scene_rgb_v1` full-image
objective. Resume always reuses the saved mode and rejects a new profile
override. Foreground `init.backend: auto` consumes user PLYs only when their
generated manifest proves foreground compatibility. A manifestless or
full-scene PLY directory is never selected implicitly; use
`init.backend: points` to make that decision explicit.

Start foreground reconstruction from
[examples/foreground_scene.yaml](examples/foreground_scene.yaml). Generate a
versioned BiRefNet mask cache (optional extra: `.[birefnet]`) and train:

```bash
ftgspp prepare masks --images data/my_scene/images \
  --output data/my_scene/masks/birefnet_v1
ftgspp train --config examples/foreground_scene.yaml --devices 0
```

`data.masks.source` may also be `alpha`. RGB is always straight/unpremultiplied;
the soft mask is stored in reference alpha only. An optional validity map is a
separate ignore signal. The foreground evaluator reports black/white-background
PSNR/SSIM, foreground PSNR, silhouette IoU, boundary F-score, background
opacity leakage, foreground holes and mask consistency.

Leave DA3 sizing knobs unset. Initialization follows `init.num_gaussians`,
`init.keyframe_stride`, `data.frames`, and the native image size.
Set `init.seed` to pin DA3 geometry and L3 sampling independently of
`train.seed`; omit it to keep the historical fallback to `train.seed`.
Starter examples pin `init.seed: 0` so training-seed sweeps reuse one
geometry cache. `init.da3.quality: auto` (default) derives resolution, fusion
cameras, and the dense-cache cap. Set `quality: fast|balanced|high` or
explicit sizing values only for a frozen recipe. DA3 is never imported by
ordinary training; install the upstream
[Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3)
package and weights separately, then set `init.da3.model_path` for offline use.

Research controls, not the default path:

- [examples/foreground_fixed_black.yaml](examples/foreground_fixed_black.yaml)
  keeps `composite_v1` but fixes the solid background to black.
- [examples/foreground_visual_hull.yaml](examples/foreground_visual_hull.yaml)
  trains from a calibrated visual-hull + IBR PLY set:

```bash
ftgspp prepare points --config examples/foreground_visual_hull.yaml \
  --output data/my_scene/points_vhull_ibr_stride10 \
  --bounds XMIN YMIN ZMIN XMAX YMAX ZMAX --resolution 192 \
  --sampling surface_band
ftgspp train --config examples/foreground_visual_hull.yaml --devices 0
```

A locked final test set can be evaluated without entering checkpoint selection:

```bash
ftgspp prepare final-test --run runs/my_foreground_scene \
  --manifest data/my_scene/manifests/final_test.json
ftgspp eval --run runs/my_foreground_scene --device 0 \
  --test-manifest data/my_scene/manifests/final_test.json
```

See [examples/final_test_manifest.example.json](examples/final_test_manifest.example.json).
New manifests should use `ftgspp-final-test-v2`. Validation rejects any
camera/frame, path, or RGB hash already used by training or checkpoint
selection. Official evaluation is `ftgspp eval`. Scripts under `tools/` are
unsupported research helpers.

Use repeated `--set key=value` for validated overrides. Resume loads the saved
effective plan without repeating learning-rate scaling. Checkpoints are V6 only.
Hold-out eval shares `train.checkpoint_interval` with optional optimizer
snapshots (`train.save_checkpoints`). The exported `gaussians.npz` is the best
hold-out joint PSNR+SSIM snapshot by default (`train.export_best`).
`train.hard_separation` is an advanced optional phase and stays off unless you
also enable `model.marginal_gating`.

The scene needs a complete multi-camera image grid; calibration is optional.
`init.backend: auto` consumes a complete per-keyframe RGB PLY set when present
and otherwise uses pose-conditioned DA3. With no calibration, DA3 estimates a
static rig from the first keyframe. Uncalibrated input cannot use
`init.backend: points`. For a video-only capture:

```bash
ftgspp prepare selfcap --scene-root <scene> --init-backend da3 --write-config scene.yaml
```

Core installation is Torch-free. Optional extras are `[da3]`, `[lpips]`,
`[birefnet]`, `[foreground]`, and `[tensorboard]`. LPIPS is never part of the
training loss. The native model-flow renderer remains diagnostic only.

## Development

```bash
uv sync --extra dev --no-install-project
uv run --no-sync pytest
```

See [architecture](docs/architecture.md), [migration](docs/migration.md), and
[contributing](CONTRIBUTING.md). CPU contract tests and native GPU numerical
tests are engineering-correctness gates, not evidence of reconstruction quality.
