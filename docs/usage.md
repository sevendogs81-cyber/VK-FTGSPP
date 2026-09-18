# Usage

Work from the clone with the Install venv activated
(`source .venv/bin/activate`). Paths inside a YAML file are relative to that
file; CLI paths are relative to the current directory. Without `--output`, the
run directory is `runs/<name>` beside the YAML.

## First training run (scene, one GPU)

Bring an existing rectangular multi-view grid; pixels are indexed in place,
not copied:

```text
data/my_capture/images/<camera_id>/<numeric-frame>.(png|jpg|...)
```

A flat `c<camera>-f<six-digit-frame>.*` directory is also accepted. Camera
calibration is optional: an EasyMocap `intri.yml` / `extri.yml` pair at the
scene root is used when present.

`--eval-cameras` must be camera ids that already exist in the grid (the
directory names under `images/`). Replace `3` below with your hold-out
camera, or omit the flag to train on every camera with no hold-out.

```bash
ftgspp prepare dataset --images data/my_capture/images --fps 30 --eval-cameras 3
```

That writes `data/my_capture/manifests/dataset.json` and
`data/my_capture/ftgspp.yaml`, then prints the next `train` command. The
starter YAML pins `init.seed: 0` so later `train.seed` sweeps reuse one
geometry cache. Default `pipeline.mode` is `scene` (mask-free
`scene_rgb_v1`).

**Initialize before the first `train`.** `init.backend: auto` looks for a
complete keyframe RGB PLY set at
`${data.root}/points_stride${init.keyframe_stride}/` (starter stride 10 →
`data/my_capture/points_stride10/`). Files are `f<six-digit-frame>.ply` for
each keyframe (`f000000.ply`, `f000010.ply`, …) and the last selected frame.
If that set is incomplete, it loads Depth Anything 3. Put PLYs elsewhere with
`init.points_path`. Ordinary training does not import DA3; without those PLYs
you must follow [Optional DA3](#optional-da3) first, or training stops at
initialization.

```bash
ftgspp train --config data/my_capture/ftgspp.yaml --devices 0
```

`examples/scene.yaml` is the same field template if you prefer to copy it
into the clone rather than use the generated file.

For arbitrary filenames, author an `ftgspp-dataset-v1` JSON with `frame`,
`camera_id`, `rgb_path`, and optional per-view `timestamp`, then run
`ftgspp prepare dataset --manifest views.json`. Timestamps must cover all
selected views or none; optional `rgb_sha256` values are verified. SelfCap
layouts use `ftgspp prepare selfcap --scene-root ...`.

Repeated `--set key=value` applies validated overrides. Resume loads the
saved effective plan and rejects `--output`, `--devices`, `--pipeline`, and
`--set`:

```bash
ftgspp train --resume data/my_capture/runs/my_capture
ftgspp eval --run data/my_capture/runs/my_capture --device 0
```

A finished run already contains `gaussians.npz` (best hold-out joint
PSNR+SSIM snapshot unless `train.export_best` is false) and
`gaussians_4dgs.ply`. `ftgspp export` only writes another PLY path:

```bash
ftgspp export --model data/my_capture/runs/my_capture/gaussians.npz \
  --output model.ply
```

Checkpoints are V6 only. Hold-out eval shares `train.checkpoint_interval`
with optional optimizer snapshots (`train.save_checkpoints`).
`train.hard_separation` stays off unless you also enable
`model.marginal_gating`.

Leave DA3 sizing knobs unset. Initialization follows `init.num_gaussians`,
`init.keyframe_stride`, `data.frames`, and the native image size.
`init.da3.quality: auto` (default) derives resolution, fusion cameras, and
the dense-cache cap. Set `quality: fast|balanced|high` or explicit sizing
only for a frozen recipe.

## Foreground reconstruction

Do this after the scene path works. `pipeline.mode: foreground` derives
`composite_v1`, requires masks, uses a separate foreground DA3 cache, enables
foreground-only DA3 filtering, trains with random solid backgrounds, and
ranks checkpoints by foreground PSNR.
`ftgspp train --config ... --pipeline foreground` is the equivalent CLI
override. Resume always reuses the saved mode.

Masks are **not** always `manifests/masks.json`:

- `data.masks.source: alpha` — RGBA images; the soft mask is the alpha
  channel (RGB stays straight/unpremultiplied).
- `data.masks.source: external` — a mask file per view. If you do not set
  `data.masks.manifest_path` / `data.masks.root`, the profile fills
  `${data.root}/manifests/masks.json`.

```bash
ftgspp prepare dataset --images data/my_scene/images --fps 30 \
  --pipeline foreground --mask-source alpha --eval-cameras 3
```

To generate a versioned BiRefNet cache (needs the `birefnet` extra, same
venv as Install):

```bash
uv pip install '.[birefnet]'
ftgspp prepare masks --images data/my_scene/images \
  --output data/my_scene/masks/birefnet_v1
```

That command writes `data/my_scene/manifests/masks.json` when `--output`
is `<scene>/masks/<name>`. Then train from
[examples/foreground_scene.yaml](../examples/foreground_scene.yaml) or from the
generated YAML.

Foreground `init.backend: auto` still needs a provenance-checked foreground
PLY set or DA3. A manifestless or full-scene PLY directory is never selected
implicitly; use `init.backend: points` to make that decision explicit.

The foreground evaluator reports black/white-background PSNR/SSIM, foreground
PSNR, silhouette IoU, boundary F-score, background opacity leakage, foreground
holes and mask consistency. An optional validity map is a separate ignore
signal.

## Optional DA3

`[da3]` installs Torch/TorchVision only. It does **not** install Depth
Anything 3. Install the upstream
[Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3)
package and weights in the same environment, then set
`init.da3.model_path` for offline use. With `model_path` unset, the default
`init.da3.model_name` (`depth-anything/DA3-BASE`) may try a network download.

```bash
uv pip install '.[da3]'
```

DA3 runs on CUDA (`init.da3.device`). That ordinal can differ from the
Vulkan `--devices` id used for training. Foreground + `auto` initialization
needs the mask extra **and** this DA3 stack.

## Research controls (not the default path)

- [examples/foreground_fixed_black.yaml](../examples/foreground_fixed_black.yaml)
  keeps `composite_v1` but fixes the solid background to black.
- [examples/foreground_visual_hull.yaml](../examples/foreground_visual_hull.yaml)
  trains from a calibrated visual-hull + IBR PLY set (requires calibration;
  uncalibrated input cannot use `init.backend: points`):

```bash
ftgspp prepare points --config examples/foreground_visual_hull.yaml \
  --output data/my_scene/points_vhull_ibr_stride10 \
  --bounds XMIN YMIN ZMIN XMAX YMAX ZMAX --resolution 192 \
  --sampling surface_band
ftgspp train --config examples/foreground_visual_hull.yaml --devices 0
```

A locked final test set is evaluated without entering checkpoint selection.
`ftgspp-final-test-v2` requires a completed run with `export_selection.json`:

```bash
ftgspp prepare final-test --run data/my_scene/runs/my_foreground_scene \
  --manifest data/my_scene/manifests/final_test.json
ftgspp eval --run data/my_scene/runs/my_foreground_scene --device 0 \
  --test-manifest data/my_scene/manifests/final_test.json
```

See [examples/final_test_manifest.example.json](../examples/final_test_manifest.example.json).
Validation rejects any camera/frame, path, or RGB hash already used by
training or checkpoint selection. Official evaluation is `ftgspp eval`.
Scripts under `tools/` are unsupported research helpers.

Optional extras: `[da3]`, `[lpips]`, `[birefnet]`, `[foreground]` (BiRefNet +
LPIPS family together), `[tensorboard]`. LPIPS is never part of the training
loss and is off unless `eval.lpips` is enabled. The native model-flow
renderer remains diagnostic only.

SelfCap video layouts:

```bash
ftgspp prepare selfcap --scene-root <scene> --init-backend da3 --write-config scene.yaml
```

## Two-device training (in development)

Passing two `--devices` values is not a supported first-release workflow.
The CLI still accepts it so engineering scripts keep working, and prints a
warning. Pair transport, `tests/gpu/check_pair_equivalence.py`,
`tests/gpu/smoke_sync_multi_gpu.py`, and `.github/workflows/gpu.yml` remain
in the repository for that work.
