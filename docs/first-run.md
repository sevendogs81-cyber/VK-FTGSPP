# First run: foreground and scene input routes

The first three routes reconstruct a masked foreground subject. For mask-free
full-frame reconstruction, use the [scene-mode routes](#scene-mode-input-routes).

All routes below use synchronized images in
`data/my_capture/images/<camera_id>/<numeric-frame>.png` (JPEG is also supported).
Use the actual capture FPS. The camera rig is static over the clip; unrelated
photos or an arbitrary moving-camera monocular video are not interchangeable
with a synchronized multi-view capture. Camera IDs must match your files.

| Available inputs | Preparation | Install |
|---|---|---|
| RGB frames only | Masks → DA3 geometry/camera estimation → carving + image coloring | `bash scripts/setup.sh --with-init` |
| XYZ PLYs without RGB | Optional point filtering + image coloring, or explicit neutral-gray conversion; provide matching calibration and masks | Core if calibration/masks exist; `--with-birefnet` if masks need generation |
| Calibration, masks and RGB PLYs | Index the dataset, then train using existing points | `bash scripts/setup.sh` |

Use `ftgspp doctor` to choose a Vulkan device for training. CUDA preparation
ordinals can differ from Vulkan device IDs. Preparation defaults to CUDA when
available; `--device cpu` selects its Torch-free geometry/coloring implementation
when no model inference is needed.

The generated YAML inherits `train.preset: balanced`. Before training, inspect
the frame-adaptive horizon without creating a run:

```bash
ftgspp train --config data/my_capture/ftgspp.yaml --print-config
```

Use `--set train.preset=preview` for a shorter diagnostic run or
`--set train.preset=extended` for more refinement. These presets keep the
2-million generated Gaussian budget (or the actual user-cloud population).
See the [argument and tuning guide](arguments.md) for memory adjustments,
per-keyframe point caps and long-sequence limitations.

## 1. RGB frames only

```bash
bash scripts/setup.sh --with-init
source .venv/bin/activate
ftgspp prepare dataset --images data/my_capture/images --fps 30 --pipeline foreground
ftgspp prepare masks --images data/my_capture/images \
  --view-manifest data/my_capture/manifests/dataset.json \
  --output data/my_capture/masks/birefnet_hr
ftgspp prepare points --method carving --config data/my_capture/ftgspp.yaml \
  --output data/my_capture/points_vhull_ibr_stride1
ftgspp train --config data/my_capture/ftgspp.yaml --devices 0
```

The dataset command indexes images and prints commands with your actual paths;
it does not run all preparation stages itself. Each completed point preparation
also prints a training command selecting its output directory, method and budget.
Use that command when `--output` differs from the YAML path.

Mask and DA3 weights are downloaded
on first model use unless cached. With no calibration files, the point preparation
stage estimates the static camera rig once at the first keyframe using DA3, saves its camera cache, and then
carves/colors the foreground. Training reloads that verified camera cache, so keep
the generated DA3 cache and its `current.json` alongside the prepared PLYs.
Estimated calibration is not guaranteed to match measured calibration accuracy.

If calibration already exists, add `--calibration /path/to/calibration` to dataset
preparation; the directory must contain the supported `intri.yml`/`extri.yml` pair.
For existing masks, pass `--mask-root /path/to/masks` or `--mask-manifest FILE` and
skip mask inference. For RGBA images whose alpha contains the mask, use
`--mask-source alpha`. Inspect masks before point preparation.

With calibrated enclosing cameras, the default carving path derives per-frame
bounds from masks and can skip DA3 entirely. Difficult masks or poorly constrained
camera layouts fall back to DA3. Preparation uses `--quality balanced` by default:
color at most `4 × ceil(total Gaussians / selected keyframes)` candidates per frame,
while all carved geometry still participates in occlusion. The final Gaussian
budget remains 2M. Choose `--quality fast` for a first diagnostic, `high` for more
candidate headroom, or `full` for uncapped comparison; see
[preparation parameters](arguments.md#max_points-per-keyframe-versus-the-total-model).
Mask maps are reused within a frame and RGB decoding overlaps coloring. The
output `manifest.json` records the actual bounds, counts and stage timings.

Reserve a camera for checkpoint selection with `--eval-cameras ID` if appropriate.
Without a hold-out camera, training exports the last model. Plain `train` with the
default carving backend expects prepared PLYs; run the printed preparation steps
first. Alternatively `--set init.backend=da3` trains from DA3 points directly after
masks are ready, without carving.

## 2. Initial XYZ point clouds without colors

Provide a directory containing `f000000.ply`, `f000001.ply`, etc. Each file must
contain `x`, `y`, `z`; normals alone do not provide color. Points and camera
calibration must use the same coordinate system. Camera estimation from unrelated
DA3 geometry cannot automatically align an arbitrary user point cloud. If your
cloud has no matching calibration, obtain that calibration/alignment first, or
use route 1 to regenerate both together.

Index the data and identify the supplied points:

```bash
ftgspp prepare dataset --images data/my_capture/images --fps 30 --pipeline foreground \
  --calibration data/my_capture/cameras --mask-root data/my_capture/masks \
  --points data/my_capture/points_xyz
```

Choose **one** preparation command. The recommended `point_filtering` route
filters the existing points against foreground masks, then applies visibility-aware
IBR image colors. It removes inconsistent points and points without trusted color
sources, so the point count can decrease. It does not create a carved voxel grid:

```bash
ftgspp prepare points --method point_filtering --config data/my_capture/ftgspp.yaml \
  --input-points data/my_capture/points_xyz --output data/my_capture/points_colored
```

To skip image coloring, explicitly initialize all colors to neutral gray:

```bash
ftgspp prepare points --method colorize --color-mode gray \
  --config data/my_capture/ftgspp.yaml \
  --input-points data/my_capture/points_xyz --output data/my_capture/points_colored
```

Use a new/empty output directory, different from the source. If you want image
colors while preserving all points, use `--method colorize` with its default
color mode instead of `point_filtering`. By default `colorize`
preserves point coordinates, order and count. Points with no trusted image source
remain gray and their count is recorded in the manifest. `--max-points` explicitly
allows subsampling. Gray initialization can train but may converge more slowly
or reach lower quality than reliable image colors.

The command prints the matching training command:

```bash
ftgspp train --config data/my_capture/ftgspp.yaml --devices 0 \
  --set init.backend=points --set init.points_path=data/my_capture/points_colored \
  --set init.num_gaussians=null
```

CLI override paths are relative to the current working directory. This selects
the actual prepared point count instead of padding to 2 million.

If you also want to rebuild geometry, `--method carving --input-points DIR` accepts
XYZ-only PLYs as a source of bounds and generates a new colored surface. This
changes geometry and uses the generated-geometry budget. `--method point_filtering`
instead removes silhouette-inconsistent input points and colors the remainder.
Only `colorize` is the geometry-preserving color option.

## 3. All files already exist

```bash
bash scripts/setup.sh
source .venv/bin/activate
ftgspp prepare dataset --images data/my_capture/images --fps 30 --pipeline foreground \
  --calibration data/my_capture/cameras --mask-root data/my_capture/masks \
  --points data/my_capture/points_rgb
ftgspp train --config data/my_capture/ftgspp.yaml --devices 0
```

This route invokes neither BiRefNet nor DA3. It reuses existing RGB PLYs and uses
their total selected point count. `--mask-root` expects the matching camera/frame
mask layout; use a manifest for arbitrary names. Pass `--keyframe-stride 10` if
your PLY set is spaced every ten input frames; the last frame is always required.
A single static PLY is not automatically copied across a dynamic sequence.

## Scene-mode input routes

Full-scene RGB reconstruction uses `pipeline.mode: scene`, requires no foreground
masks and uses its own adaptive training/CC/HS recipe. The synchronized static
rig and coordinate-alignment requirements still apply. These are supported
pinhole/distortion camera inputs, not an equirectangular 360-degree-image loader.

| Available scene inputs | Route |
|---|---|
| Raw synchronized RGB, optional calibration | Index → DA3 initialization and optional camera estimation → train → HS |
| XYZ PLYs plus matching calibration | Index with `--points` → optional image coloring or gray conversion → train → HS |
| RGB PLYs plus matching calibration | Index with `--points` → train → HS; no model inference |

Raw RGB route (replace the held-out camera ID with an actual camera in your rig):

```bash
bash scripts/setup.sh --with-da3
source .venv/bin/activate
ftgspp prepare dataset --images data/my_capture/images --fps 30 \
  --pipeline scene --eval-cameras 3
ftgspp train --config data/my_capture/ftgspp.yaml --print-config
ftgspp train --config data/my_capture/ftgspp.yaml --devices 0
```

Default scene stride follows FPS: at 30 FPS it is 5 (short clips clamp to their
length). DA3 `quality: auto` sizes its dense cache using the resulting keyframe
count. Unlike the foreground carving route, scene `train` can initialize from
raw RGB directly when DA3 and its weights are available. BiRefNet is not needed.

For supplied XYZ/RGB PLYs, specify their actual spacing, including a PLY for the
last frame. For example, files for every indexed frame use stride 1:

```bash
ftgspp prepare dataset --images data/my_capture/images --fps 30 \
  --pipeline scene --calibration data/my_capture/cameras \
  --points data/my_capture/points_xyz --keyframe-stride 1 --eval-cameras 3
ftgspp prepare points --method colorize --config data/my_capture/ftgspp.yaml \
  --input-points data/my_capture/points_xyz --output data/my_capture/points_colored
ftgspp train --config data/my_capture/ftgspp.yaml --devices 0 \
  --set init.backend=points --set init.points_path=data/my_capture/points_colored \
  --set init.num_gaussians=null
```

Add `--color-mode gray` to skip image coloring. Scene coloring uses the full
valid image; it does not require foreground masks. Output preserves XYZ and
point count unless an explicit `--max-points` is requested. If the input PLYs
already contain RGB, skip color conversion and train directly from the generated
YAML (use `--points data/my_capture/points_rgb` during indexing).

Generated geometry defaults to 2M total Gaussians. Explicit user `points` uses
the selected PLY population. In both cases `train.preset` selects the training
budget, with independent numeric overrides. Balanced scene training is 30k main
plus 3k HS updates at 60 frames, or 150k main plus 15k HS at 300 frames. See the
[exact scene formulas and parameter guide](arguments.md#scene-adaptation).

If a valid YAML already exists, skip dataset preparation and train directly.
Dataset preparation refuses to replace an existing starter unless `--overwrite`
is explicitly provided. Managed point caches retain their manifest/hash checks;
keep the complete cache, not only individual PLY files. Invalid or partial caches
fail explicitly instead of silently mixing files.

These routes have preparation/initialization integration tests. The raw-RGB test
uses model prediction fixtures while exercising real mask files, estimated-camera
cache persistence, carving, coloring and training initialization. It is not a
reconstruction-quality benchmark or a claim that every new capture will produce
accurate estimated cameras.
