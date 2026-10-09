# Usage

Work from the clone with the environment created by `scripts/setup.sh` activated
(`source .venv/bin/activate`). Paths inside a YAML file are relative to that
file; CLI paths are relative to the current directory. Without `--output`, the
run directory follows `run.output_path`, whose default is `runs/<name>` beside
the YAML. Paths supplied through `--set` are relative to the current directory.

## First training run (scene, one GPU)

Bring an existing rectangular multi-view grid; pixels are indexed in place,
not copied:

```text
data/my_capture/images/<camera_id>/<numeric-frame>.(png|jpg|...)
```

A flat `c<camera>-f<six-digit-frame>.*` directory is also accepted. Camera
calibration is optional: an EasyMocap `intri.yml` / `extri.yml` pair at the
scene root is used when present.

`--eval-cameras` must be camera IDs that already exist in the grid (the
directory names under `images/`). Replace `3` below with your hold-out
camera, or omit the flag to train on every camera with no hold-out.

```bash
ftgspp prepare dataset --images data/my_capture/images --fps 30 --eval-cameras 3
```

That writes `data/my_capture/manifests/dataset.json` and
`data/my_capture/ftgspp.yaml`, then prints the required initialization step.
The starter YAML pins `init.seed: 0` so later `train.seed` sweeps reuse one
geometry cache. Default `pipeline.mode` is `scene` (mask-free
`scene_rgb_v1`).

**Choose an initialization source before the first `train`.** With camera
calibration, `init.backend: auto` looks for a complete keyframe RGB PLY set at
`${data.root}/points_stride${init.keyframe_stride}/` (automatic spacing is
roughly 1/6 second: 30 FPS → stride 5 → `data/my_capture/points_stride5/`). Files are `f<six-digit-frame>.ply` for
each keyframe (`f000000.ply`, `f000005.ply`, … at stride 5) and the last selected frame.
If that set is incomplete, training invokes Depth Anything 3 (DA3) during
initialization. Uncalibrated captures also require DA3 to estimate cameras.
Put PLYs elsewhere with `init.points_path`. The default lightweight install
expects calibration and a complete PLY set. Run
`bash scripts/setup.sh --with-da3` (or `--with-init`) when DA3 is needed.

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

Hard separation (HS) is optional. When it is enabled in the initial
configuration, ordinary training first selects its best hold-out snapshot and
HS starts from that snapshot. A run originally completed without HS can add it
later without retraining the ordinary phase:

```bash
ftgspp train --resume data/my_capture/runs/my_capture --hard-separation
```

This requires `model.marginal_gating: true` and `train.batch_size: 1` in the
original run. The ordinary selection is retained as `ckpt/main_selected.npz`;
an HS run additionally retains `ckpt/hard_final.npz`. These phase checkpoints
contain the model, its matching color correction and the run identity needed for the phase
handoff. Numbered checkpoints and `latest.json` remain exact Adam resume state.
Scene HS defaults to 10% of the effective main horizon, with one group-local
relocation halfway through HS. At 30k main steps this is 3000 HS steps and
relocation at local step 1500. It can be overridden independently:

```yaml
train:
  hard_separation_steps: 3000
  hard_relocation_steps: [1500]  # [] disables HS relocation
```

HS relocation always scores donors from opacity and accumulated image gradient;
the main phase may still use `train.relocation.score_mode: gate_included`.

A finished run already contains `gaussians.npz` and `gaussians_4dgs.ply`.
For a main-only scene run this is the best hold-out joint full-frame PSNR+SSIM
snapshot. Foreground runs use joint `foreground_psnr`+`foreground_ssim` over
the reference mask interior. This selection is disabled when `train.export_best`
is false; an HS run exports the terminal HS model.
Standalone `ftgspp eval` streams the model SHA-256 check
and loads only this selected model; it does not restore an optimizer checkpoint.
`ftgspp export` only writes another PLY path:

```bash
ftgspp export --model data/my_capture/runs/my_capture/gaussians.npz \
  --output model.ply
```

Default checkpoints remain V6; local batch four uses V7. Resume preserves
the saved batch size and algorithm identity and rejects an
incompatible configuration. Hold-out eval shares `train.checkpoint_interval`
with optional optimizer snapshots (`train.save_checkpoints`).
The `scene` profile enables color correction, marginal gating,
gate-included relocation scoring, and hard separation by default. The
`foreground` profile enables marginal gating and uses opacity/gradient-based
relocation scoring (`default`), with color correction and hard separation off. Explicit YAML
settings override these profile defaults. When HS is enabled, it binarizes
persistent/transient gates, preserves and refines velocity independently,
restricts relocation to the same gate group, and exports the terminal hard-phase
model.

Leave DA3 sizing knobs unset. Generated geometry uses the total experiment
budget `init.num_gaussians` (2 million by default); initialization resolves DA3 work from that
budget, `init.keyframe_stride`, `data.frames`, native image size, camera count,
and foreground-mask coverage. It then reallocates the total across keyframes
from candidate count, spatial extent, shape complexity, and inter-frame motion.
`init.da3.quality: auto` (default) derives resolution, fusion cameras, and
the dense-cache cap. Set `quality: fast|balanced|high` or explicit sizing
only for a frozen recipe.

Point preparation has a separate `--quality fast|balanced|high|full` option.
The default `balanced` caps generated color candidates at
`4 × ceil(init.num_gaussians / selected keyframes)` before IBR aggregation,
retaining all geometry for occlusion. User clouds remain uncapped. Calibrated
foreground carving first tries per-frame mask-frustum bounds, so a surrounding
rig can avoid depth inference. Automatic bounds that cannot be safely resolved
fall back to DA3. Stage timings and the resolved plan are saved in the point
manifest. See [preparation arguments](arguments.md#max_points-per-keyframe-versus-the-total-model).
Scene geometry still uses DA3: foreground silhouette bounds do not reconstruct
background/concave scene structure. Scene DA3 retains its budget/FPS-aware
schedule and now prefetches input images; scene XYZ coloring shares the bounded
IBR decoder and remains geometry-preserving by default.

## Adaptive scene defaults

New scene runs also accept `train.preset: preview|balanced|extended` (balanced
by default). The scene reference uses 30k main steps at 60 frames; the balanced
rule is `ceil_1000(30000 × max(0.5, frames / 60))`. A 300-frame default run now
uses **150k main + 15k HS updates**. Color-correction warm-up, gate-score warm-up,
main relocation and validation clocks follow the main horizon; HS relocation
uses the independent HS clock. Explicit numeric overrides and saved plans keep
their values. `train.iterations: null` restores the automatic main budget for a
new run; remove copied clock overrides to restore those defaults too.

Scene retains its mask-free full-frame objective, half-resolution starter,
fixed background, CC, marginal gating and HS. Its automatic keyframe stride is
`min(frame_count, max(1, round(fps / 6)))`, keeping roughly 1/6-second spacing.
For fast motion choose a smaller explicit stride and regenerate matching points.
For supplied PLYs always specify their actual stride. `prepare dataset --points`
selects the user-cloud route and its actual selected population; generic `auto`
keeps a 2M budget while choosing between existing PLYs and DA3.

```bash
ftgspp train --config data/my_capture/ftgspp.yaml --print-config
ftgspp train --config data/my_capture/ftgspp.yaml --set train.preset=preview
```

See [scene input routes](first-run.md#scene-mode-input-routes), the
[scene preview example](../examples/scene_preview.yaml) and
[full scene formulas / tuning guide](arguments.md#scene-adaptation).
These rules keep approximate temporal training exposure; they do not establish
matching quality across sequence lengths or compensate for fixed model capacity.

## Foreground reconstruction

For a masked subject capture, `pipeline.mode: foreground` derives
`composite_v1`, requires masks, defaults to calibrated visual-hull carving,
trains with random solid backgrounds, and ranks checkpoints by the joint
foreground PSNR and foreground SSIM score over valid mask-interior windows.
`ftgspp train --config ... --pipeline foreground` is the equivalent CLI
override. Resume always reuses the saved mode.

### Defaults for a first foreground run

Selecting `pipeline.mode: foreground` applies the following starting recipe.
`prepare dataset --pipeline foreground` and the foreground examples use the
same defaults. Choose `train.preset: preview|balanced|extended` for the automatic
training budget. Explicit step counts win; `train.iterations: null` restores
automatic sizing for a new foreground run. Use `train --print-config` to inspect
the concrete settings before training. See [arguments.md](arguments.md) for exact
formulas, limits, and the distinction between per-keyframe point caps and the
total Gaussian budget. It targets quality on a 24 GB GPU; image size, projected overlap
and initialization can still change memory use substantially.

| Setting | Default | Reason |
|---|---|---|
| `data.scale` | `1.0` | Preserve input detail and mask boundaries |
| `init.keyframe_stride` | `1` | Initialize each captured frame rather than interpolate sparse geometry |
| `init.num_gaussians` | `2000000` for generated geometry | User-supplied points default to their selected population; see below |
| `init.seed`, `init.num_velocity_nns` | `0`, `3` | Repeatable geometry with local velocity initialization |
| `init.duration` | `null` → `1 / data.fps` | Derive temporal support from the actual capture rate |
| `train.preset`, `train.batch_size` | `balanced`, `1` | 70k updates at 121 frames; frame-adaptive thereafter |
| `train.iterations` | Automatic unless explicitly set | `ceil_1000(70000 × max(0.5, frames / 121))`; 174k at 300 frames |
| Relocation start / stop / interval | Proportional to the resolved horizon | At 70k: `2000` / `35000` / `500`; second half for refinement |
| `train.checkpoint_interval` | About 1/35 of the horizon | Bound hold-out evaluation and optimizer snapshot count |
| `train.relocation.score_mode` | `default` | Opacity/gradient scoring used by DNA v3 and 4DAnyone v8 |
| Duration / marginal-gate learning rates | `0.001` / `0.0005` | Gentler temporal updates than the earlier foreground defaults |
| Position / velocity final LR ratios | `0.03` / `0.03` | Ratios over the configured training length, not absolute learning rates |
| `eval.frame_interval`, `eval.candidate_max_views` | `1`, `128` | Denser temporal validation with bounded candidate scoring |

Random solid training backgrounds, marginal gating, SH degree 3, and
`train.export_best: true` remain enabled. Color correction and hard separation
remain off. Reserve at least one calibrated camera with `--eval-cameras` for
checkpoint selection; without a hold-out, the last model is exported.

This is a conservative starting recipe informed by existing experiments, not
a measured optimum for arbitrary captures. DNA v3's ten scenes selected
checkpoints between steps 58,000 and 69,999. In 4DAnyone v8, the 200k terminal
models improved boundary F-score on 10 of 12 scenes, but all 12 had lower
foreground SSIM than their selected checkpoints. Long refinement therefore
has a boundary/detail tradeoff rather than a universal quality benefit.
Those studies used scene-specific budgets and filtered point clouds; their
scores do not establish the quality of a new capture's carved initialization.

Keep the general recipe's trainable gate (`model.gate_init_logit: -1.0`). The
4DAnyone experiments also explored a softer gate (`-0.23`) and a gate learning
rate of `0.0001` or zero. These are explicit alternatives to test against a
hold-out for uncertain masks; freezing gates has not been validated as a
universal replacement on DNA. Do not copy v8's tiny final LR ratios into a
shorter run: they were adjusted for its 200,000-step schedule.

Every-frame preparation and full-resolution training cost more than the old
stride-10, half-resolution starter. For a smaller GPU, begin with
`data.scale: 0.5` and `init.num_gaussians: 1000000`, then increase them after a
successful run. Increasing `init.keyframe_stride` trades temporal initialization
detail for preparation time; regenerate matching point caches after changing
it. Keep `data.fps` equal to the actual indexed frame rate. Optimizer snapshots
can consume substantial disk space; `train.save_checkpoints: false` disables
periodic exact-resume snapshots while preserving selected-model export.

Explicit YAML values always win. Existing `config.resolved.yaml` files and
resumes retain their original settings; old editable YAML files retain any
values they explicitly specify.

### Bring your own initial point clouds

For step-by-step commands covering raw RGB, XYZ-only PLYs and complete inputs,
see [First foreground run](first-run.md). `prepare dataset --points DIR` writes
a starter that consumes your own geometry. For XYZ-only files, the recommended
`prepare points --method point_filtering` route filters foreground points and
assigns image colors. Use `--method colorize` to retain all geometry, optionally
adding `--color-mode gray` to skip image coloring. All produce RGB PLYs suitable
for the initializer.

For an existing calibrated RGB PLY set, select `init.backend: points` and set
`init.points_path` to its directory. Omit `init.num_gaussians` to use the sum
of the selected keyframe point counts as the foreground Gaussian budget:

```yaml
pipeline:
  mode: foreground
data:
  root: /path/to/capture
  fps: 30
  frames: {start: 0, stop: 121}
init:
  backend: points
  points_path: /path/to/my_points
  keyframe_stride: 1
  num_gaussians: null  # also clears a 2M value copied from another example
```

Supply `f000000.ply`, `f000001.ply`, … for this example. Each PLY needs
`x/y/z` and `red/green/blue` vertex properties. Calibration and foreground
masks are still required. Only keyframes in the configured frame interval
count, including its last frame even when it is not aligned with the stride.
A single static PLY is not automatically repeated across a dynamic sequence.
Incomplete or empty selected clouds fail explicitly.

The count is resolved to an integer before the run plan is saved; loading an
existing run does not recount the source files. With this exact source
population, initialization keeps all source points without filling up to 2M
with replicas. Set a positive numeric `init.num_gaussians` to intentionally
subsample or expand that total instead. `carving`, `da3`, and `point_filtering`
keep the 2M default because they prepare new geometry; to use a prepared set's
exact population, consume it through `backend: points`. Generated manifests
are still validated on that path.

### Batch size

Single-device foreground training supports `train.batch_size: 4` (default: 1).
These modes are called B4 and B1, respectively.
Four sequential views are rendered at the same parameters, their complete
gradients (including regularization) are averaged on the GPU in FP32, and Adam
updates once. This requires `distributed.mode: off`,
`distributed.sampling_policy: flat_epoch`,
`distributed.world_size_lr_scaling: none`, `train.color_correction: false`, and
`train.hard_separation: false`.
Learning rates, Adam betas and epsilon are not automatically scaled.

`train.iterations`, LR decay, relocation, logging and checkpoint intervals count
**optimizer updates**. `views_seen` and SH activation count image exposures.
For equal exposure to a 200,000-update B1 run, use 50,000 updates at B4 and divide
update-based interval/boundary settings by four, retaining the LR endpoint ratios.
This matches the exposure schedule, not the number of Adam updates or convergence.
The flat-epoch view order and random backgrounds match the corresponding B1
exposure slots. Overflow replays all four views before committing any update.
Only a complete batch may be checkpointed. Backward accumulates directly into
the existing Adam gradient buffers, with no separate mean buffer or copy-back.
The fourth view's parameter backward, final averaging and Adam share the B1
fused shader at commit. Non-logging batches queue the first three views and
validate completion and overflow for the whole batch before updating. Logged
batches retain per-view loss readback. This requires no additional setting.
For `N` Gaussians, expect approximately `260 * N` extra bytes over fused B1 for
FP32 gradients, plus padding and relocation backup buffers. The B1 fused update
is unchanged.

### Masks and geometry preparation

The mask location depends on `data.masks.source`:

- `data.masks.source: alpha` — RGBA images; the soft mask is the alpha
  channel (RGB stays straight/unpremultiplied).
- `data.masks.source: external` — a mask file per view. If you do not set
  `data.masks.manifest_path` / `data.masks.root`, the profile fills
  `${data.root}/manifests/masks.json`.

```bash
ftgspp prepare dataset --images data/my_scene/images --fps 30 \
  --pipeline foreground --mask-source alpha --eval-cameras 3
```

BiRefNet-HR is an optional preparation component. For the complete raw-RGB
foreground workflow, install it together with DA3 before generating masks:

```bash
bash scripts/setup.sh --with-init
ftgspp prepare masks --images data/my_scene/images \
  --output data/my_scene/masks/birefnet_hr
```

Use `--with-birefnet` alone when a point-cloud set or explicit carving bounds
already exists and only mask generation is needed.

The pipeline pins `ZhengPeng7/BiRefNet_HR` model code and weights to an audited
revision and uses its 2048-square inference size. CUDA uses float16 weights and
autocast and starts with a batch of two; an
out-of-memory batch is retried at a smaller size. Use `--precision float32` for
a numerical control, `--batch-size 1` for a fixed low-memory run, or select a
specific model revision with `--model-revision`. The resolved model,
revision, resolution, precision, effective batch sizes and throughput are
recorded in the manifest. Image decode/hash is prefetched on CPU while the
current batch runs; restored masks, optional raw arrays and QA overlays are
encoded by bounded output workers. Shape-compatible masks are restored as a
batch, and each input/output file is hashed from the bytes already being read
or written.

That command writes `data/my_scene/manifests/masks.json` when `--output`
is `<scene>/masks/<name>`. The default geometry path is `carving`. With a
calibrated enclosing rig, it derives per-frame bounds from training-view masks
and carves the surface without a DA3 point cloud. Otherwise it estimates the
missing rig once and/or falls back to full-sequence DA3 bounds:

```bash
ftgspp prepare points --method carving \
  --config examples/foreground_scene.yaml \
  --output data/my_scene/points_vhull_ibr_stride1 \
  --resolution 192 --coarse-resolution 48
ftgspp train --config examples/foreground_scene.yaml --devices 0
```

Carving uses CUDA automatically when Torch and a CUDA device are available,
otherwise it uses NumPy on CPU. Projection is chunked and a CUDA OOM retries
with a smaller chunk. With `--coarse-resolution`, the fine pass enumerates only
cells covered by the dilated coarse hull. CUDA computes support and keep votes
on-device and transfers only retained indices and diagnostics. The same
`--device` setting covers point filtering and IBR coloring. In `auto` mode,
all implemented projection, sparse z-buffer, robust color, unprojection,
consistency, and voxel-fusion accelerators use CUDA whenever it is available.
`--device cpu` forces the deterministic CPU path. An explicit `--bounds ...`
skips DA3-based bound estimation; `--input-points` derives bounds from an
existing DA3/user PLY set.

The second method directly filters the generated/reused DA3 PLYs before IBR
recoloring:

```bash
ftgspp prepare points --method point_filtering \
  --config examples/foreground_scene.yaml \
  --output data/my_scene/points_vhull_filtered_stride1
```

Train that filtered set with the matching explicit backend:

```bash
ftgspp train --config examples/foreground_scene.yaml --devices 0 \
  --set init.backend=point_filtering \
  --set init.points_path=data/my_scene/points_vhull_filtered_stride1
```

Each generated manifest records the selected method. The `carving` and
`point_filtering` backends accept only a complete, integrity-checked manifest
for that exact method. `--input-points` remains available to reuse an existing
DA3/user PLY directory.

| Method | Geometry source | Practical tradeoff |
|---|---|---|
| `carving` (default) | Mask-supported surface band inside per-frame mask bounds or explicit/DA3/user-derived bounds | Stronger silhouette control and less dependence on DA3 point placement; voxel projection costs more time and memory |
| `point_filtering` | DA3/user surface points retained by multi-view masks | Usually faster because it skips the voxel grid; it inherits missing surfaces, depth noise, and sampling bias from the source cloud |

Both methods produce surface candidates before the shared Gaussian initializer.
Carving's voxel volume is an intermediate occupancy representation; the default
`surface_band` output does not fill the object interior. Shared IBR coloring
projects and aggregates points in vectorized arrays, and its robust fusion is
chunked to bound temporary memory.

### Evaluation

The foreground evaluator reports black/white-background PSNR/SSIM, foreground
PSNR and foreground SSIM, silhouette IoU, boundary F-score, background opacity
leakage, foreground holes and mask consistency. An optional validity map
separately marks pixels to exclude from loss and evaluation.

## DA3 point generation

`bash scripts/setup.sh --with-da3` and the `[da3]` extra install the pinned
Depth Anything 3 0.1.1 inference stack. `--with-init` installs it together
with BiRefNet-HR. The lock replaces the upstream package's broad
demo/server/training metadata with the direct inference dependencies used here;
the runtime has been tested with this project's NumPy 2/OpenCV 4.12 stack. The upstream
[Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3)
weights can be selected with `init.da3.model_path` for offline use. With
`model_path` unset, the default
`init.da3.model_name` (`depth-anything/DA3-BASE`) may try a network download.

```bash
uv sync --locked --no-dev --no-editable --extra da3
```

DA3 uses CUDA when available and otherwise falls back to CPU
(`init.da3.device`). Its unprojection, cross-view depth consistency, and voxel
winner selection follow the resolved model device; masks, cache I/O, and
control flow remain on CPU. The manifest records the postprocessing device and
the L2 cache identity includes the CUDA-capable implementation version. That
ordinal can differ from the Vulkan `--devices` ID used for training.
`init.backend: da3` uses its cloud directly. On uncalibrated captures,
`prepare points --method carving` can first estimate/cache a rig with DA3 and
then carve it; training reloads that camera cache. Explicit user points or bounds
require matching calibration because DA3 cannot infer their coordinate alignment.

```bash
ftgspp train --config examples/foreground_scene.yaml --devices 0 \
  --set init.backend=da3
```

## Research controls

- [examples/foreground_fixed_black.yaml](../examples/foreground_fixed_black.yaml)
  keeps `composite_v1` but fixes the solid background to black.
- [examples/foreground_visual_hull.yaml](../examples/foreground_visual_hull.yaml)
  spells out the default calibrated visual-hull + IBR path.
- [examples/foreground_point_filtering.yaml](../examples/foreground_point_filtering.yaml)
  exposes the DA3-cloud filtering route with `init.backend: point_filtering`.

For the visual-hull example:

```bash
ftgspp prepare points --method carving \
  --config examples/foreground_visual_hull.yaml \
  --output data/my_scene/points_vhull_ibr_stride1 \
  --resolution 192 --sampling surface_band
ftgspp train --config examples/foreground_visual_hull.yaml --devices 0
```

## Locked final-test evaluation

A locked final test set is evaluated without entering checkpoint selection.
Create an independent `ftgspp-final-test-v2` manifest using the
[template](../examples/final_test_manifest.example.json), then validate it
against a completed run with `export_selection.json`:

```bash
ftgspp prepare final-test --run runs/my_foreground_scene \
  --manifest data/my_scene/manifests/final_test.json
ftgspp eval --run runs/my_foreground_scene --device 0 \
  --test-manifest data/my_scene/manifests/final_test.json
```

LPIPS is an optional metric on this locked final-test command only. It never
loads during training, checkpoint selection, or the training-end full
validation. Install the `[lpips]` extra, then add `--lpips`:

```bash
bash scripts/setup.sh --with-lpips
ftgspp eval --run runs/my_foreground_scene --device 0 \
  --test-manifest data/my_scene/manifests/final_test.json --lpips
```

Use `bash scripts/setup.sh --with-init --with-lpips` when the same environment
must retain both raw-RGB initialization components as well.

For scene mode it scores each complete RGB frame. For foreground mode it
scores a black-composited ROI determined only by the reference mask. Equal
image shapes are batched and consumed while rendering, so memory does not grow
with the number of test views. `--lpips-device`, `--lpips-batch-size`, and
`--lpips-min-size` control the optional worker. The default `auto` device
matches the Vulkan GPU to CUDA when possible, reduces the batch after CUDA OOM,
and can fall back to CPU. Results and per-view skip reasons are written only to
`final_test_metrics.jsonl`.

Validation rejects any camera/frame, path, or RGB hash already used by
training or checkpoint selection. Official evaluation is `ftgspp eval`.
Scripts under `dev/tools/` are unsupported research helpers.

See [installation options](install.md) for optional components. The native
model-flow renderer remains diagnostic only.

## SelfCap video preparation

The default setup installs `ffmpeg` and `ffprobe` for video extraction. If you
used `--skip-apt` or a manual installation, both commands must be on `PATH`.

```bash
ftgspp prepare selfcap --scene-root /path/to/scene \
  --init-backend da3 --write-config scene.yaml
```

## Two-device training (in development)

Passing two `--devices` values is not a supported first-release workflow.
The CLI still accepts it so engineering scripts keep working, and prints a
warning. Pair transport, `dev/tests/gpu/check_pair_equivalence.py`,
`dev/tests/gpu/smoke_sync_multi_gpu.py`, and `.github/workflows/gpu.yml` remain
in the repository for that work.
