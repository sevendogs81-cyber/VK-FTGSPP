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
Camera splits and output destinations are checked before either file is written.
`--overwrite` permits replacing generated outputs, but never an input image or
the source manifest. The YAML and manifest must have distinct paths; each file
is published atomically after its contents have been written successfully.
The starter YAML pins `init.seed: 0` so later `train.seed` sweeps reuse one
geometry cache. Default `pipeline.mode` is `scene` (mask-free
`scene_rgb_v1`).

**Choose an initialization source before the first `train`.** With camera
calibration, `init.backend: auto` looks for a complete keyframe RGB PLY set at
`${data.root}/points_stride${init.keyframe_stride}/` (starter stride 10 →
`data/my_capture/points_stride10/`). Files are `f<six-digit-frame>.ply` for
each keyframe (`f000000.ply`, `f000010.ply`, …) and the last selected frame.
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
The default HS schedule is 3000 optimizer steps with one group-local relocation
at local step 1500. It can be adjusted without changing the main schedule:

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
By default it evaluates every held-out view selected by `eval.frame_interval`
and appends to `full_validation_metrics.jsonl`, including foreground diagnostics.
For the smaller checkpoint-selection subset, explicitly pass `--quick`; its
results append to `metrics.jsonl` and are marked `candidate_subset`. This follows
`eval.candidate_max_views` and omits the full foreground diagnostic suite.
`--quick` cannot be combined with a locked final-test manifest, including one
saved in the run config. Final-test results remain in `final_test_metrics.jsonl`.
`ftgspp export` only writes another PLY path:

```bash
ftgspp export --model data/my_capture/runs/my_capture/gaussians.npz \
  --output model.ply
```

PLY export publishes the completed file atomically. A failed write leaves an
existing destination unchanged and removes its temporary file.

Default checkpoints remain V6; local batch four uses V7. Resume preserves
the saved batch size and algorithm identity and rejects an
incompatible configuration. Hold-out eval shares `train.checkpoint_interval`
with optional optimizer snapshots (`train.save_checkpoints`).
The `scene` profile enables color correction, marginal gating,
gate-included relocation scoring, and hard separation by default. The
`foreground` profile enables marginal gating and gate-included relocation
scoring while leaving color correction and hard separation off. Explicit YAML
settings override these profile defaults. When HS is enabled, it binarizes
persistent/transient gates, preserves and refines velocity independently,
restricts relocation to the same gate group, and exports the terminal hard-phase
model.

Leave DA3 sizing knobs unset. The total experiment budget remains
`init.num_gaussians`; initialization dynamically resolves DA3 work from that
budget, `init.keyframe_stride`, `data.frames`, native image size, camera count,
and foreground-mask coverage. It then reallocates the total across keyframes
from candidate count, spatial extent, shape complexity, and inter-frame motion.
`init.da3.quality: auto` (default) derives resolution, fusion cameras, and
the dense-cache cap. Set `quality: fast|balanced|high` or explicit sizing
only for a frozen recipe.

## Foreground reconstruction

For a masked subject capture, `pipeline.mode: foreground` derives
`composite_v1`, requires masks, defaults to calibrated visual-hull carving,
trains with random solid backgrounds, and ranks checkpoints by the joint
foreground PSNR and foreground SSIM score over valid mask-interior windows.
`ftgspp train --config ... --pipeline foreground` is the equivalent CLI
override. Resume always reuses the saved mode.

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
is `<scene>/masks/<name>`. The default geometry path is `carving`. It first
generates or reuses the configured foreground DA3 cloud, derives robust
training-only world bounds from it, and carves the mask-supported surface:

```bash
ftgspp prepare points --method carving \
  --config examples/foreground_scene.yaml \
  --output data/my_scene/points_vhull_ibr_stride10 \
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
  --output data/my_scene/points_vhull_filtered_stride10
```

Train that filtered set with the matching explicit backend:

```bash
ftgspp train --config examples/foreground_scene.yaml --devices 0 \
  --set init.backend=point_filtering \
  --set init.points_path=data/my_scene/points_vhull_filtered_stride10
```

Each generated manifest records the selected method. The `carving` and
`point_filtering` backends accept only a complete, integrity-checked manifest
for that exact method. `--input-points` remains available to reuse an existing
DA3/user PLY directory.

| Method | Geometry source | Practical tradeoff |
|---|---|---|
| `carving` (default) | Mask-supported surface band inside DA3/user-derived bounds | Stronger silhouette control and less dependence on DA3 point placement; voxel projection costs more time and memory |
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
`init.backend: da3` uses its cloud directly; uncalibrated captures must use DA3
because visual-hull projection requires a camera rig.

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
  --output data/my_scene/points_vhull_ibr_stride10 \
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
