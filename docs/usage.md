# Train a capture

First complete [installation](install.md) and the [tiny workflow check](../README.md#check-the-complete-workflow).
Run commands from the clone with `.venv` activated. Review the
[data requirements](data.md) before preparing your own images.

## 1. Choose a mode and initialization route

| Goal | Mode | Inputs for the route below |
|---|---|---|
| Reconstruct the complete RGB image | `scene` | Images plus calibration and existing points, or DA3 |
| Reconstruct only a masked subject | `foreground` | Calibrated images, masks, and carving initialization |

For uncalibrated foreground captures, use the [direct DA3 route](#other-initialization-routes).
The modes have different losses, backgrounds and defaults; see
[configuration](configuration.md#mode-defaults). Foreground mode is not an installation extra.

## 2. Index the images

Set these values for your capture. `FPS=30` and camera `3` are examples, not
inferred metadata. `CAPTURE` must contain your images; `RUN` must be a new output
path. Keep these variables for all subsequent commands in this guide.

```bash
CAPTURE=/absolute/path/to/my_capture
CONFIG="$CAPTURE/ftgspp.yaml"
RUN=/absolute/path/to/runs/my_capture
FPS=30
EVAL_CAMERA=3
DEVICE=0  # Vulkan ID verified with ftgspp doctor
```

Choose **one** indexing command. It writes a dataset manifest and `CONFIG`;
it does not copy the images or complete initialization.

For a full scene:

```bash
ftgspp prepare dataset --images "$CAPTURE/images" --fps "$FPS" \
  --eval-cameras "$EVAL_CAMERA" --config "$CONFIG" --pipeline scene
```

For foreground with existing external masks:

```bash
ftgspp prepare dataset --images "$CAPTURE/images" --fps "$FPS" \
  --eval-cameras "$EVAL_CAMERA" --config "$CONFIG" \
  --pipeline foreground --mask-root "$CAPTURE/fmasks"
```

For foreground with RGBA images, replace `--mask-root ...` with
`--mask-source alpha`. For RGB that needs new masks, omit both flags and
then run:

```bash
ftgspp prepare masks --images "$CAPTURE/images" \
  --output "$CAPTURE/masks/birefnet_hr"
```

Mask generation requires BiRefNet (`--with-init` or `--with-birefnet` at setup).
Skip it when using existing masks or alpha, even if the indexing command prints
a generic mask-generation suggestion. Do not switch to an `examples/*.yaml`
after indexing: that would discard your generated FPS, frame range and split.

## 3. Prepare initialization

**Scene:** with calibration, `init.backend: auto` consumes a complete keyframe
RGB PLY set in `points_stride10/`; otherwise it uses DA3. For an uncalibrated
scene, DA3 estimates cameras too. Install `--with-da3` before training when
needed. Check [point-cloud naming](data.md#existing-point-clouds).

**Foreground:** inspect `CONFIG` and use its exact `init.points_path` as
`POINTS` below. For at least ten frames, the generated path is normally
`points_vhull_ibr_stride10`; shorter captures use a smaller stride.

```bash
POINTS="$CAPTURE/points_vhull_ibr_stride10"
ftgspp prepare points --method carving --config "$CONFIG" \
  --output "$POINTS" --resolution 192 --coarse-resolution 48
```

This command uses the configured masks and training cameras. By default it uses
DA3 to estimate bounds, so the DA3 extra and weights are required. If you know
the subject's world-space bounds, append `--bounds xmin ymin zmin xmax ymax zmax`
with six numeric values instead; this skips DA3 bound estimation. Explicit
bounds plus `--device cpu` allow preparation in the Torch-free core environment.
[Bounds must match your calibration](data.md#camera-calibration).

Before training, inspect the masks and generated PLYs for coverage, alignment and
color. A successful preparation command alone does not establish correct geometry.

## 4. Train and resume

Review FPS, frame range, train/eval cameras, image scale and initialization path
in `CONFIG`. The [configuration guide](configuration.md) explains inherited
settings and starter budgets. Then run:

```bash
ftgspp train --config "$CONFIG" --output "$RUN" --devices "$DEVICE"
```

On success the run contains the selected model, resolved config and viewer
metadata. Defaults use 30,000 main optimizer updates; scene mode additionally
enables a 3,000-step hard-separation phase. Foreground leaves that phase off.
Actual time and memory depend on your scene and hardware.

Resume an interrupted run from its saved plan:

```bash
ftgspp train --resume "$RUN"
```

Resume rejects new `--config`, `--output`, `--devices`, `--pipeline` and `--set`
overrides. Editing the original YAML does not alter a resumed run. Use a fresh
output directory for a changed experiment. See [optional hard separation](configuration.md#hard-separation)
for adding that phase to a completed compatible run.

## Check results

```bash
ftgspp eval --run "$RUN" --device "$DEVICE"
ftgspp render --run "$RUN" --camera 0 --time 0 \
  --device "$DEVICE" --output "$RUN/preview"
ftgspp viewer --remote --run "$RUN" --device "$DEVICE"
```

Choose a scene time within the recorded range; `0` suits a zero-based capture.
`--camera` is the viewer's camera-list index, unlike the dataset camera IDs used
for splitting. The render output directory must be new. Follow the
[SSH viewer instructions](viewer.md#ssh-browser-viewer-no-server-display) when
viewing from another computer.

| Run artifact | Purpose |
|---|---|
| `config.resolved.yaml` | Effective settings, including profile defaults and resolved paths |
| `run.log`, `train_metrics.jsonl` | Training progress and optimization diagnostics |
| `gaussians.npz`, `viewer_manifest.json` | Native model and matching camera/time metadata for viewing |
| `export_selection.json` | Which model was exported and how it was selected |
| `metrics.jsonl` | Sampled checkpoint-selection/ordinary evaluation scores |
| `full_validation_metrics.jsonl` | Foreground diagnostic validation written at training completion |
| `ckpt/` | Saved checkpoints for resume and phase handoff |
| `gaussians_4dgs.ply` | Compatibility export; omits dynamic gate state and cannot replace native NPZ viewing |
| `performance_summary.json` | Optimizer timing, not total preparation-to-result wall time |

On this version, ordinary `eval` uses the sampled selection evaluator. Do not
interpret it as an independent final test or assume every frame was evaluated.
Foreground training completion also writes the full diagnostic validation file.
Hold-out cameras used to select a checkpoint are a **validation set**. An
independent final test uses a separately locked manifest below.

Inspect the beginning, middle, end and fast-motion portions from multiple
held-out cameras. Compare GT and reconstruction at identical camera poses and
scene timestamps. Use lossless PNGs for fine detail; viewer motion can use a
reduced resolution and MP4 is lossy. Check masks, geometry and synchronization
before changing iteration counts when you see missing limbs, blur or ghosting.

The tiny fixture checks execution only. The generic templates are not tuned
quality presets, and this release does not bundle a real-scene reproduction
package with quality acceptance thresholds. Do not infer showcase quality from
a passing smoke test or from the train-only speed tables.

## Other initialization routes

Use these only after choosing their inputs. Preserve the same `CONFIG` and
write a new run for each experiment.

- **Existing foreground PLYs:** set `init.backend: points` and `init.points_path`
  in `CONFIG`. The PLYs must share the calibration's world coordinates.
- **Direct DA3, including uncalibrated captures:** set `init.backend: da3` in
  `CONFIG`; masks are still required for foreground. Skip the carving step.
- **Point filtering:** set `init.backend: point_filtering` and a fresh
  `init.points_path` in `CONFIG`, then run the following before training:

```bash
ftgspp prepare points --method point_filtering --config "$CONFIG" \
  --output "$CAPTURE/points_filtered"
```

Set `init.points_path` to that exact output. Filtering uses DA3 points, or
existing PLYs supplied with `--input-points`. Carving builds a mask-supported
surface band; filtering inherits the supplied point cloud's coverage and noise.
Both generated caches validate method, input identity and PLY hashes.

For offline DA3, set `init.da3.model_path` to a local model. The default model is
`depth-anything/DA3-BASE`; leave sizing knobs unset initially. The preparation
`--device` and DA3 device refer to CPU/CUDA, not the training Vulkan ordinal.
See [installation](install.md) and [configuration](configuration.md).

SelfCap video layouts have a separate adapter (FFmpeg/ffprobe required):

```bash
ftgspp prepare selfcap --scene-root /path/to/scene \
  --init-backend da3 --write-config scene.yaml
```

Check `ftgspp prepare selfcap --help` for its layout and frame-selection options.

## Locked final-test evaluation

Create an independent [final-test manifest](../examples/final_test_manifest.example.json)
and validate it against the completed run. It must not reuse training or
validation views, paths or RGB hashes.

```bash
ftgspp prepare final-test --run "$RUN" --manifest /path/to/final_test.json
ftgspp eval --run "$RUN" --device "$DEVICE" \
  --test-manifest /path/to/final_test.json
```

Results go to `final_test_metrics.jsonl` and do not influence model selection.
To add VGG LPIPS, include `--with-lpips` alongside all your other setup flags,
then add `--lpips` to this final-test command. It never runs during training or
ordinary validation. On first use it may download roughly 528 MiB of VGG16
weights. `--lpips-device`, `--lpips-batch-size` and `--lpips-min-size` control
execution; see `ftgspp eval --help`. Scene mode scores full RGB frames;
foreground mode scores a reference-mask ROI on black.

## Troubleshooting

| Symptom | First checks |
|---|---|
| Missing mask or point cache | Confirm `CONFIG` paths, prepare the required artifact, and keep its manifest with it |
| Unexpected DA3 download | `auto` falls back to DA3 when the PLY set is incomplete; check keyframes and the last frame |
| Out of GPU memory | Check device choice and other jobs, then reduce `data.scale` or `init.num_gaussians` in a new run |
| Mask/geometry misalignment | Check camera names, source image dimensions, intrinsics, world units and mask correspondence |
| Blur, ghosting or missing surfaces | Compare GT at matching times; check FPS/synchronization and initialization coverage before tuning |
| Output directory already exists | Resume a training run, or choose a fresh output for a new experiment/render; do not delete a valuable run |
| Viewer unavailable | Check [viewer troubleshooting](viewer.md#troubleshooting), the remote extra, tunnel and printed token |

When reporting a problem, include the command, resolved config, doctor output,
commit and relevant log excerpt, with private paths and viewer tokens removed.
