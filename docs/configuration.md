# Configuration

Generate a capture-specific YAML with `ftgspp prepare dataset`, then keep using
that file through preparation and training. Files in `examples/` are templates;
edit their FPS, frames, paths and camera IDs before use. They do not reproduce
the showcase or define quality acceptance thresholds.

## Mode defaults

An omitted setting inherits package defaults and then its mode's profile.
Explicit YAML fields override both. The effective run is saved in
`RUN/config.resolved.yaml`; inspect that file when reproducing or reviewing a run.
Reading `defaults.yaml` alone misses profile overrides.

| Behavior | `scene` | `foreground` |
|---|---|---|
| Training target | Full-frame RGB | Masked subject |
| Masks | Not accepted | Required |
| Default initialization | Existing PLYs, otherwise DA3 (`auto`) | Prepared visual-hull cache (`carving`) |
| Background | Fixed black | Random solid colors |
| Marginal gating | On | On |
| Color correction | On | Off |
| Hard separation after main training | On | Off |
| Hold-out selection | Joint full-frame PSNR + SSIM | Joint foreground PSNR + SSIM |

Foreground defaults do not require enabling HS or color correction to be valid.
Changing these controls is an experiment, not a general fix for poor quality.

## Settings to review first

| Key | Default / meaning | What to check |
|---|---|---|
| `data.fps` | 60 frames/s when unspecified | Use the actual capture rate |
| `data.frames` | `{start: 0, stop: 300}` in generic templates | Start inclusive, stop exclusive; generated YAML uses indexed frames |
| `data.train_cameras`, `data.eval_cameras` | Capture-specific | Explicit disjoint camera IDs; generated YAML records the chosen split |
| `data.scale` | 0.5 | Scales source image dimensions and intrinsics; 2048 becomes 1024 without an undistortion-size change |
| `init.backend`, `init.points_path` | Mode-dependent | Match the prepared artifact and its method |
| `init.num_gaussians` | 2,000,000 total initial Gaussians | Memory/quality budget, not points per frame |
| `init.keyframe_stride` | 10 frames | Initialization spacing; generated configs shorten this for captures under ten frames |
| `init.duration` | `keyframe_stride / fps` seconds | Initial temporal sigma, not total clip length |
| `init.num_velocity_nns` | 0 | Zero initial velocity; a positive value estimates it from neighboring keyframe points; velocity is still optimized |
| `train.iterations` | 30,000 main optimizer updates | Does not include optional HS; not a convergence guarantee |
| `train.batch_size` | 1 | One image per update; foreground can explicitly use 4 |
| `train.checkpoint_interval` | 2,000 updates | Also controls candidate evaluation frequency |
| `eval.frame_interval` | 10 frames | Evaluation sampling stride, not video export FPS |

The base learning rates and remaining settings are in
[defaults.yaml](../src/ftgspp/config/defaults.yaml); mode overrides are in
[compose.py](../src/ftgspp/config/compose.py). Unknown/retired fields are rejected.

Preparation has its own CLI settings. For example, carving's `--resolution`
controls a voxel grid, `--max-points` caps retained initialization candidates,
and `--bounds` uses calibration-world coordinates. None is an alias for
`init.num_gaussians`. The training guide uses resolution 192 and coarse
resolution 48, as in the visual-hull example; these are starting choices,
not dataset-specific optimal settings.

## Paths and overrides

Paths written inside YAML resolve relative to that YAML's directory. Paths
passed through CLI arguments or `--set` resolve relative to the current working
directory. Use absolute paths when moving configs between locations. A bare
`$HOME` or `$JYD` inside YAML is not shell expansion; use an explicit path or
OmegaConf `${oc.env:VARIABLE}` syntax.

```bash
ftgspp train --config /path/to/capture/ftgspp.yaml \
  --output /path/to/runs/trial --devices 0 \
  --set data.scale=0.25 --set init.num_gaussians=500000
```

These overrides start a new experiment with a smaller budget. They do not
establish equivalent quality. Resume uses the saved effective plan and rejects
new run overrides. To compare changes, keep the data split and evaluation
protocol fixed and use separate run directories.

## Hard separation

HS starts from the selected main-phase model and exports its terminal model.
A completed B1 run with `model.marginal_gating: true` can add it later:

```bash
ftgspp train --resume /path/to/run --hard-separation
```

For a new run, configure its schedule before starting:

```yaml
train:
  hard_separation: true
  hard_separation_steps: 3000
  hard_relocation_steps: [1500]  # [] disables relocation within this phase
```

Ordinary selected and terminal HS snapshots are retained in
`ckpt/main_selected.npz` and `ckpt/hard_final.npz`. Numbered checkpoints contain
optimizer resume state. See [phase details](architecture.md#phases-and-checkpoints).

## Batch size

B1 is the default. B4 is supported for single-device foreground with color
correction and HS disabled, `distributed.mode: off`,
`distributed.sampling_policy: flat_epoch`, and no automatic LR scaling.
Four sequential views share one FP32 mean gradient and one Adam update.

Iterations and checkpoint/LR/relocation schedules count optimizer updates;
`views_seen` and SH activation count image exposures. Equal updates at B4 see
four times as many images as B1. Dividing update-based schedules by four matches
exposure counts, not convergence. B4 also requires extra memory; see
[architecture](architecture.md#batch-size-four) and [benchmarks](performance.md#local-batch-execution-cost).

## Optional DA3 settings

Install DA3 using the locked setup route. Use `init.da3.model_path` for local
weights, or `init.da3.model_name` (default `depth-anything/DA3-BASE`) for a model
that may download on first use. `init.da3.device` uses Torch/CUDA device naming.

Leave `quality: auto`, resolution, fusion-camera count and point limits unset
initially. They are derived from capture size and Gaussian budget. Explicit
`fast`, `balanced` or `high` quality presets are available for controlled
experiments; see the [initialization architecture](architecture.md#initialization-decision-tree).
