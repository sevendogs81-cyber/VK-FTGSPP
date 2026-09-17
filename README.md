# FTGSPP Vulkan Community

A Linux/NVIDIA, Python 3.12 implementation of explicit-velocity 4D Gaussian
training. The installed interface is the `ftgspp` CLI plus the YAML keys in
this README and [docs/architecture.md](docs/architecture.md). It is not a
library you `import`.

**First release trains on one GPU.** One view produces one optimizer update.
Two-device synchronous training is **in development**: the pair code,
`tests/gpu/` scripts, and the controlled GPU workflow stay in the tree for
engineering, but they are not a supported user path. Local gradient
accumulation is not part of the interface.

**License:** non-commercial research and educational use only, not an
OSI-approved or commercially permissive open-source license. Read [LICENSE](LICENSE)
and [third-party notices](THIRD_PARTY.md) before use or redistribution.

## Requirements

- Linux x86_64 and an NVIDIA driver that exposes Vulkan compute
- **Python 3.12.x** (3.11 and 3.13 will fail the native build)
- C++17 compiler (`g++`) and Vulkan headers/loader (`libvulkan-dev`)
- Slang **2026.10** with `slangc` on `PATH`
- [uv](https://docs.astral.sh/uv/) (CI uses 0.11.29)

Building the wheel compiles C++ and every shader. That step does **not** need
a GPU. `ftgspp doctor`, `train`, and `eval` do. CUDA ordinals and Vulkan
device indices can differ; `--devices` is the Vulkan id from `doctor`, not
`nvidia-smi`.

The core wheel is Torch-free. Optional extras add Torch only when you ask.

## Install

Release wheels are not published to PyPI. Clone this repository, build a
wheel, then install it. YAML examples live in the **clone**; the wheel only
provides the `ftgspp` command.

```bash
git clone git@github.com:sevendogs81-cyber/FastGS.git
cd FastGS

sudo apt-get update
sudo apt-get install -y build-essential libvulkan-dev

# Slang 2026.10 (same tarball and checksum as CI)
curl -fsSL https://github.com/shader-slang/slang/releases/download/v2026.10/slang-2026.10-linux-x86_64-glibc-2.27.tar.gz \
  -o /tmp/slang.tgz
echo "beddadd9b48151aa2d7b556056ea02b9826bde59d4d4fa827b33c5184e42a983  /tmp/slang.tgz" | sha256sum -c -
mkdir -p "$HOME/.local/slang-2026.10"
tar -xzf /tmp/slang.tgz -C "$HOME/.local/slang-2026.10"
export PATH="$HOME/.local/slang-2026.10/bin:$PATH"
slangc -v   # expect 2026.10

# uv 0.11.29; skip if you already have it
curl -LsSf https://astral.sh/uv/0.11.29/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv build --python 3.12
uv pip install dist/ftgspp_vulkan_community-*.whl
ftgspp doctor --devices 0
```

Keep this venv activated for every later `ftgspp` and extra-install command
in this README (`source .venv/bin/activate` in a new shell). `.venv` is
gitignored.

`uv build` needs CMake 3.24+; scikit-build-core can download that CMake if the
system copy is older. GLM is fetched during the build when it is not installed.

The JSON `name` from `doctor` should be an NVIDIA GPU, not `llvmpipe`. Use
that object's `device_id` as `--devices`. Doctor opens a real device; it fails
on a CPU-only build machine even when the wheel compiled.

If `slangc: not found`, Python/CMake cannot find 3.12, or Vulkan headers are
missing, stop here and fix the step above. Do not pass `--devices` with two
ids.

## First training run (scene, one GPU)

Work from the clone. Bring an existing rectangular multi-view grid; pixels
are indexed in place, not copied:

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
`scene_rgb_v1`). Paths inside a YAML file are relative to that file; CLI
paths are relative to the current directory. Without `--output`, the run
directory is `runs/<name>` beside the YAML.

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
uv pip install --find-links dist --no-index \
  'ftgspp-vulkan-community[birefnet]'
ftgspp prepare masks --images data/my_scene/images \
  --output data/my_scene/masks/birefnet_v1
```

That command writes `data/my_scene/manifests/masks.json` when `--output`
is `<scene>/masks/<name>`. Then train from
[examples/foreground_scene.yaml](examples/foreground_scene.yaml) or from the
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
uv pip install --find-links dist --no-index \
  'ftgspp-vulkan-community[da3]'
```

DA3 runs on CUDA (`init.da3.device`). That ordinal can differ from the
Vulkan `--devices` id used for training. Foreground + `auto` initialization
needs the mask extra **and** this DA3 stack.

## Research controls (not the default path)

- [examples/foreground_fixed_black.yaml](examples/foreground_fixed_black.yaml)
  keeps `composite_v1` but fixes the solid background to black.
- [examples/foreground_visual_hull.yaml](examples/foreground_visual_hull.yaml)
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

See [examples/final_test_manifest.example.json](examples/final_test_manifest.example.json).
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

## Training speed (SelfCap)

Head-to-head on the six SelfCap clips (`dance1`, `dance2`, `corgi1`, `corgi2`,
`bike1`, `bike2`): 60 frames, 500k Gaussians, 30k optimizer steps, one training
view per step, NVIDIA RTX 4090. **Every rate below is train-only.** Hold-out
eval is stripped out of both implementations.

| Label in the tables | Implementation | What was trained |
|---|---|---|
| **FTGSPP original** | CUDA + gsplat  | scene RGB |
| **Ours** | Vulkan + vksplat | scene RGB and foreground |

### How train-only speed is measured

**FTGSPP original.** The loop prints tqdm `n/30000 [elapsed, it/s]` and runs a
60-view PSNR eval every 2000 steps. Rates are the median of those 2k-step
windows *between* evals, after dropping the first window (CUDA compile /
warmup) and any window below 30 it/s. MCMC relocation every 100 steps stays in
the number; eval does not.

**Ours.** `performance_summary.json` records `step_seconds` around the optimizer
update only (`_evaluate` is outside the timer). `steps_per_second` is
`30000 / sum(step_seconds)` (mean). `step_seconds_p50` / `p95` are the median /
95th-percentile step. MCMC relocate is also outside the timer; 244 relocations
do not change the speedups below.

### Scene RGB — train-only throughput

`it/s` is optimizer steps per second (higher is better). `ms/step` is
`1000 / it/s`. Ours columns use the mean step rate; `p50` is the typical step
with less log-sync noise.

| Scene | FTGSPP original it/s | ms/step | Ours it/s | ms/step | p50 it/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| dance1 | 44.1 | 22.7 | 138.6 | 7.2 | 151.3 | **3.14×** |
| dance2 | 61.2 | 16.3 | 143.7 | 7.0 | 153.2 | **2.35×** |
| corgi1 | 43.1 | 23.2 | 131.6 | 7.6 | 144.5 | **3.05×** |
| corgi2 | 45.1 | 22.2 | 69.2 | 14.5 | 74.5 | **1.54×** |
| bike1 | 47.4 | 21.1 | 97.7 | 10.2 | 109.6 | **2.06×** |
| bike2 | 54.1 | 18.5 | 106.0 | 9.4 | 118.3 | **1.96×** |
| **Unweighted avg** | **49.2** | **20.3** | **114.5** | **8.7** | **125.2** | **2.33×** |

### Scene RGB — time for 30k train steps, eval excluded

Wall time implied by the train-only rates above (`30000 / it/s`). Lower is
better.

| Scene | FTGSPP original | Ours | Speedup |
|---|---:|---:|---:|
| dance1 | 11.3 min (680 s) | 3.6 min (216 s) | 3.14× |
| dance2 | 8.2 min (490 s) | 3.5 min (209 s) | 2.35× |
| corgi1 | 11.6 min (697 s) | 3.8 min (228 s) | 3.05× |
| corgi2 | 11.1 min (666 s) | 7.2 min (434 s) | 1.54× |
| bike1 | 10.5 min (633 s) | 5.1 min (307 s) | 2.06× |
| bike2 | 9.3 min (555 s) | 4.7 min (283 s) | 1.96× |
| **Avg** | **10.3 min (620 s)** | **4.7 min (279 s)** | **2.22×** |

Per-step detail for Ours (scene RGB), eval excluded:

| Scene | mean / p50 / p95 (ms) |
|---|---|
| dance1 | 7.21 / 6.61 / 9.44 |
| dance2 | 6.96 / 6.53 / 9.46 |
| corgi1 | 7.60 / 6.92 / 11.03 |
| corgi2 | 14.45 / 13.43 / 23.64 |
| bike1 | 10.23 / 9.12 / 18.55 |
| bike2 | 9.43 / 8.45 / 16.79 |

### Foreground protocol (Ours)

FTGSPP original `runs_main_da3` is scene RGB, so there is no original-foreground
column. Same 30k / 500k / 60-frame recipe; rates are again
`30000 / sum(step_seconds)`.

| Scene | Ours it/s | ms/step | 30k train |
|---|---:|---:|---:|
| dance1 | 147.1 | 6.8 | 3.4 min (204 s) |
| dance2 | 130.0 | 7.7 | 3.8 min (231 s) |
| corgi1 | 101.0 | 9.9 | 5.0 min (297 s) |
| corgi2 | 100.3 | 10.0 | 5.0 min (299 s) |
| bike1 | 177.5 | 5.6 | 2.8 min (169 s) |
| bike2 | 301.8 | 3.3 | 1.7 min (99 s) |
| **Avg** | **159.6** | **6.3** | **3.6 min (216 s)** |

Foreground is faster on the square `bike*` clips (smaller raster) and slower
on `corgi*` than Ours scene RGB. Across six scenes it still beats FTGSPP
original scene RGB (~49 it/s) by about **3.2×**.

## Two-device training (in development)

Passing two `--devices` values is not a supported first-release workflow.
The CLI still accepts it so engineering scripts keep working, and prints a
warning. Pair transport, `tests/gpu/check_pair_equivalence.py`,
`tests/gpu/smoke_sync_multi_gpu.py`, and `.github/workflows/gpu.yml` remain
in the repository for that work.

## Development

```bash
uv sync --python 3.12 --extra dev --no-install-project
uv run --no-sync pytest
```

Use this only when changing code. It rewrites `.venv` from the lockfile and
does **not** install the wheel; CPU tests skip `_ftgspp_vk` when it is
missing. Keep the Install venv for running `ftgspp`. Shader or C++ changes
need `slangc` on `PATH` and `uv build --python 3.12`. See
[architecture](docs/architecture.md),
[migration](docs/migration.md), and [maintainer notes](CONTRIBUTING.md). CPU
contract tests and native GPU numerical tests are engineering-correctness
gates, not evidence of reconstruction quality.
