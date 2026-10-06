# Architecture

The installed CLI resolves user YAML once, validates an immutable effective plan,
loads a shared Scene/Camera/View manifest, initializes or resumes explicit
Gaussians, trains, then exports. This is a fixed sequence, not a configurable
stage list. Evaluation and PLY export consume explicit model/state outputs.
The first community release documents **one-device** training. Two-device pair
transport remains in the tree (native collective, `dev/tests/gpu/`, GPU workflow)
but is in development and not a supported user path.

- `src/ftgspp/config`: schema, path rules, immutable ResolvedRunPlan and validation.
- `src/ftgspp/data`: preparation, calibration, timestamps, lazy pixel loading.
- `src/ftgspp/training`: sampling, initialization, single-device optimizer
  updates, two phase schedules, evaluation and bounded image prefetch. Pair
  updates exist but are in development.
- `src/ftgspp/io`: explicit NPZ models, V6 and local-batch V7 checkpoints, PLY.
- `src/ftgspp/viewer`: viewer asset validation, sessions, camera paths,
  headless export and the loopback SSH browser server.
- `src/ftgspp/optional`: BiRefNet-HR masks, visual-hull/IBR preparation,
  final-test-only LPIPS, and lazy DA3 integration. CLI imports remain lazy.
- `native/src/device`: buffers, compute/transfer submission, timing and telemetry.
- `native/src/render`: projection/rasterization, adaptive dispatch selection and
  the reusable offscreen viewer renderer.
- `native/src/viewer`: optional GLFW / Dear ImGui presentation and controls.
- `native/src/training`: backward, Adam, relocation and hard-separation kernels.
- `native/src/collective`: session-scoped pair transport and generation state.
- `native/src/bindings`: NumPy/Python boundary; `common` holds native configuration.
- `native/slang` and `native/shaders/radix_sort`: shader source only.
- `dev/tests/reference`: numerical reference implementations, not installed runtime.
- `scripts/setup.sh`: one-command user install. Pins match the artifact workflow.
- `dev/tools`: unsupported research helpers for dataset baking, TensorBoard and
  monitoring. Official evaluation is `ftgspp eval`.

## Initialization decision tree

`Scene.from_config` scans the complete image grid before any model is allocated
and checks for a complete EasyMocap calibration pair. With calibration, the
existing camera/world geometry is retained; `init.backend=auto` first consumes a
complete `f<frame>.ply` set under `init.points_path` (default
`<data.root>/points_stride<keyframe_stride>/`, e.g. `points_stride10/f000000.ply`)
and otherwise invokes pose-conditioned DA3. With no calibration, the loader
creates provisional pinhole views, and `auto`/`da3` invokes DA3 on the first
keyframe to estimate one static rig. Large rigs are
processed in bounded chunks with shared anchor cameras; later keyframes are
conditioned on that rig. Auto DA3 fuses every camera on small rigs and a
geometrically spread subset on dense arrays, requires two-view depth support
when a second camera exists, and applies the sky mask only for monocular
fusion. DA3 depth is unprojected into world points and its
processed RGB initializes the same SH-DC coefficient used by PLY input. Each
keyframe's views are voxel-fused in world space before any point cap; the fused
dense cloud is cached as the 3-NN velocity reference, and Gaussian means are
spatially sampled from it to the resolved per-keyframe budgets. The shared
initializer allocates the total `init.num_gaussians` adaptively across keyframes.

Explicit Gaussian conversion (algorithm `ftgspp-init-l3-v4`) time-anchors means so
`x(t_ref)` recovers the source point, isolates unique-source KNN scales from
replica points, and records L3 identity separately from the DA3 L2 cache.
The initializer has one implementation path: sequential camera chunks,
cross-view depth consistency, bounded-probe adaptive surface-grid sampling, and
exact adaptive Gaussian budgeting. L3 allocates final arrays once, while PLY
loading validates content hashes during the same read. The
estimated rig and generated PLYs are cached together, so a subsequent run can
resume without re-estimation or consume a user PLY set in the cached coordinate
system. Per-point attribute sidecars are not written because L3 consumes only
XYZ/RGB. New DA3 cache directories use a readable configuration prefix
(`scene-init0-kf60-v8-r504-<12hex>`) followed by a collision-resistant short
identity. `init.seed` can pin this namespace independently of `train.seed`.
`init.backend=points` remains an explicit, calibrated/user-cloud path.

The foreground profile defaults to `init.backend=carving`. Both public geometry
preparations can reuse or generate the foreground DA3 cache. Explicit
`--input-points` supplies existing geometry; carving also accepts `--bounds`.
Without explicit bounds, carving derives robust world bounds from the input
cloud, then creates a mask-supported surface-band grid. Its coarse pass expands
to sparse fine-grid indices, so projection work is
limited to surviving regions; the CUDA path keeps vote arrays on-device and
returns only selected indices. Point filtering retains DA3 samples supported by
the silhouettes. Their training selectors are `init.backend=carving` and
`init.backend=point_filtering`. Training verifies the selected method,
dataset identity, keyframe set, and every PLY hash. Source identity remains
audit metadata and emits a warning when it differs from the current runtime.
`init.backend=points` remains the explicit generic/user-cloud path, while
`init.backend=da3` consumes the foreground-aware DA3 cloud without either
visual-hull preparation.

## Optimizer updates

### Batch size one

A single GPU with batch size one (B1) uses the fused backward/Adam path.
Duplicate backward/Adam calls are rejected; a forward loss is consumed exactly
once. Overflow rollback restores relocation statistics and explicitly replays
the same generation and views.

### Batch size four

Ordinary single-device foreground training supports batch size four (B4). See
[configuration requirements](usage.md#batch-size).

A compile-time specialization of the shared parameter backward accumulates
the first three complete gradients with weight 0.25 directly into the existing
FP32 Adam input buffers. At commit, a specialization of the
shared B1 fused backward/Adam shader computes the fourth parameter gradient,
finishes the mean in registers and applies ordinary Adam once. The first view
overwrites stale gradients and clears only the final SH block's padding; later
views use separate FP32 multiply and add operations. No intermediate gradient,
weighted-add dispatch, or mean copy-back is needed.
Parameter and moment buffers stay unchanged until all views succeed; tile
overflow restores pre-batch relocation statistics and replays the same
views/backgrounds. The image objective and six loss-buffer fields are unchanged.

B1 and B4 share the same `NativeRunner` training loop, data prefetch, loss,
evaluation, relocation and checkpoint machinery. Local batch four is an update
strategy inside that loop, not a second training pipeline. It adds no loss
shader, optimizer algorithm or full-model host transfer; batch one does not
allocate gradient accumulation storage or call the batch update API. B4 keeps
the existing command submission boundaries but lets the first three non-logging
views remain queued. The compute queue orders scratch-buffer and mean-gradient
dependencies; submission slots and upload slots are completed before reuse.
Retired tile errors remain latched across slot reuse until replay clears them.
The fourth raster backward completes and the whole batch passes overflow
validation before the fused commit can update any parameter. Logging views
still complete individually for loss readback. Explicit per-view image/validity
buffers are drained before release; ordinary RGB uses the shared upload ring.
Aborting after any of the four views is safe. No separate Adam dispatch or
gradient clearing is needed in B4; the next first view overwrites the buffers.

### Reference images

Training views without a validity file use implicit all-valid pixels, avoiding
full-image allocation and scanning on the CPU. Explicit validity masks retain
their existing thresholding and supervision semantics.

### Two-device engineering path

Two-device pair training is in development and is not part of the first-release
supported surface. The implementation still performs rank-local backward, exact
sparse/dense exchange, and rank-ordered FP32 mean plus Adam; native pair state
is Idle → LocalReady → GlobalReady → Idle. The pair regularizer receives two
scalar timestamps in rank order. Each run owns a PairSession with finite
mailbox timeouts. Keep these contracts when changing collective code; do not
document them as a user workflow.

## Phases and checkpoints

Main training and the hard-separation phase share one optimizer-update
implementation. Hard separation defaults to 3000 steps with one group-local
relocation at local step 1500. It has its own local Adam step, seed, LR/freezes
and fresh moments. It binarizes persistence without zeroing motion, keeps gates
fixed, continues low-rate velocity optimization, uses opacity-and-gradient donor
scoring within each persistence group, and exports the terminal hard-phase state.
Before the split, the runner restores the selected main-phase model rather than
using the last main step. The same retained handoff lets a completed non-HS B1
run add HS later. Resume restores phase configuration without splitting twice.
For the engineering pair path, a phase barrier recreates the pair mailbox and
its generation counter.
Phase-specific relocation/checkpoint hooks still have separate schedule loops.

V6 checkpoints save model and Adam arrays, the phase cursor, optimizer/sample/wall
clocks, sampler identity, phase seeds, data/build fingerprints, and per-rank
rasterizer state. Local batch four uses V7 with an explicit algorithm/sampler
identity and four exposures per optimizer update. Resume loads the saved
effective YAML without rescaling learning rates or schedule length.

`train.checkpoint_interval` is the shared hold-out eval and optional checkpoint
cadence; `train.save_checkpoints` decides whether optimizer snapshots are written.
One numbered checkpoint is written and `latest.json` is atomically replaced when
saving is enabled. Independently of that cadence, `main_selected.npz` retains
the selected ordinary model and matching color affine for an optional HS
handoff; HS writes `hard_final.npz` for its terminal result. These omit
main-phase Adam arrays because HS deliberately starts with fresh moments. One
`gaussians.npz` is the standalone final model. Main-only runs use the best
hold-out joint task-PSNR+SSIM score unless `train.export_best=false`; foreground
runs pair `foreground_psnr` with mask-interior `foreground_ssim`, while scene
runs pair full-frame PSNR with full-frame SSIM. HS runs use the terminal
hard-phase state.
Training also writes `run.export_filename` (default `gaussians_4dgs.ply`);
`ftgspp export` writes another PLY path.

## Evaluation

In-loop hold-out evaluation reduces the task PSNR and matching SSIM_2 on the
GPU; foreground mode uses mask-interior `foreground_ssim` instead of full-frame
SSIM and does not compute foreground
morphology/ROI diagnostics.
Full foreground eval reuses one F/T render for every fixed background and
reduces their PSNR/SSIM plus ROI SSIM on the GPU. CPU work is reserved for the
final silhouette diagnostics. Synchronous replicas shard final validation
views as well as in-loop ranking. LPIPS has no training YAML field and is not
loaded by training or full validation. The explicit locked final-test command
can attach a single-device LPIPS metric extension to the same render pass.
It scores full frames for scene mode and fixed reference-foreground ROIs for
foreground mode, batching equal shapes with bounded host memory. After the
selected export is chosen, a foreground run always writes the diagnostic suite to
`full_validation_metrics.jsonl` (and to `final_test_metrics.jsonl` when a locked
final-test identity is present). `eval.full_validation_on_export` is removed.

## Runtime logging

Interactive training uses one TTY progress bar per phase with throughput and ETA.
The latest sampled loss and hold-out PSNR/SSIM remain in its postfix; normal log
records are redirected safely around the bar. Non-interactive runs retain stable
INFO lines at `train.log_interval`. Every training invocation owns a scoped
`<output_path>/run.log` handler, installed before dataset preflight and removed on
exit; resume appends a new session and fresh training replaces the file.

## Build

PEP 517/scikit-build-core drives CMake and the shader compiler. Wheel builds
require Python 3.12 exactly, `slangc` from Slang 2026.10, Vulkan headers, and
CMake 3.24+ (scikit-build-core may download CMake). The shader job manifest
drives compilation and the installed resource manifest used at startup. All
generated artifacts go to build directories or installed package resources.
Examples stay in the git checkout and sdist; they are not installed by the
wheel. The default wheel is headless and includes offscreen rendering plus the
browser asset; `FTGSPP_BUILD_VIEWER=ON` additionally compiles the X11
GLFW / Dear ImGui module.
