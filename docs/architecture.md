# Core boundaries

The installed CLI resolves user YAML once, validates an immutable effective plan,
loads a shared Scene/Camera/View manifest, initializes or resumes explicit
Gaussians, trains, then exports. This is a fixed sequence, not a configurable
stage list. Evaluation and PLY export consume explicit model/state outputs.

- `src/ftgspp/config`: schema, path rules, immutable ResolvedRunPlan and validation.
- `src/ftgspp/data`: preparation, calibration, timestamps, lazy pixel loading.
- `src/ftgspp/training`: sampling, initialization, single/pair optimizer updates,
  two phase schedules, evaluation and bounded image prefetch.
- `src/ftgspp/io`: explicit NPZ models, V6 checkpoints, PLY.
- `src/ftgspp/optional`: evaluation-only LPIPS and the lazy DA3 initializer; default
  imports do not load Torch or any upstream model package.
- `native/src/device`: buffers, compute/transfer submission, timing and telemetry.
- `native/src/render`: projection/rasterization and adaptive dispatch selection.
- `native/src/training`: backward, Adam, relocation and hard-separation kernels.
- `native/src/collective`: session-scoped pair transport and generation state.
- `native/src/bindings`: NumPy/Python boundary; `common` holds native configuration.
- `native/slang` and `native/shaders/radix_sort`: shader source only.
- `tests/reference`: numerical reference implementations, not installed runtime.
- `tools`: unsupported research helpers for dataset baking, TensorBoard and
  monitoring. Official evaluation is `ftgspp eval`.

## Initialization decision tree

`Scene.from_config` scans the complete image grid before any model is allocated
and checks for a complete EasyMocap calibration pair. With calibration, the
existing camera/world geometry is retained; `init.backend=auto` first consumes a
complete `f<frame>.ply` set and otherwise invokes pose-conditioned DA3. With no
calibration, the loader creates provisional pinhole views, and `auto`/`da3`
invokes DA3 on the first keyframe to estimate one static rig. Large rigs are
processed in bounded chunks with shared anchor cameras; later keyframes are
conditioned on that rig. Auto DA3 fuses every camera on small rigs and a
geometrically spread subset on dense arrays, requires two-view depth support
when a second camera exists, and applies the sky mask only for monocular
fusion. DA3 depth is unprojected into world points and its
processed RGB initializes the same SH-DC coefficient used by PLY input. Each
keyframe's views are voxel-fused in world space before any point cap; the fused
dense cloud is cached as the 3-NN velocity reference, and Gaussian means are
spatially sampled from it to `init.num_gaussians // keyframes`. Explicit
Gaussian conversion (algorithm `ftgspp-init-l3-v3`) time-anchors means so
`x(t_ref)` recovers the source point, isolates unique-source KNN scales from
replica points, and records L3 identity separately from the DA3 L2 cache.
The initializer has one implementation path: sequential camera chunks,
cross-view depth consistency, adaptive surface-grid sampling, and exact adaptive
Gaussian budgeting. The
estimated rig and generated PLYs are cached together, so a subsequent run can
resume without re-estimation or consume a user PLY set in the cached coordinate
system. Per-point attribute sidecars are not written because L3 consumes only
XYZ/RGB. New DA3 cache directories use a readable configuration prefix
(`scene-init0-kf60-v8-r504-<12hex>`) followed by a collision-resistant short
identity. `init.seed` can pin this namespace independently of `train.seed`.
`init.backend=points` remains an explicit, calibrated/user-cloud path.

## One update

Single GPU uses the fused backward/Adam path. A pair performs rank-local backward,
exact sparse/dense exchange, rank-ordered FP32 mean and Adam on each GPU.
The native pair state is Idle → LocalReady → GlobalReady → Idle.
Duplicate backward/exchange/Adam calls are rejected; a forward loss is consumed
exactly once, including the fused single-device path. Overflow rollback restores
relocation statistics and explicitly replays the same generation and views.

The pair regularizer receives two scalar timestamps in rank order via push
constants. Its mean, pixel reductions, cross-rank mean, Adam moments and
relocation history are essential mathematics, not local gradient accumulation.

Each run owns a PairSession. Mailbox and generation waits have finite configured
timeouts. A worker error aborts peers before joining futures. No process-global
collective owner is shared across independent runs.

## Phases and persistence

Main training and the 5000-step hard-separation phase share one optimizer-update
implementation. Hard separation has its own local Adam step, seed, LR/freezes and
fresh moments; resume restores phase configuration without splitting twice.
A phase barrier recreates the pair mailbox and its generation counter.
Phase-specific relocation/checkpoint hooks still have separate schedule loops.

V6 saves model/Adam arrays, phase cursor, optimizer/sample/wall clocks, sampler
identity and phase seeds, data/build fingerprints and per-rank adaptive rasterizer
state. The saved effective YAML is loaded without rescaling LR or schedule length.
`train.checkpoint_interval` is the shared hold-out eval and optional checkpoint
cadence; `train.save_checkpoints` decides whether optimizer snapshots are written.
One numbered checkpoint is written and latest.json is atomically replaced when
saving is enabled. One gaussians.npz is the standalone final model and comes from
the best hold-out joint PSNR+SSIM score unless `train.export_best=false`; PLY is
an explicit export. In-loop hold-out eval reduces PSNR and SSIM_2 on the GPU and
does not compute foreground morphology/ROI diagnostics. After the selected export
is chosen, a foreground run always writes the diagnostic suite to
`full_validation_metrics.jsonl` (and to `final_test_metrics.jsonl` when a locked
final-test identity is present). `eval.full_validation_on_export` is removed.

Interactive training uses one TTY progress bar per phase with throughput and ETA.
The latest sampled loss and hold-out PSNR/SSIM remain in its postfix; normal log
records are redirected safely around the bar. Non-interactive runs retain stable
INFO lines at `train.log_interval`. Every training invocation owns a scoped
`<output_path>/run.log` handler, installed before dataset preflight and removed on
exit; resume appends a new session and fresh training replaces the file.

## Build

PEP 517/scikit-build-core drives CMake and the shader compiler. The shader job
manifest drives compilation and the installed resource manifest used at startup.
All generated artifacts go to build directories or installed package resources.
