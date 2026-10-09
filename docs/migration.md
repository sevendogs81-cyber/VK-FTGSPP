# Breaking community-release migration

There is no implicit migration of old execution semantics.

- New point preparation uses per-frame silhouette bounds with DA3 fallback,
  frame-scoped map reuse, bounded RGB prefetch, and budget-adaptive color queries.
  `--quality balanced` uses `4 × ceil(init.num_gaussians / selected keyframes)`
  for generated geometry; supplied clouds remain uncapped. Geometry still supplies
  full occlusion coverage. `--bounds-method da3 --quality full` restores the old
  bound/density choices. New manifests are `ftgspp-visual-hull-ibr-v4` and
  `ftgspp-point-color-v2`; validated v2/v3 and colorize v1 outputs remain readable.
  First-keyframe DA3 camera recovery is accepted only after validating that exact
  request against the complete current dataset and model identity. See
  [arguments](arguments.md#max_points-per-keyframe-versus-the-total-model).

- `setup.sh` now remembers enabled extras and the desktop build in `.venv`.
  Use `--reset-profile` to replace that selection. Legacy environments without
  a saved profile must specify their desired options once. The Torch extras now
  pin Torch 2.6.0 / TorchVision 0.21.0 (Linux CUDA 12.4).
- `prepare points --method colorize` accepts XYZ-only PLYs and writes versioned
  `ftgspp-point-color-v2` outputs for `init.backend: points`. Image coloring is
  optional (`--color-mode gray`). Preparation requires a new/empty output
  directory to avoid overwriting source clouds or completed caches.

- New foreground runs default to full input resolution, keyframe stride 1,
  three-neighbor velocity initialization, a frame-adaptive training budget, and the gentler
  temporal/relocation recipe documented in [usage.md](usage.md#defaults-for-a-first-foreground-run).
  `train.preset: balanced` uses 70k updates at 121 frames and 174k at 300 frames.
  `preview` and `extended` select smaller/larger budgets. Relocation and eval
  clocks scale with the effective horizon unless explicitly overridden. Remove
  copied numeric settings (or set `train.iterations: null`) to enable adaptation.
  Saved effective clocks remain unchanged; see [arguments.md](arguments.md).
  Generated starters and foreground examples now use `points_vhull_ibr_stride1`
  (or `points_vhull_filtered_stride1` for point filtering). Regenerate geometry
  when changing a capture's stride. Explicit YAML overrides and saved resolved
  run plans keep their original settings.
  Both modes with `init.backend: points` now default to the selected source PLY
  population when no numeric budget is specified. Use `init.num_gaussians: null`
  to clear a copied numeric override. Saved plans contain the resolved integer.
- New scene runs now use the same `train.preset` interface with a separate
  60-frame / 30k-main-step reference. Balanced 300-frame runs use 150k main
  steps plus 15k HS steps; warm-ups, relocation and eval clocks scale together.
  An omitted scene keyframe stride follows FPS at roughly 1/6-second spacing
  (30 FPS: 5; 60 FPS: 10). Generated generic starters leave that choice adaptive.
  Existing PLYs require an explicit matching stride. Saved numeric run plans
  and explicit user settings are retained. See [arguments.md](arguments.md#scene-adaptation).
- Replace Hydra groups/launcher with `ftgspp train --config scene.yaml` and
  repeated `--set key=value`. Personal scene presets are no longer distributed.
- Remove `pipeline.stages` and `pipeline.check_only`. Training has one fixed
  initialize/resume → train → export sequence; use the separate doctor, prepare,
  eval and export CLI commands for those operations.
- Remove `distributed.global_batch_size`, `K`/`microsteps` and configurable
  transfer-ring depths. `train.batch_size` now accepts only 1 (default) or 4;
  four-view accumulation is limited to ordinary single-device foreground runs.
- Default training retains V6 checkpoints. Local batch four uses V7 with
  an explicit algorithm identity; a V6 run cannot be resumed as a batch-four run.
  Old Torch/field conversion and compatibility models have been removed.
  Retain an old checkout/environment to inspect old artifacts.
- Python modules now live under `src/ftgspp/{config,data,training,io,optional}/`.
  Numerical references and smoke helpers are tests, not installed public APIs.
- The only synchronous backend is `pair_host` (`auto` resolves to it);
  the former `hierarchical_host` name and P≥3 implementations are removed.
  Two-device sync is in development in this community release; the documented
  first-release path is one GPU.
- The pair optimizer takes two scalar timestamps. There is no gradient finalizer,
  arbitrary batch-size setting, host gradient uploader or CPU gradient reducer.
  Unknown native configuration fields fail explicitly.
- Initialization stays in memory. The redundant standalone `separated.npz` is gone.
  The final standalone model is `gaussians.npz`. A main-only run selects it by
  hold-out joint PSNR+SSIM (`train.export_best`, default true); an enabled hard
  separation phase starts from that main selection and exports its terminal
  state. `ckpt/main_selected.npz` retains the phase handoff and
  `ckpt/hard_final.npz` retains the terminal hard-phase state. Optimizer
  checkpoints are optional
  (`train.save_checkpoints`) and share `train.checkpoint_interval` with eval.
- `scripts/setup.sh` now installs the Torch-free Vulkan training runtime by
  default. Use `--with-birefnet`, `--with-da3`, or `--with-init`; add
  `--with-lpips` only for optional locked final-test LPIPS. The former
  aggregate `[foreground]` extra and `--core-only` switch are removed.
- `FTGSPP_TRAIN_EVAL_INTERVAL` is removed. Mid-run hold-out eval uses
  `train.checkpoint_interval`. Set `train.save_checkpoints: false` to evaluate
  without writing optimizer checkpoint files.
- `eval.lpips`, `eval.lpips_device`, `eval.lpips_batch_size`, and
  `eval.lpips_min_roi_size` are removed from training YAML. Optional LPIPS is
  requested only on a locked test split with `ftgspp eval --lpips`; device,
  batch size, and minimum region size are command-line options.
- `eval.full_validation_on_export` is removed. Hold-out ranking (scene full-frame
  PSNR+SSIM_2 or foreground object PSNR+mask-interior foreground SSIM_2) runs on
  the GPU. Foreground
  diagnostic metrics are computed after the best snapshot is selected, as a
  required training-pipeline step, not a per-dataset switch. Old YAML that still
  names the retired key is ignored.
- `run.log_dir` is removed. Training always writes stable application logs to
  `<output_path>/run.log`; resumed sessions append to that file.
- Source builds require pybind11 3.0.1 for NumPy 2: the old 2.11.1 boundary
  produced zero-stride one-dimensional arrays and incorrect relocation replay.
- Generic captures use `data.view_manifest` (`ftgspp-dataset-v1`);
  `ftgspp prepare dataset` creates that index and a starter YAML.
  `ftgspp prepare` with no subcommand, and `ftgspp prepare --help`, list the
  prepare commands. SelfCap is `ftgspp prepare selfcap`. A flags-only invocation
  such as `ftgspp prepare --scene-root ...` still forwards to SelfCap as a
  compatibility alias and prints a warning.
- Foreground auto initialization no longer trusts a manifestless PLY directory.
  Generated visual-hull manifests are accepted automatically; choose
  `init.backend: points` to acknowledge unversioned user geometry.
- Earlier GPU-capable preparation used `ftgspp-visual-hull-ibr-v3` to record
  geometry/coloring devices and the explicit `carving` or `point_filtering`
  method. Those v2/v3 artifacts remain readable alongside new v4 artifacts.
  Select the matching `init.backend`; generic/user PLYs use `points`.
- New Gaussian initialization manifests use `ftgspp-init-l3-v4`. The L3 change
  replaces repeated full-cloud voxel searches with bounded sampling plus exact
  calibration, and preallocates final model arrays. Existing L2 DA3 point
  caches remain valid; rerunning initialization creates the new L3 result.
- `pipeline.foreground_version`, `pipeline.foreground_preset`,
  `train.image_objective`, and `eval.protocol` are rejected. Foreground training
  is `composite_v1`; the objective and evaluation protocol are derived from
  `pipeline.mode`.
- New final-test manifests should use `ftgspp-final-test-v2`; evaluation rejects
  frame/camera, path, or RGB-content overlap with checkpoint-selection views.
