# Breaking community-release migration

There is no implicit migration of old execution semantics.

- Replace Hydra groups/launcher with `ftgspp train --config scene.yaml` and
  repeated `--set key=value`. Personal scene presets are no longer distributed.
- Remove `pipeline.stages` and `pipeline.check_only`. Training has one fixed
  initialize/resume → train → export sequence; use the separate doctor, prepare,
  eval and export CLI commands for those operations.
- Remove train.batch_size, distributed.global_batch_size, K/microsteps and
  configurable transfer-ring depths. One view is used per device and update.
- Use V6 checkpoints only. Old Torch/field conversion and compatibility models
  have been removed. Retain an old checkout/environment to inspect old artifacts.
- Python modules now live under config/data/training/io/optional in src layout.
  Numerical references and smoke helpers are tests, not installed public APIs.
- The only synchronous backend is `pair_host` (`auto` resolves to it);
  the former `hierarchical_host` name and P≥3 implementations are removed.
- Native timestamps are two scalar arguments. There is no gradient finalizer,
  arbitrary batch-size setting, host gradient uploader or CPU gradient reducer.
  Unknown native configuration fields fail explicitly.
- Initialization stays in memory. The redundant standalone separated.npz is gone.
  The final standalone model is gaussians.npz, selected by the hold-out
  joint PSNR+SSIM score (`train.export_best`, default true). Optimizer
  checkpoints are optional
  (`train.save_checkpoints`) and share `train.checkpoint_interval` with eval.
- `FTGSPP_TRAIN_EVAL_INTERVAL` is removed. Mid-run hold-out eval uses
  `train.checkpoint_interval`. Set `save_checkpoints: false` to eval without
  writing ckpt files.
- `eval.full_validation_on_export` is removed. Hold-out ranking (scene PSNR or
  foreground object PSNR, plus SSIM_2) always runs on the GPU. Foreground
  diagnostic metrics are computed after the best snapshot is selected, as a
  required library pipeline step, not a per-dataset switch. Old YAML that still
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
- `pipeline.foreground_version`, `pipeline.foreground_preset`,
  `train.image_objective`, and `eval.protocol` are rejected. Foreground training
  is `composite_v1`; the objective and evaluation protocol are derived from
  `pipeline.mode`.
- New final-test manifests should use `ftgspp-final-test-v2`; evaluation rejects
  frame/camera, path, or RGB-content overlap with checkpoint-selection views.
