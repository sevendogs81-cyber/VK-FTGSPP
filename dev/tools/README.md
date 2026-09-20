# Research tools

These scripts are **unsupported**. They are not part of the installed CLI
and may change without notice. Official evaluation is `ftgspp eval`.

Run them from a checkout with `src/` on `PYTHONPATH`, for example:

```bash
PYTHONPATH=src python dev/tools/evaluate_task_run.py --help
```

## Evaluation

- `ftgspp eval` — supported command for a completed run or locked final-test split.
- `evaluate_task_run.py` — writes a full-validation JSON report for a selected
  export. Unstable.
- `evaluate_scene_run.py` — scene-only PSNR / SSIM_2 / LPIPS-VGG helper with a
  separate LPIPS convention. Unstable; not used for foreground runs.
- `summarize_task_evaluations.py` — aggregates task-evaluation JSON files.

## Other helpers

- `jsonl_to_tensorboard.py` — optional TensorBoard export from run JSONL.
- `bake_undistort_images.py` — offline undistort cache baker.
- `subsample_ply.py` — point-cloud thinning.
- `monitor_gpu_run.py` — `nvidia-smi` sampler beside a training run.
- `audit_selfcap_initialization.py` — read-only replay of recorded SelfCap
  initialization (`ftgspp-init-l3-v4` and historical v1/v2/v3 manifests).
