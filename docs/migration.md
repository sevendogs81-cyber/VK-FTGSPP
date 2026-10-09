# Upgrade an existing installation

New users can start with [installation](../README.md#install) and
[the first-capture guide](first-run.md).

## Update the environment

Stop jobs using `.venv` before rerunning `scripts/setup.sh`. Setup remembers
previously enabled components. If your older installation has no saved profile,
pass all desired flags once, for example `--with-init --with-remote`.
Use `--reset-profile` only to replace the selection.

## Reuse a config or run

- Resume existing runs with `ftgspp train --resume RUN`; their saved settings
  stay fixed. Retain the original checkout/environment for incompatible older
  checkpoints, especially legacy Torch or field models.
- For a new capture, generate a new YAML with `ftgspp prepare dataset` instead
  of copying an old research config. Review FPS, frames, camera split and the
  effective settings with `ftgspp train --config FILE --print-config`.
- New runs use adaptive training budgets. Remove a copied `train.iterations`
  value, or set it to `null`, to use the preset. See [training settings](arguments.md).
- Current foreground initialization uses every frame. When changing stride,
  scale or camera split, regenerate matching geometry in a new directory.
  Existing point clouds need their actual `--keyframe-stride` when indexing.
- For an older model missing viewer metadata, use
  `ftgspp prepare viewer --run RUN`. See [older models](viewer-reference.md#open-older-models).

## Replace old options

| Old option | Current usage |
|---|---|
| `--core-only` or aggregate foreground extra | Core setup needs no flags; add `--with-init` for masks and DA3 |
| `FTGSPP_TRAIN_EVAL_INTERVAL` | Set `train.checkpoint_interval` in YAML |
| LPIPS keys in training YAML | Use `ftgspp eval --test-manifest FILE --lpips` |
| `run.log_dir` | Logs are always written to `RUN/run.log` |
| Separate image-objective / eval-protocol selectors | Choose `pipeline.mode: scene` or `foreground` |

For a config rejected because of a removed research option, generate a fresh
starter and transfer only the settings you need. The full internal migration
record is kept in [development history](../dev/docs/migration-history.md).
