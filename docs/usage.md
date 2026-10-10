# Work with a capture

Start a new capture with [the first-run guide](first-run.md). Run the commands
below from the repository with `.venv` activated, replacing these paths:

```bash
CONFIG=/absolute/path/to/capture/ftgspp.yaml
RUN=/absolute/path/to/capture/runs/foreground
```

## Resume training

```bash
ftgspp train --resume "$RUN"
```

Resume keeps the saved settings. Retain the original environment for older
incompatible checkpoints. To change settings, start a new run as below.

## Change settings

Inspect the settings, then use a new output directory for each trial:

```bash
ftgspp train --config "$CONFIG" --print-config
ftgspp train --config "$CONFIG" --output /path/to/new-run \
  --set train.preset=preview
```

| What you want | Setting or action |
|---|---|
| A short trial | `train.preset=preview` |
| A normal first run | `train.preset=balanced` (default) |
| A longer run | `train.preset=extended` |
| An explicit number of steps | `train.iterations=90000`; use `null` to return to the preset |
| Lower image-memory use | Reduce `data.scale`, for example to `0.5` |
| A smaller model | Reduce `init.num_gaussians` |
| A shorter clip | Set `data.frames.start` and `.stop`; stop is exclusive |
| Correct capture timing | Set `data.fps` to the indexed image rate |
| Save disk space | `train.save_checkpoints=false`; optimizer resume then lacks periodic snapshots |

Preview shortens training but keeps the model size. Presets adapt to the clip;
use `--print-config` to see the resulting step count. Prepared geometry must
match the image scale, frame range, calibration and camera split. Regenerate
it in a new directory if these inputs change.

YAML paths are relative to the YAML file; CLI paths are relative to your working
directory. Use `${oc.env:VARIABLE}` for environment variables inside YAML.

## Find the outputs

| File | Use |
|---|---|
| `gaussians.npz` + `viewer_manifest.json` | Open the result in the viewer |
| `config.resolved.yaml` | Review the settings actually used |
| `run.log` | Read progress or diagnose a failure |
| `ckpt/` | Resume training |
| `export_selection.json` | Identify the exported model |
| `gaussians_4dgs.ply` | Compatibility export; use NPZ for native viewing |

For an older run missing viewer metadata, use
`ftgspp prepare viewer --run "$RUN"` before opening it.

## Evaluate

```bash
ftgspp eval --run "$RUN" --device 0
```

This evaluates all hold-out views selected by `eval.frame_interval` and writes
`full_validation_metrics.jsonl`. Add `--quick` to evaluate the smaller selection
subset and write `metrics.jsonl`. `--quick` cannot be used with a locked final test.

For an independent final test, fill in the [manifest template](../examples/final_test_manifest.example.json)
with views not used for training or checkpoint selection:

```bash
ftgspp prepare final-test --run "$RUN" --manifest /path/to/final_test.json
ftgspp eval --run "$RUN" --device 0 --test-manifest /path/to/final_test.json
```

Results go to `final_test_metrics.jsonl`. For LPIPS, install `--with-lpips` and
add `--lpips` to the final-test command; weights may download on first use.

## Troubleshoot

| Symptom | Check first |
|---|---|
| Blurred motion or double edges | Capture FPS, synchronization, masks and point-cloud coverage |
| Missing or misplaced subject | Camera IDs, calibration, mask alignment and point coordinates |
| Out of memory | Other GPU jobs; then image scale or Gaussian count |
| Unexpected DA3 download | Missing calibration or incomplete input point clouds |
| Existing output error | Resume the run, or choose a new directory |
| Viewer connection failure | [Viewer troubleshooting](viewer.md#troubleshooting) |

Compare a paused render with the source image at the same camera and time.
To report a problem, include the command, commit, doctor output, log excerpt and
resolved config, removing private paths and viewer tokens.

For SelfCap videos, use `ftgspp prepare selfcap --help`.
