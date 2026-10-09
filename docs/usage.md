# Resume, evaluate and troubleshoot

For a new capture, start with [the training guide](first-run.md).
The commands below use `RUN=/absolute/path/to/your/completed-run` and Vulkan
device `0`; replace them with your own values.

## Resume training

```bash
ftgspp train --resume "$RUN"
```

Resume uses the saved configuration. To change settings, start a new run with
`--config` and a new `--output` directory instead.

## Find your outputs

| File | Use |
|---|---|
| `gaussians.npz` + `viewer_manifest.json` | Open the trained model in the viewer |
| `config.resolved.yaml` | See the settings actually used |
| `run.log` | Read progress or diagnose a failure |
| `ckpt/` | Resume checkpoints |
| `export_selection.json` | Identify the selected model |
| `gaussians_4dgs.ply` | Compatibility export; use NPZ for faithful native viewing |

## Evaluate

```bash
ftgspp eval --run "$RUN" --device 0
```

Ordinary evaluation uses sampled hold-out views and writes `metrics.jsonl`.
Foreground training also writes diagnostic results to
`full_validation_metrics.jsonl` at completion. Cameras used to select the best
checkpoint are validation data, not an independent final test.

For an independent test, fill in the [manifest template](../examples/final_test_manifest.example.json)
with views not used for training or selection:

```bash
ftgspp prepare final-test --run "$RUN" --manifest /path/to/final_test.json
ftgspp eval --run "$RUN" --device 0 --test-manifest /path/to/final_test.json
```

These results go to `final_test_metrics.jsonl`. Optional VGG LPIPS requires
`bash scripts/setup.sh --with-lpips`; add `--lpips` to the final-test command.
Weights may download on first use.

## When results look wrong

Inspect several recorded times, including fast motion, from a held-out camera.
Match the source image's camera and timestamp; use a paused, faithful PNG for
detail rather than a compressed video.

| Symptom | Check first |
|---|---|
| Blurred motion or double edges | Capture FPS, synchronization, masks and initial point-cloud coverage |
| Missing or misplaced subject | Camera IDs, calibration, mask alignment and point-cloud coordinates |
| Out of memory | Other GPU jobs; then lower image scale or Gaussian count in a new experiment |
| Unexpected DA3 download | Missing calibration or an incomplete initial point-cloud set |
| Existing output error | Resume the run, or choose a new directory for a new experiment |
| Viewer connection failure | [Viewer troubleshooting](viewer.md#troubleshooting) |

If changing image scale, frames or camera splits, regenerate matching prepared
geometry. More iterations alone cannot repair incorrect inputs. See
[training settings](arguments.md) for budget and memory choices.

## Report a problem

Include the command, repository commit, doctor output, relevant log excerpt and
resolved config. Remove private paths and viewer tokens before sharing.

For SelfCap video input, start with `ftgspp prepare selfcap --help`.
