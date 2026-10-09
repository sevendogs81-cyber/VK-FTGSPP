# Training performance

Training time depends on image resolution, clip length, Gaussian count and GPU.
Use a preview run on your capture to estimate the cost of a longer run.

## Reference measurements

Historical SelfCap measurements on an RTX 4090 used 60 frames, 500k Gaussians
and 30,000 main steps:

| Mode | Reported training time across six captures |
|---|---|
| Scene | 3.5–7.2 minutes; mean 4.7 minutes |
| Foreground | 1.7–5.0 minutes; mean 3.6 minutes |

These are optimizer timings, excluding preparation, evaluation, relocation and
hard separation. Current defaults use a larger model and adaptive step counts;
these figures are not estimates of a complete default run. The
[detailed measurements](../dev/docs/benchmarks.md#selfcap-timing-comparison)
record the workload and timing definitions.

## Local batch execution cost

Start with the default batch size of one. Batch size four can improve image
throughput on some foreground captures, but uses more memory and does not
consistently outperform batch size one. It also changes the number of images
seen per optimizer update, so equal iteration counts are different budgets.
See [batch-size settings](arguments.md#batch-size) before changing it.

## Estimate your own run

Check the effective iteration count with `ftgspp train --print-config` using
your config. Watch the training progress and ETA, and inspect
`performance_summary.json` afterward. Allow extra time for preparation,
validation and export. Keep the same resolution and model size when comparing
presets or machines.
