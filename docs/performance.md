# Training performance

This page distinguishes the historical SelfCap train-only timing comparison
from the fixed-model B1/B4 execution benchmark. B1 uses one image per optimizer
update; B4 averages four images before one update. Compare their image
throughput, rather than optimizer updates per second.

## SelfCap timing comparison

Head-to-head on the six SelfCap clips (`dance1`, `dance2`, `corgi1`, `corgi2`,
`bike1`, `bike2`): 60 frames, 500k Gaussians, 30k main-phase optimizer steps,
one training view per step, NVIDIA RTX 4090. Rates in this section are train-only.
Hold-out eval and the separate hard-separation phase are excluded from both
implementations, so the time tables are not total default run times.

| Label in the tables | Implementation | What was trained |
|---|---|---|
| **FTGSPP original** | CUDA + gsplat  | scene RGB |
| **Ours** | Vulkan + vksplat | scene RGB and foreground |

### Timing definitions

**FTGSPP original.** The loop prints tqdm `n/30000 [elapsed, it/s]` and runs a
60-view PSNR eval every 2000 steps. Rates are the median of those 2k-step
windows *between* evals, after dropping the first window (CUDA compile /
warmup) and any window below 30 it/s. MCMC relocation every 100 steps stays in
the number; eval does not.

**Ours.** `performance_summary.json` records `step_seconds` around the optimizer
update only (`_evaluate` is outside the timer). `steps_per_second` is
`30000 / sum(step_seconds)` (mean). `step_seconds_p50` / `p95` are the median /
95th-percentile step. The 244 MCMC relocation events are outside this timer.

The original measurement includes relocation while the Vulkan measurement
excludes it. The ratios below compare those reported timing intervals; they
are not measurements of identical end-to-end workloads.

### Scene RGB: train-only throughput

`it/s` is optimizer steps per second (higher is better). `ms/step` is
`1000 / it/s`. Ours columns use the mean step rate; `p50` is the typical step
with less log-sync noise. The summary row averages the per-scene rates, derives
`ms/step` from those mean rates, and divides the two mean rates for its speedup.

| Scene | FTGSPP original it/s | ms/step | Ours it/s | ms/step | p50 it/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| dance1 | 44.1 | 22.7 | 138.6 | 7.2 | 151.3 | **3.14×** |
| dance2 | 61.2 | 16.3 | 143.7 | 7.0 | 153.2 | **2.35×** |
| corgi1 | 43.1 | 23.2 | 131.6 | 7.6 | 144.5 | **3.05×** |
| corgi2 | 45.1 | 22.2 | 69.2 | 14.5 | 74.5 | **1.54×** |
| bike1 | 47.4 | 21.1 | 97.7 | 10.2 | 109.6 | **2.06×** |
| bike2 | 54.1 | 18.5 | 106.0 | 9.4 | 118.3 | **1.96×** |
| **Unweighted avg** | **49.2** | **20.3** | **114.5** | **8.7** | **125.2** | **2.33×** |

### Scene RGB: time for 30,000 main-phase steps

Wall time implied by the train-only rates above (`30000 / it/s`). Lower is
better. Evaluation and hard separation are excluded. The summary row averages
the per-scene durations and divides those averages; this is the source of the
2.22× aggregate speedup. It differs from the ratio of mean rates above.

| Scene | FTGSPP original | Ours | Speedup |
|---|---|---|---|
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

### Foreground protocol (Vulkan)

FTGSPP original `runs_main_da3` is scene RGB, so there is no original-foreground
column. Same 30k / 500k / 60-frame recipe; rates are again
`30000 / sum(step_seconds)`. The summary row averages rates and durations
separately; its `ms/step` is the reciprocal of the mean rate.

| Scene | Ours it/s | ms/step | 30k train |
|---|---:|---:|---:|
| dance1 | 147.1 | 6.8 | 3.4 min (204 s) |
| dance2 | 130.0 | 7.7 | 3.8 min (231 s) |
| corgi1 | 101.0 | 9.9 | 5.0 min (297 s) |
| corgi2 | 100.3 | 10.0 | 5.0 min (299 s) |
| bike1 | 177.5 | 5.6 | 2.8 min (169 s) |
| bike2 | 301.8 | 3.3 | 1.7 min (99 s) |
| **Avg** | **159.6** | **6.3** | **3.6 min (216 s)** |

In these measurements, foreground is faster on the square `bike*` clips
(smaller raster) and slower on `corgi*` than the Vulkan scene-RGB profile.
The foreground and scene objectives differ, so their rates describe workload
cost rather than equivalent reconstruction quality.

## Local batch execution cost

See [batch-size configuration](usage.md#batch-size) for B4's supported scope.
B1 keeps its fused backward/Adam update. B4 reuses the same parameter-backward
and fused-commit shader sources: the first three views accumulate gradients
with weight 0.25 into the existing FP32 Adam input buffers; the fourth finishes
the mean in registers and updates once. The first view overwrites old gradients,
including after an aborted batch. There is no separate mean buffer or copy-back.

Non-logging batches queue the first three views without unconditional per-view
completion waits. Existing ring-slot reuse still waits where necessary, and
the complete batch must pass overflow validation before commit. Logged batches
retain per-view loss readback. Retired errors survive submission-slot reuse;
explicit image/validity buffers are completed before release. Both B1 and B4
skip construction/scanning of an implicit all-valid image when no validity
file exists. These changes require no new configuration and preserve the
objective, optimizer, sampler order and V6/V7 checkpoint contracts.

Measured on 2026-10-06, RTX 4090 (PCI 0000:42:00.0), CPU affinity 4–7. Each
mode has five 512-exposure rounds, alternating B1/B4 order with 32 warm-up
exposures. Models, 32 decoded reference views, SH degree 3 and rasterizer choices
are fixed. Image binding/upload and GPU completion are included; disk decoding,
startup, evaluation, logging and relocation are excluded. Updates are below
FP32 resolution and unchanged positions are checked after every round. This is
execution throughput, not total training time or a convergence/quality result.

The baseline already includes direct gradient accumulation and fused B4 commit.
The final build adds the host/async changes above. Results below are the last
complete baseline/final runs on the same device; entries are median images/s.

| Dataset / scene | Baseline B1 | Final B1 | Baseline B4 | Final B4 | B4 change |
|---|---:|---:|---:|---:|---:|
| 4DAnyone / karina | 449.10 | 449.65 | 305.16 | 466.50 | +52.9% |
| 4DAnyone / engineait800 | 478.35 | 478.36 | 383.70 | 498.37 | +29.9% |
| DNA-Rendering / 0094_02 | 115.65 | 177.02 | 86.09 | 148.39 | +72.4% |

The final build's B4 exceeds B1 on these two 4DAnyone workloads; DNA remains
faster with B1. Earlier repeat runs of the async build put the two 4DAnyone B4
rates near 466 and 498 images/s as well. DNA and baseline B4 timings fluctuate
more, so medians are observations rather than guaranteed gains. All 66 SPIR-V
files are byte-identical between this baseline and final build.

A common upload prototype using a second VkQueue in the compute family and
independent upload/compute timeline semaphores regressed 4DAnyone throughput
(about 16% for B1 and 8% for B4) and was removed. This result applies to that
prototype, not every possible dedicated-transfer design. Upload and compute
retain the existing shared queue. Unused transfer queue aliases, family flags
and the never-created separate command-pool branch were also removed.

Reproduce using a read-only trained model and its resolved dataset config:

```bash
taskset -c 4-7 .venv/bin/python dev/tests/gpu/benchmark_training_views.py \
  --device 5 --case /path/to/config.yaml /path/to/gaussians.npz \
  --rounds 5 --exposures 512 --rasterizer-states /tmp/rasterizer-states.json \
  --output /tmp/real-scene-benchmark.json
```

Choose an idle Vulkan device and CPU cores appropriate for the host. Run builds
sequentially, reusing the first build's frozen rasterizer-state file. Inspect
all rounds, not only medians. Run `--profile` separately: GPU stage queries and
per-update telemetry perturb throughput, and CPU fence time includes useful
GPU work. B1 resets telemetry after image binding while B4 resets before it,
so their upload/submission counters are not directly comparable.

`dev/tests/gpu/benchmark_local_batch.py` provides a data-free synthetic benchmark.
`check_local_batch.py` independently checks gradients/Adam, SH boundaries and
padding, actual and injected overflow replay, interrupted batches, state
restoration, and image/validity lifetime across ring reuse.
