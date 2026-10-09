# Developer performance benchmarks

User-facing comparisons with the original FTGSPP implementation, including all
SelfCap results and timing definitions, are in [performance](../../docs/performance.md).
This page covers internal B1/B4 optimization, profiling and reproduction.
B1 uses one image per optimizer update; B4 averages four images before one update.

## Local batch execution cost

See [batch-size configuration](../../docs/arguments.md#batch-size) for B4's supported scope.
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
