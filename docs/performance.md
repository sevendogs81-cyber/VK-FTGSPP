# Training speed (SelfCap)

Head-to-head on the six SelfCap clips (`dance1`, `dance2`, `corgi1`, `corgi2`,
`bike1`, `bike2`): 60 frames, 500k Gaussians, 30k main-phase optimizer steps,
one training view per step, NVIDIA RTX 4090. **Every rate below is train-only.**
Hold-out eval and the separate hard-separation phase are excluded from both
implementations, so the time tables are not total default run times.

| Label in the tables | Implementation | What was trained |
|---|---|---|
| **FTGSPP original** | CUDA + gsplat  | scene RGB |
| **Ours** | Vulkan + vksplat | scene RGB and foreground |

## How train-only speed is measured

**FTGSPP original.** The loop prints tqdm `n/30000 [elapsed, it/s]` and runs a
60-view PSNR eval every 2000 steps. Rates are the median of those 2k-step
windows *between* evals, after dropping the first window (CUDA compile /
warmup) and any window below 30 it/s. MCMC relocation every 100 steps stays in
the number; eval does not.

**Ours.** `performance_summary.json` records `step_seconds` around the optimizer
update only (`_evaluate` is outside the timer). `steps_per_second` is
`30000 / sum(step_seconds)` (mean). `step_seconds_p50` / `p95` are the median /
95th-percentile step. MCMC relocate is also outside the timer; 244 relocations
do not change the speedups below.

## Scene RGB — train-only throughput

`it/s` is optimizer steps per second (higher is better). `ms/step` is
`1000 / it/s`. Ours columns use the mean step rate; `p50` is the typical step
with less log-sync noise.

| Scene | FTGSPP original it/s | ms/step | Ours it/s | ms/step | p50 it/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| dance1 | 44.1 | 22.7 | 138.6 | 7.2 | 151.3 | **3.14×** |
| dance2 | 61.2 | 16.3 | 143.7 | 7.0 | 153.2 | **2.35×** |
| corgi1 | 43.1 | 23.2 | 131.6 | 7.6 | 144.5 | **3.05×** |
| corgi2 | 45.1 | 22.2 | 69.2 | 14.5 | 74.5 | **1.54×** |
| bike1 | 47.4 | 21.1 | 97.7 | 10.2 | 109.6 | **2.06×** |
| bike2 | 54.1 | 18.5 | 106.0 | 9.4 | 118.3 | **1.96×** |
| **Unweighted avg** | **49.2** | **20.3** | **114.5** | **8.7** | **125.2** | **2.33×** |

## Scene RGB — time for 30k main-phase steps, eval and HS excluded

Wall time implied by the train-only rates above (`30000 / it/s`). Lower is
better.

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

## Foreground protocol (Ours)

FTGSPP original `runs_main_da3` is scene RGB, so there is no original-foreground
column. Same 30k / 500k / 60-frame recipe; rates are again
`30000 / sum(step_seconds)`.

| Scene | Ours it/s | ms/step | 30k train |
|---|---:|---:|---:|
| dance1 | 147.1 | 6.8 | 3.4 min (204 s) |
| dance2 | 130.0 | 7.7 | 3.8 min (231 s) |
| corgi1 | 101.0 | 9.9 | 5.0 min (297 s) |
| corgi2 | 100.3 | 10.0 | 5.0 min (299 s) |
| bike1 | 177.5 | 5.6 | 2.8 min (169 s) |
| bike2 | 301.8 | 3.3 | 1.7 min (99 s) |
| **Avg** | **159.6** | **6.3** | **3.6 min (216 s)** |

Foreground is faster on the square `bike*` clips (smaller raster) and slower
on `corgi*` than Ours scene RGB. Across six scenes it still beats FTGSPP
original scene RGB (~49 it/s) by about **3.2×**.
