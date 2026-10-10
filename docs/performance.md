# Results

![Dynamic reconstruction on DNA-Rendering](assets/vk-ftgspp_dna_teaser.gif)

## Comparison with FTGSPP

SelfCap · NVIDIA RTX 4090 · reported train-only throughput (higher is better).
VK-FTGSPP reaches **2.33× the average reported step rate** across these six captures.

| Capture | FTGSPP (steps/s) | VK-FTGSPP (steps/s) | Ratio |
|---|---:|---:|---:|
| dance1 | 44.1 | 138.6 | 3.14× |
| dance2 | 61.2 | 143.7 | 2.35× |
| corgi1 | 43.1 | 131.6 | 3.05× |
| corgi2 | 45.1 | 69.2 | 1.54× |
| bike1 | 47.4 | 97.7 | 2.06× |
| bike2 | 54.1 | 106.0 | 1.96× |
| **Average** | **49.2** | **114.5** | **2.33×** |

Evaluation and hard separation are excluded. Relocation is included only in
FTGSPP's timings; these ratios are not end-to-end speedups.
[Measurement record](../dev/docs/performance-measurements.md).

To reconstruct your own capture, follow the [first-run guide](first-run.md).
