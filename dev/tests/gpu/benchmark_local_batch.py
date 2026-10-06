#!/usr/bin/env python3
"""Controlled single-device B1/B4 throughput at a fixed synthetic workload.

Updates are below FP32 resolution so geometry, visibility and tile counts do not drift.
This measures execution efficiency, not convergence or reconstruction quality.
Use separate processes/PYTHONPATH to compare installed builds; do not run them
concurrently on the same GPU. No model or rendered-image artifacts are saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import ftgspp._ftgspp_vk as extension
from ftgspp.training.backend import _shader_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--splats', type=int, default=1_000_000)
    parser.add_argument('--extent', type=int, default=768)
    parser.add_argument('--exposures', type=int, default=240)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.splats, args.extent, args.exposures, args.rounds) <= 0 or args.exposures % 4:
        parser.error('sizes must be positive and exposures divisible by four')
    rng = np.random.default_rng(901)
    n, extent = args.splats, args.extent
    means = rng.uniform((-1.2, -1.2, 2.8), (1.2, 1.2, 3.2), (n, 3)).astype(np.float32)
    logs = np.tile(np.array([-5.6, -5.9, -6.2], np.float32), (n, 1))
    rotations = np.tile(np.array([1., 0., 0., 0.], np.float32), (n, 1))
    opacity = np.full((n, 1), -2., np.float32)
    sh = rng.normal(0., .1, (n, 16, 3)).astype(np.float32)
    times = np.full((n, 1), .5, np.float32)
    duration = np.zeros((n, 1), np.float32)
    velocity = rng.normal(0., .005, (n, 3)).astype(np.float32)
    gate = np.full((n, 1), -.1, np.float32)
    images = [rng.integers(0, 256, (extent, extent, 4), dtype=np.uint8) for _ in range(4)]
    backgrounds = [tuple(float(v) for v in row) for row in rng.uniform(.1, .9, (4, 3))]
    native = extension.FTGSPPVulkan()
    native.initialize(str(_shader_dir()) + '/', args.device)
    try:
        native.configure_ftgspp(dict(
            max_steps=1_000_000, max_sh_degree=3, image_objective=1,
            max_duration=1., marginal_threshold=.05, scene_scale=1.,
            # The position scheduler evaluates final/start. A positive tiny
            # rate avoids 0/0 while leaving all fixture positions bit-identical.
            means_lr=1e-30, means_lr_final=1e-30, velocities_lr=0., velocities_lr_final=0.,
            scales_lr=0., quats_lr=0., opacities_lr=0., features_dc_lr=0.,
            features_rest_lr=0., times_lr=0., durations_lr=0., gates_lr=0.,
            scale_reg=.01, opacity_reg=.01, gate_reg=.001,
            color_correction=False, local_batch_size=1))
        native.set_dynamic_gaussians(means, logs, rotations, opacity, sh, times, duration, velocity, gate)
        intrinsics = np.array([[extent, 0, extent / 2], [0, extent, extent / 2], [0, 0, 1]], np.float32)
        for i in range(4):
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = (i - 1.5) * .03
            native.add_dynamic_view_meta(extent, extent, pose, intrinsics, .2 + .2 * i, i, i, True)

        step = 4000  # SH3 for both modes; outside all SH activation boundaries.
        def run(batch_size: int, exposures: int) -> None:
            nonlocal step
            for offset in range(0, exposures, batch_size):
                if batch_size == 1:
                    i = offset % 4
                    native.bind_dynamic_view_image(i, True, images[i])
                    native.train_dynamic_view(i, step, False, backgrounds[i])
                else:
                    for attempt in range(5):
                        native.begin_batch(step, False)
                        for i in range(4):
                            native.bind_dynamic_view_image(i, True, images[i])
                            if not native.accumulate_batch_view(i, backgrounds[i]):
                                break
                        else:
                            native.commit_batch()
                            break
                    else:
                        raise RuntimeError('persistent tile overflow')
                step += 1
            native.drain_for_structure_change()

        samples = {1: [], 4: []}
        allocations = {}
        for round_index in range(args.rounds):
            # Alternate order to reduce thermal/order bias. First B1 allocation
            # precedes all raw-gradient allocation and is the clean B1 baseline.
            for batch_size in ((1, 4) if round_index % 2 == 0 else (4, 1)):
                native.configure_ftgspp(dict(local_batch_size=batch_size))
                run(batch_size, 32)
                if batch_size not in allocations:
                    allocations[batch_size] = dict(native.get_vram_breakdown())
                start = time.perf_counter()
                run(batch_size, args.exposures)
                samples[batch_size].append(time.perf_counter() - start)
                np.testing.assert_array_equal(np.asarray(native.base_xyz_ws), means)
        result = {
            'workload': 'fixed synthetic foreground; sub-FP32 updates; SH3; no eval/logging/relocation',
            'device': native.device_info(),
            'extension_sha256': hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
            'splats': n, 'extent': extent, 'exposures_per_round': args.exposures,
            'includes': 'image binding/upload, forward/loss/backward, Adam, GPU completion',
            'modes': {str(b): {
                'seconds': values,
                'median_views_per_second': args.exposures / float(np.median(values)),
                'first_warm_vram_bytes': sum(allocations[b].values()),
                'allocations': allocations[b],
            } for b, values in samples.items()},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(args.output)
        print(json.dumps({b: v['median_views_per_second'] for b, v in result['modes'].items()}), flush=True)
    finally:
        native.abort_batch()
        native.cleanup()


if __name__ == '__main__':
    main()
