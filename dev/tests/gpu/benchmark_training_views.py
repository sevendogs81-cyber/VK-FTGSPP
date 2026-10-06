#!/usr/bin/env python3
"""Read-only real-scene B1/B4 execution benchmark, using an installed build.

Models and decoded references stay fixed; this is not a quality/convergence run.
Run baseline and candidate sequentially on the same idle device. Use --profile
separately: GPU stage queries and per-update telemetry perturb throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import numpy as np
from omegaconf import OmegaConf
from ftgspp.config.compose import resolve_config
from ftgspp.data.scene import Scene
from ftgspp.io.checkpoint import load_model
from ftgspp.training.backend import NativeVulkanBackend
from ftgspp.training.sampler import sample_batch_views


def rasterizer_choices(state: str) -> list[list[int]]:
    tokens = shlex.split(state)
    return [[int(v) for v in tokens[i:i+5]] for i in range(2, len(tokens), 7)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', nargs=2, action='append', metavar=('CONFIG', 'MODEL'), required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--exposures', type=int, default=128)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--rasterizer-states', type=Path,
                        help='save baseline frozen schedules, or reuse an existing schedule file')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.exposures <= 0 or args.exposures % 32 or args.rounds <= 0:
        parser.error('exposures must be a positive multiple of 32; rounds must be positive')
    os.environ['FTGSPP_VK_STAGE_TIMERS'] = '1' if args.profile else '0'
    backend = NativeVulkanBackend(args.device)
    native = backend.native
    result = {'device': native.device_info(), 'profile_enabled': args.profile,
              'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'protocol': 'fixed trained model; 32 decoded real views; SH3; no relocation/eval/logging',
              'exposures_per_round': args.exposures, 'cases': []}
    schedules = (json.loads(args.rasterizer_states.read_text())
                 if args.rasterizer_states and args.rasterizer_states.exists() else {})
    reuse_schedules = bool(schedules)
    try:
        for config_name, model_name in args.case:
            config_path, model_path = Path(config_name).resolve(), Path(model_name).resolve()
            config = resolve_config(OmegaConf.load(config_path)).config
            if config.train.color_correction or config.train.hard_separation:
                raise ValueError('benchmark requires ordinary foreground without color correction')
            dataset = Scene.from_config(config)
            model = load_model(model_path)
            backend.configure(config, 1.)
            # Positive tiny position LR avoids 0/0 in the exponential scheduler.
            native.configure_ftgspp(dict(means_lr=1e-30, means_lr_final=1e-30,
                velocities_lr=0., velocities_lr_final=0., scales_lr=0., quats_lr=0.,
                opacities_lr=0., features_dc_lr=0., features_rest_lr=0.,
                times_lr=0., durations_lr=0., gates_lr=0.))
            backend.upload_model(model)
            backend.upload_dataset(dataset, rgb_cache=64, workers=0)
            indices = tuple(i for s in range(8) for i in sample_batch_views(
                seed=31, step=s, batch_size=4, num_views=len(dataset.train_views)))
            backend.prefetch_train_views(indices)
            backgrounds = np.random.default_rng(82).uniform(.1, .9, (32, 3)).astype(np.float32)
            step = 4000

            def run(batch, exposures, collect=False):
                nonlocal step
                telemetry = {}
                for start in range(0, exposures, batch):
                    slot = start % 32
                    if batch == 1:
                        backend.train_view(indices[slot], step, collect_loss=False, background=backgrounds[slot])
                    else:
                        backend.train_batch(indices[slot:slot+4], step, collect_loss=False,
                                            backgrounds=backgrounds[slot:slot+4])
                    if collect:
                        for key, value in native.get_pipeline_telemetry().items():
                            if key.startswith('cpu_') or key in ('fence_waits_per_step', 'submits_per_step', 'tile_replay'):
                                telemetry[key] = telemetry.get(key, 0.) + value
                    step += 1
                native.drain_for_structure_change()
                return telemetry

            case = {'name': config.data.name, 'config': str(config_path), 'model': str(model_path),
                    'splats': len(model), 'views': [
                        {'index': i, 'frame': dataset.train_views[i].frame,
                         'camera': dataset.train_views[i].camera_id, 'size': list(dataset.train_views[i].size)}
                        for i in indices], 'modes': {'1': [], '4': []}}
            if args.rasterizer_states:
                identity = {'device_uuid': result['device']['device_uuid'], 'model': str(model_path),
                            'splats': len(model), 'views': case['views']}
                if reuse_schedules:
                    saved = schedules[config.data.name]
                    if saved['identity'] != identity:
                        raise ValueError('rasterizer schedule device/model/views mismatch')
                    frozen = saved['state']
                else:
                    native.restore_rasterizer_state('1 0')
                    for _ in range(8):
                        for batch in (1, 4):
                            native.configure_ftgspp(dict(local_batch_size=batch))
                            run(batch, 128)
                        frozen = native.rasterizer_state()
                        entries = rasterizer_choices(frozen)
                        if entries and all(0 <= row[4] < 5 for row in entries):
                            break
                    else:
                        raise RuntimeError('rasterizer did not freeze after warmup')
                    schedules[config.data.name] = {'identity': identity, 'state': frozen}
                case['rasterizer_state_sha256'] = hashlib.sha256(frozen.encode()).hexdigest()
                case['rasterizer_choices'] = rasterizer_choices(frozen)
            for repeat in range(args.rounds):
                for batch in ((1, 4) if repeat % 2 == 0 else (4, 1)):
                    native.configure_ftgspp(dict(local_batch_size=batch))
                    if args.rasterizer_states:
                        native.restore_rasterizer_state(frozen)
                    run(batch, 32)
                    before = native.get_timing_breakdown()
                    start = time.perf_counter()
                    telemetry = run(batch, args.exposures, args.profile)
                    elapsed = time.perf_counter() - start
                    after = native.get_timing_breakdown()
                    if args.rasterizer_states:
                        if rasterizer_choices(native.rasterizer_state()) != case['rasterizer_choices']:
                            raise RuntimeError('rasterizer choices changed during measurement')
                    stages = {key: {'count': value[0] - before.get(key, (0, 0.))[0],
                                    'seconds': value[1] - before.get(key, (0, 0.))[1]}
                              for key, value in after.items()}
                    np.testing.assert_array_equal(np.asarray(native.base_xyz_ws), model.means)
                    case['modes'][str(batch)].append({'seconds': elapsed,
                        'views_per_second': args.exposures / elapsed, 'stages': stages,
                        'telemetry': telemetry, 'vram_bytes': native.get_vram_usage()})
            result['cases'].append(case)
            print(config.data.name, {b: float(np.median([r['views_per_second'] for r in rows]))
                  for b, rows in case['modes'].items()}, flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.output.with_suffix('.tmp')
            tmp.write_text(json.dumps(result, indent=2) + '\n'); tmp.replace(args.output)
        if args.rasterizer_states and not reuse_schedules:
            args.rasterizer_states.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.rasterizer_states.with_suffix('.tmp')
            tmp.write_text(json.dumps(schedules, indent=2) + '\n')
            tmp.replace(args.rasterizer_states)
    finally:
        native.abort_batch(); native.cleanup()


if __name__ == '__main__':
    main()
