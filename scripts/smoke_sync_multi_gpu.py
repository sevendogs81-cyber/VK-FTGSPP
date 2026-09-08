#!/usr/bin/env python3
"""Data-free dual-device smoke for production pair_exchange_gpu_merge.

Exercises startup bucket calibration, active-SH auto_exact, GPU peer mean,
replica Adam, color correction, relocation sum/broadcast and hard separation.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path[:0] = [str(_REPO_ROOT)]


def _stage(message: str) -> None:
    print(f"[sync-smoke] {message}", flush=True)


def _make_gaussians(n: int = 2):
    means = np.array([[-0.05, 0.0, 3.0], [0.05, 0.0, 3.0]], np.float32)[:n]
    log_scales = np.full((n, 3), -2.5, np.float32)
    quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (n, 1))
    opacity = np.array([[2.0], [-100.0]], np.float32)[:n]
    sh = np.zeros((n, 16, 3), np.float32)
    times = np.full((n, 1), 0.5, np.float32)
    durations = np.zeros((n, 1), np.float32)
    velocities = np.tile(np.array([[0.05, 0.0, 0.0]], np.float32), (n, 1))
    gates = np.full((n, 1), -100.0, np.float32)
    return means, log_scales, quats, opacity, sh, times, durations, velocities, gates


def _build_rank(shader_dir: Path, device: int):
    from _ftgspp_vk import FTGSPPVulkan

    n, height, width = 2, 32, 32
    means, log_scales, quats, opacity, sh, times, durations, velocities, gates = (
        _make_gaussians(n)
    )
    native = FTGSPPVulkan()
    native.initialize(str(shader_dir.resolve()) + "/", device)
    native.set_dynamic_gaussians(
        means, log_scales, quats, opacity, sh, times, durations, velocities, gates,
    )
    native.configure_ftgspp(
        {
            "max_steps": 4,
            "scene_scale": 1.0,
            "ssim_lambda": 0.2,
            "means_lr": 1.6e-4,
            "means_lr_final": 1.6e-6,
            "features_dc_lr": 2.5e-3,
            "features_rest_lr": 1.25e-4,
            "opacities_lr": 5e-2,
            "scales_lr": 5e-3,
            "quats_lr": 1e-3,
            "times_lr": 1.6e-4,
            "durations_lr": 5e-3,
            "velocities_lr": 1.6e-4,
            "velocities_lr_final": 1.6e-6,
            "gates_lr": 1e-3,
            "max_duration": -1.0,
            "marginal_threshold": 0.05,
        }
    )
    w2c = np.eye(4, dtype=np.float32)
    # Slightly different references so ranks produce distinct raw gradients.
    reference = np.zeros((height, width, 3), np.uint8)
    reference[..., 0] = 32 + device
    intrinsic = np.asarray(
        [[24.0, 0.0, width / 2], [0.0, 24.0, height / 2], [0.0, 0.0, 1.0]],
        np.float32,
    )
    native.clear_dynamic_views()
    native.add_dynamic_view(reference, w2c, intrinsic, 0.5, 0, 0, True, False)
    native.add_dynamic_view(reference, w2c, intrinsic, 0.55, 1, 1, False, False)
    native.configure_ftgspp(
        {
            "color_correction": True,
            "color_num_cameras": 2,
            "color_lr": 1e-3,
            "color_reg": 1e-3,
            "color_correction_start": -1,
        }
    )
    return native, n, height, width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0, 2],
        help="Vulkan physical-device indices (not CUDA ordinals). "
        "Default [0,2] skips llvmpipe on this host.",
    )
    parser.add_argument(
        "--shader-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "native" / "shader",
    )
    args = parser.parse_args()
    if len(args.devices) < 2:
        raise SystemExit("need at least two --devices for sync smoke")

    ranks = []
    owner_created = False
    try:
        for device in args.devices:
            ranks.append(_build_rank(args.shader_dir, device))
        _stage(f"initialized devices={args.devices}")

        natives = [r[0] for r in ranks]
        if len(natives) != 2:
            raise SystemExit(
                "production pair smoke requires exactly two --devices"
            )
        candidates = tuple(
            value * 1024 * 1024 for value in (8, 16, 32, 64)
        )
        _stage("calibrating pair-transfer buckets")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    native.benchmark_pair_transfer_buckets, candidates, 3,
                )
                for native in natives
            ]
            calibration = [future.result() for future in futures]
        _stage("pair-transfer bucket calibration complete")
        scores = {
            candidate: max(
                float(next(
                    item["wall_seconds"] for item in rank_values
                    if int(item["bucket_bytes"]) == candidate
                ))
                for rank_values in calibration
            )
            for candidate in candidates
        }
        selected_bucket = min(
            candidates, key=lambda value: (scores[value], value)
        )
        for candidate in candidates:
            print(
                f"[sync-smoke] calibrate {candidate // (1024 * 1024)} MiB "
                f"slower_wall={scores[candidate]*1e3:.2f}ms",
                flush=True,
            )
        probe = natives[0].create_pair_exchange_gpu_merge_owner(
            2, selected_bucket, True,
        )
        owner_created = True
        if not bool(probe["sparse"]):
            raise AssertionError(f"pair sparse owner unavailable: {probe}")
        sparse_bucket = int(probe.get("sparse_bucket_bytes") or 0)
        for native in natives:
            native.configure_pair_exchange_gpu_merge("auto_exact")
        _stage(
            f"configured auto_exact mailbox bucket_mib="
            f"{selected_bucket // (1024 * 1024)} "
            f"sparse_bucket_mib={sparse_bucket // (1024 * 1024)}"
        )

        step = 0
        with ThreadPoolExecutor(max_workers=2) as executor:
            _stage("running rank-local backward")
            list(executor.map(
                lambda native: native.backward_dynamic_view(0, step, False),
                natives,
            ))
            _stage("finalizing rank-local raw gradients")
            list(executor.map(
                lambda native: native.finalize_local_gradient(1), natives,
            ))
            _stage("exchanging pair gradients")
            futures = [
                executor.submit(
                    native.exchange_pair_gradient, rank, 1,
                )
                for rank, native in enumerate(natives)
            ]
            exchange_reports = [future.result() for future in futures]
            _stage("running replica optimizer steps")
            list(executor.map(
                lambda native: native.optimizer_step_pair_exchange_gpu_merge(
                    step, [0.5, 0.5]
                ),
                natives,
            ))
        _stage("pair optimizer step complete")
        codecs = {str(report["gaussian_codec"]) for report in exchange_reports}
        if len(codecs) != 1:
            raise AssertionError(f"pair codec decisions diverged: {codecs}")
        states = [
            {
                **{
                    f"model.{key}": np.asarray(value)
                    for key, value in native.get_dynamic_gaussians().items()
                },
                **{
                    f"optimizer.{key}": np.asarray(value)
                    for key, value in native.get_optimizer_state().items()
                },
            }
            for native in natives
        ]
        local_relocation = [
            state["optimizer.relocation_stats"].copy() for state in states
        ]
        for key in states[0]:
            # This accumulator remains rank-local until the explicit relocation
            # sum below; different references should produce different values.
            if key == "optimizer.relocation_stats":
                continue
            np.testing.assert_allclose(
                states[0][key], states[1][key], rtol=1e-6, atol=1e-7,
                err_msg=f"replica state diverged at {key}",
            )

        snapshot = {
            key: np.asarray(value).copy()
            for key, value in natives[0].get_dynamic_gaussians().items()
        }
        _stage("injecting rank0 tile overflow before generation 2")
        natives[0].inject_sync_tile_overflow(65536)
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(
                lambda native: native.backward_dynamic_view(0, 1, False),
                natives,
            ))
            list(executor.map(
                lambda native: native.finalize_local_gradient(1), natives,
            ))
            futures = [
                executor.submit(
                    native.exchange_pair_gradient, rank, 2,
                )
                for rank, native in enumerate(natives)
            ]
            overflow_reports = [future.result() for future in futures]
        if not all(bool(report.get("aborted")) for report in overflow_reports):
            raise AssertionError(
                f"forced overflow did not abort every rank: {overflow_reports}"
            )
        after_abort = natives[0].get_dynamic_gaussians()
        for key, before in snapshot.items():
            np.testing.assert_array_equal(
                np.asarray(after_abort[key]), before,
                err_msg=f"rank0 overflow still mutated {key} before replay",
            )
        after_abort_peer = natives[1].get_dynamic_gaussians()
        for key, before in snapshot.items():
            np.testing.assert_array_equal(
                np.asarray(after_abort_peer[key]), before,
                err_msg=f"rank1 Adam ran while rank0 overflowed on {key}",
            )
        _stage("replaying generation 2 after global capacity growth")
        for native in natives:
            native.apply_sync_tile_capacity(
                int(overflow_reports[0].get("max_exact_count", 65536)),
            )
        type(natives[0]).clear_sync_collectives_for_replay()
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(
                lambda native: native.backward_dynamic_view(0, 1, False),
                natives,
            ))
            list(executor.map(
                lambda native: native.finalize_local_gradient(1), natives,
            ))
            futures = [
                executor.submit(
                    native.exchange_pair_gradient, rank, 2,
                )
                for rank, native in enumerate(natives)
            ]
            replay_reports = [future.result() for future in futures]
            if any(bool(report.get("aborted")) for report in replay_reports):
                raise AssertionError(
                    f"overflow replay still aborted: {replay_reports}"
                )
            list(executor.map(
                lambda native: native.optimizer_step_pair_exchange_gpu_merge(
                    1, [0.5, 0.5]
                ),
                natives,
            ))
        replay_states = [
            {
                key: np.asarray(value)
                for key, value in native.get_dynamic_gaussians().items()
            }
            for native in natives
        ]
        for key in replay_states[0]:
            np.testing.assert_allclose(
                replay_states[0][key], replay_states[1][key],
                rtol=1e-6, atol=1e-7,
                err_msg=f"replica state diverged after overflow replay at {key}",
            )
        for native in natives:
            native.drain_for_structure_change()
        local_relocation = [
            np.asarray(native.relocation_stats_state()).copy()
            for native in natives
        ]

        # Relocation statistics are summed first; rank 0 plans once and every
        # replica applies exactly the same mapping/counts.
        with ThreadPoolExecutor(max_workers=2) as executor:
            _stage("summing relocation statistics")
            futures = [
                executor.submit(
                    native.exchange_pair_relocation, rank, 1,
                )
                for rank, native in enumerate(natives)
            ]
            for future in futures:
                future.result()
        _stage("relocation statistics sum complete")
        expected_relocation = np.sum(
            np.stack(local_relocation, axis=0), axis=0, dtype=np.float32,
        )
        for native in natives:
            np.testing.assert_allclose(
                np.asarray(native.relocation_stats_state()),
                expected_relocation,
                rtol=1e-6,
                atol=1e-7,
                err_msg="summed relocation statistics diverged",
            )
        plan = natives[0].plan_relocation_dynamic_native(1, False, 1234)
        if plan["all_dead"] or int(plan["alive"]) != 1:
            raise AssertionError(f"relocation plan failed: {plan}")
        plan_map = np.asarray(plan["mapping"], dtype=np.int32)
        plan_counts = np.asarray(plan["counts"], dtype=np.int32)
        natives[0].apply_planned_relocation("partial_copy")
        for native in natives[1:]:
            native.relocate_dynamic(
                plan_map, plan_counts, "partial_copy",
                require_dead_destinations=False,
                require_alive_donors=False,
            )

        color = natives[0].get_optimizer_state()
        if "color_params" not in color or np.asarray(color["color_params"]).size != 24:
            raise AssertionError("color correction state missing after sync step")

        # Exercise the destructive phase boundary once: every replica must
        # produce the same binary gate assignment and static velocities.
        _stage("running hard separation gate split")
        with ThreadPoolExecutor(max_workers=2) as executor:
            separation = list(executor.map(
                lambda native: native.begin_hard_separation(), natives,
            ))
        if any(dict(value) != dict(separation[0]) for value in separation[1:]):
            raise AssertionError(f"hard split counts diverged: {separation}")
        hard_states = [native.get_dynamic_gaussians() for native in natives]
        for state in hard_states:
            velocities_gates = np.asarray(state["velocities_gates"])
            gates = velocities_gates[:, 3]
            if not np.all(np.isin(gates, (-100.0, 100.0))):
                raise AssertionError("hard separation left non-binary gate values")
            if np.any(np.abs(velocities_gates[gates > 0.0, :3]) > 1e-7):
                raise AssertionError("hard separation left static velocities")
        np.testing.assert_array_equal(
            np.asarray(hard_states[0]["velocities_gates"]),
            np.asarray(hard_states[1]["velocities_gates"]),
        )

        print(
            "FTGSPP Vulkan pair sync smoke passed:",
            f"devices={args.devices}",
            f"bucket_mib={selected_bucket // (1024 * 1024)}",
            f"codec={next(iter(codecs))}",
            f"relocated={int(plan['relocated'])}",
            "transport=pair_exchange_gpu_merge",
        )
    finally:
        if owner_created:
            for native, *_ in ranks:
                native.reset_pair_exchange_gpu_merge_attachments()
            type(ranks[0][0]).release_pair_exchange_gpu_merge_owner()
        for native, *_ in ranks:
            native.cleanup()


if __name__ == "__main__":
    main()
