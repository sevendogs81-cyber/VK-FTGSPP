#!/usr/bin/env python3
"""Compare composite pair overflow replay with an uninterrupted control.

The fixture is data-free and runs two dense pair updates.  Rank 0 injects a
tile overflow before update two; the globally aborted generation must replay
with the exact same non-black background and match every model/Adam array from
the no-overflow run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ftgspp.training.backend import _shader_dir
from smoke_sync_multi_gpu import _build_rank


BACKGROUNDS = (
    (0.23, 0.47, 0.71),
    (0.61, 0.29, 0.13),
)


def _state(native) -> dict[str, np.ndarray]:
    return {
        **{
            f"model.{name}": np.asarray(value).copy()
            for name, value in native.get_dynamic_gaussians().items()
        },
        **{
            f"optimizer.{name}": np.asarray(value).copy()
            for name, value in native.get_optimizer_state().items()
        },
    }


def _run(devices: tuple[int, int], *, inject_overflow: bool):
    natives = []
    owner = False
    retry_count = 0
    try:
        for device in devices:
            native, *_ = _build_rank(_shader_dir(), device)
            natives.append(native)
        natives[0].create_pair_exchange_gpu_merge_owner(
            2, 8 * 1024 * 1024, True,
        )
        owner = True
        natives[1].join_pair_session(natives[0])
        for native in natives:
            native.configure_pair_exchange_gpu_merge("dense")
        timestamps = tuple(0.5 + 0.01 * device for device in devices)
        for step, background in enumerate(BACKGROUNDS):
            if inject_overflow and step == 1:
                before_abort = _state(natives[0])
                natives[0].inject_sync_tile_overflow(65536)
            generation = step + 1
            while True:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    list(executor.map(
                        lambda native: native.backward_dynamic_view(
                            0, step, False, background,
                        ),
                        natives,
                    ))
                    futures = [
                        executor.submit(
                            native.exchange_pair_gradient, rank, generation,
                        )
                        for rank, native in enumerate(natives)
                    ]
                    reports = [future.result() for future in futures]
                if not any(bool(report.get("aborted")) for report in reports):
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        list(executor.map(
                            lambda native: native.optimizer_step_pair_exchange_gpu_merge(
                                step, *timestamps,
                            ),
                            natives,
                        ))
                    break
                if not inject_overflow or step != 1 or retry_count:
                    raise AssertionError(
                        f"unexpected or repeated pair abort: {reports}"
                    )
                retry_count += 1
                exact = max(
                    int(report.get("max_exact_count", 0)) for report in reports
                )
                for native in natives:
                    native.apply_sync_tile_capacity(exact)
                rolled_back = _state(natives[0])
                for name, value in before_abort.items():
                    np.testing.assert_array_equal(
                        rolled_back[name], value,
                        err_msg=f"aborted generation was not rolled back at {name}",
                    )
                natives[0].clear_sync_collectives_for_replay()
        rank_states = [_state(native) for native in natives]
        for name in rank_states[0]:
            # This visibility accumulator stays rank-local until the explicit
            # relocation collective; it is not optimizer state consumed by
            # the pair Adam update.
            if name == "optimizer.relocation_stats":
                continue
            np.testing.assert_array_equal(
                rank_states[0][name], rank_states[1][name],
                err_msg=f"replicas diverged at {name}",
            )
        return rank_states[0], retry_count
    finally:
        for native in natives:
            native.reset_pair_exchange_gpu_merge_attachments()
        if owner:
            natives[0].release_pair_exchange_gpu_merge_owner()
        for native in natives:
            native.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs=2, type=int, default=[0, 2])
    args = parser.parse_args()
    devices = (int(args.devices[0]), int(args.devices[1]))
    control, control_retries = _run(devices, inject_overflow=False)
    replay, replay_retries = _run(devices, inject_overflow=True)
    if control_retries != 0 or replay_retries != 1:
        raise AssertionError(
            f"unexpected retry counts: control={control_retries}, replay={replay_retries}"
        )
    for name, value in control.items():
        np.testing.assert_allclose(
            replay[name], value, rtol=2e-6, atol=2e-7,
            err_msg=f"overflow replay differs from no-overflow control at {name}",
        )
    print(
        "Composite non-black pair overflow replay matches control:",
        f"devices={list(devices)} arrays={len(control)} retries={replay_retries}",
    )


if __name__ == "__main__":
    main()
