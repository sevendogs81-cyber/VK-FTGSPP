"""Compare one dense/sparse pair update, including two-time regularizers."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np

from ftgspp.training.backend import _shader_dir
from smoke_sync_multi_gpu import _build_rank, _make_gaussians


def update(devices, codec, *, hard=False):
    natives = []
    background = (0.23, 0.47, 0.71)
    try:
        for device in devices:
            native, *_ = _build_rank(_shader_dir(), device)
            natives.append(native)
            values = list(_make_gaussians())
            values[3][:] = 1.0  # Both points alive.
            values[8][:, 0] = [0.02, -0.015]  # Nonzero gate derivatives.
            native.set_dynamic_gaussians(*values)
            native.configure_ftgspp({"scale_reg": 0.002, "opacity_reg": 0.003,
                                     "gate_reg": 0.001})
        if hard:
            for native in natives:
                native.begin_hard_separation()
        natives[0].create_pair_exchange_gpu_merge_owner(2, 8 * 1024 * 1024, True)
        natives[1].join_pair_session(natives[0])
        for native in natives:
            native.configure_pair_exchange_gpu_merge(codec)
        timestamps = [0.5 + 0.01 * device for device in devices]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(
                lambda n: n.backward_dynamic_view(
                    0, 0, False, background,
                ),
                natives,
            ))
            exchanges = [pool.submit(n.exchange_pair_gradient, rank, 1)
                         for rank, n in enumerate(natives)]
            for future in exchanges:
                assert not future.result().get("aborted", False)
            list(pool.map(lambda n: n.optimizer_step_pair_exchange_gpu_merge(0, *timestamps), natives))
        return {**{f"model.{k}": np.array(v) for k, v in natives[0].get_dynamic_gaussians().items()},
                **{f"adam.{k}": np.array(v) for k, v in natives[0].get_optimizer_state().items()}}
    finally:
        for native in natives:
            native.reset_pair_exchange_gpu_merge_attachments()
        if natives:
            natives[0].release_pair_exchange_gpu_merge_owner()
        for native in natives:
            native.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs=2, type=int, default=[0, 2])
    devices = parser.parse_args().devices
    for hard in (False, True):
        dense = update(devices, "dense", hard=hard)
        for codec in ("auto_exact", "sorted_sparse"):
            actual = update(devices, codec, hard=hard)
            for name, value in dense.items():
                np.testing.assert_allclose(actual[name], value, rtol=1e-6, atol=1e-7,
                                           err_msg=f"{codec}, hard={hard}: {name}")
    print("Dense/sparse pair Adam and two-timestamp regularizer equivalence passed")


if __name__ == "__main__":
    main()
