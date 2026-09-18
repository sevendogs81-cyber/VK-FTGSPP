from __future__ import annotations

import unittest
import threading

import numpy as np

from ftgspp.config.schema import AppConfig, DistributedConfig
from ftgspp.config.validate import DISTRIBUTED_SAMPLING_POLICIES
from ftgspp.training.sampler import (
    sample_rank_views,
    sampling_phase_seed,
    _flat_epoch_permutation,
    _torch_random_sampler_stream,
)
from ftgspp.training.distributed import (
    DistributedPlan,
    SynchronousBackendGroup,
)


class _FakeModel:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def arrays(self) -> dict[str, np.ndarray]:
        return {"value": self.value.copy()}


class _FakeBackend:
    def __init__(self) -> None:
        self.parameter = np.zeros(2, dtype=np.float32)
        self.gradient = np.zeros(2, dtype=np.float32)
        self.moment = np.zeros(2, dtype=np.float32)
        self.stats = np.zeros((2, 2), dtype=np.float32)
        self.loss = {
            "l1": 0.0, "ssim": 0.0, "reg_opacity": 0.0,
            "reg_scale": 0.0, "gate": 0.0, "color_cc": 0.0,
            "image_loss": 0.0,
            "total_loss": 0.0,
        }
        self.plan_calls = 0
        self.apply_planned_calls: list[str] = []
        self.relocation_calls: list[
            tuple[np.ndarray, np.ndarray, str, dict[str, object]]
        ] = []
        self.loss_reads = 0
        self.closed = False
        self.apply_sync_capacity_calls: list[int] = []
        self.drain_structure_calls = 0
        self._collect_geometry_grad_stats = False
        self._geometry_grad_stats = {
            "grad_xyz_rms": 0.01,
            "grad_scaling_rms": 0.02,
            "grad_oc_rms": 0.03,
            "grad_rotation_rms": 0.04,
            "grad_sh_dc_rms": 0.05,
            "grad_sh_rest_rms": 0.06,
            "grad_xyz_l2": 1.0,
            "grad_scaling_l2": 2.0,
            "grad_oc_l2": 3.0,
            "grad_rotation_l2": 4.0,
            "grad_sh_dc_l2": 5.0,
            "grad_sh_rest_l2": 6.0,
            "grad_num_splats": 2.0,
        }

    def prefetch_train_views(self, view_indices, *, executor=None) -> None:
        self.prefetched_views = tuple(int(value) for value in view_indices)
        self.prefetch_executor = executor

    def schedule_train_views(self, view_indices, *, executor) -> None:
        self.scheduled_views = tuple(int(value) for value in view_indices)
        self.schedule_executor = executor

    def backward_view(
        self, view_index: int, step: int, *, collect_loss: bool = True,
    ) -> None:
        del step
        value = np.float32(view_index + 1)
        self.gradient[:] = (value, value * 2.0)
        self.stats[:, 0] += value
        self.stats[:, 1] += 1.0
        l1, ssim = float(value), float(value * 0.25)
        terms = {
            "l1": l1, "ssim": ssim, "reg_opacity": float(value * 0.1),
            "reg_scale": 0.5, "gate": 0.2, "color_cc": 0.3,
        }
        terms["image_loss"] = l1 + ssim
        terms["total_loss"] = sum(
            terms[key] for key in (
                "l1", "ssim", "reg_opacity", "reg_scale", "gate",
                "color_cc",
            )
        )
        if collect_loss:
            self.loss = terms

    def apply_sync_tile_capacity(self, exact_count: int) -> None:
        self.apply_sync_capacity_calls.append(int(exact_count))
        self.gradient.fill(0.0)

    def clear_sync_collectives_for_replay(self) -> None:
        self.clear_sync_calls = getattr(self, "clear_sync_calls", 0) + 1

    def drain_for_structure_change(self) -> None:
        self.drain_structure_calls += 1

    def inject_sync_tile_overflow(self, exact_count: int) -> None:
        self.injected_overflow = int(exact_count)

    def optimizer_step(self, step: int) -> None:
        del step
        self.moment = self.gradient.copy()
        self.parameter -= self.gradient
        self.gradient.fill(0.0)

    def set_collect_geometry_grad_stats(self, enabled: bool) -> None:
        self._collect_geometry_grad_stats = bool(enabled)

    def geometry_gradient_stats(self):
        if not self._collect_geometry_grad_stats:
            raise RuntimeError("geometry gradient stats were not collected")
        return dict(self._geometry_grad_stats)

    def last_loss(self):
        self.loss_reads += 1
        return dict(self.loss)

    def relocation_stats_state(self) -> np.ndarray:
        return self.stats.copy()

    def set_relocation_stats_state(self, values: np.ndarray) -> None:
        self.stats = np.array(values, dtype=np.float32, copy=True)

    def clear_relocation_stats_state(self) -> None:
        self.stats.fill(0.0)

    def apply_planned_relocation(self, mode: str) -> None:
        self.apply_planned_calls.append(str(mode))
        self.stats.fill(0.0)

    def plan_relocation_native(self, step: int, include_gate: bool, seed: int):
        del step, include_gate, seed
        self.plan_calls += 1
        self.assert_global_stats = self.stats.copy()
        return {
            "mapping": np.array([-1, 0], dtype=np.int32),
            "counts": np.array([1, 0], dtype=np.int32),
            "relocated": 1,
            "alive": 1,
            "all_dead": False,
        }

    def relocate(
        self, mapping: np.ndarray, counts: np.ndarray, mode: str, **kwargs,
    ) -> None:
        self.relocation_calls.append(
            (mapping.copy(), counts.copy(), mode, dict(kwargs))
        )
        self.stats.fill(0.0)

    def relocate_native(self, step, mode, include_gate, seed):
        raise AssertionError("distributed relocation must plan once and broadcast")

    def model(self) -> _FakeModel:
        return _FakeModel(self.parameter)

    def optimizer_state(self):
        return {
            "moment": self.moment.copy(),
            "relocation_stats": self.stats.copy(),
        }

    def close(self) -> None:
        self.closed = True


class _PairExchangeFakeBackend(_FakeBackend):
    def __init__(self, rank):
        super().__init__()
        self.rank = int(rank)

    _condition = threading.Condition()
    _gradients: dict[int, dict[int, np.ndarray]] = {}
    _statistics: dict[int, dict[int, np.ndarray]] = {}
    _world_size = 2

    def supports_pair_exchange_gpu_merge(self) -> bool:
        return True

    def benchmark_pair_transfer_buckets(
        self, candidate_bytes, *, rounds: int = 3,
    ):
        base = {8: 0.50, 16: 0.30, 32: 0.20, 64: 0.26}
        return [
            {
                "bucket_bytes": int(value),
                "payload_bytes": 64 * 1024 * 1024,
                "rounds": int(rounds),
                "wall_seconds": base[int(value) // (1024 * 1024)]
                + 0.01 * self.rank,
                "download_seconds": 0.1,
                "upload_seconds": 0.1,
                "persistent_mapping": True,
            }
            for value in candidate_bytes
        ]

    def create_pair_exchange_gpu_merge_owner(self, **kwargs):
        type(self)._world_size = int(kwargs["world_size"])
        with self._condition:
            type(self)._gradients = {}
            type(self)._statistics = {}
        return {
            "abi": "ftgspp-host-mailbox-v3",
            "cpu_reduction": False,
            "async_pipeline": True,
            "borrowed_readback": True,
            "sparse": bool(
                kwargs.get("sparse", False) and type(self)._world_size == 2
            ),
        }

    def join_pair_session(self, leader) -> None:
        self.session_leader = leader

    def abort_pair_session(self, reason) -> None:
        with self._condition:
            self._condition.notify_all()

    def configure_pair_exchange_gpu_merge(
        self, *, sparse: bool, codec_policy: str = "auto_exact",
    ) -> None:
        self.collective_sparse = bool(sparse)
        self.collective_codec_policy = str(codec_policy)

    def exchange_pair_gradient(self, *, rank: int, generation: int):
        with self._condition:
            exchange = type(self)._gradients.setdefault(generation, {})
            exchange[rank] = self.gradient.copy()
            self._condition.notify_all()
            self._condition.wait_for(
                lambda: len(type(self)._gradients.get(generation, {}))
                == type(self)._world_size
            )
            self.peer_gradients = [
                type(self)._gradients[generation][source_rank].copy()
                for source_rank in range(type(self)._world_size)
            ]
        return {
            "download_seconds": 0.1,
            "upload_seconds": 0.1,
            "host_memcpy_seconds": 0.02,
            "sparse_pack_seconds": 0.01,
            "peer_wait_seconds": 0.03,
            "transfer_wall_seconds": 0.12,
            "wall_seconds": 0.15,
            "async_pipeline": True,
            "vulkan_persistent_mapping": True,
            "aborted": False,
        }

    def optimizer_step_pair_exchange_gpu_merge(
        self, step: int, rank0_timestamp: float, rank1_timestamp: float,
    ):
        self.assert_timestamp_count = 2
        self.gradient = np.mean(
            np.stack(self.peer_gradients), axis=0,
            dtype=np.float32,
        )
        self.optimizer_step(step)
        return {"gpu_reduction_seconds": 0.0, "optimizer_seconds": 0.0}

    def exchange_pair_relocation(self, *, rank: int, generation: int):
        with self._condition:
            exchange = type(self)._statistics.setdefault(generation, {})
            exchange[rank] = self.stats.copy()
            self._condition.notify_all()
            self._condition.wait_for(
                lambda: len(type(self)._statistics.get(generation, {}))
                == type(self)._world_size
            )
            self.stats = np.sum(
                np.stack([
                    type(self)._statistics[generation][source_rank]
                    for source_rank in range(type(self)._world_size)
                ]),
                axis=0,
                dtype=np.float32,
            )
        return {"cpu_reduction_seconds": 0.0}

    def train_view_timestamp(self, view_index: int) -> float:
        return float(view_index) * 0.1

    def reset_pair_exchange_gpu_merge_attachments(self) -> None:
        pass

    def release_pair_exchange_gpu_merge_owner(self) -> None:
        pass


class _OverflowOncePairExchangeFakeBackend(_PairExchangeFakeBackend):
    def __init__(self, rank: int) -> None:
        super().__init__(rank)
        self.exchange_calls = 0
        self.optimizer_calls = 0

    def exchange_pair_gradient(self, *, rank: int, generation: int):
        self.exchange_calls += 1
        if self.exchange_calls == 1:
            return {
                "aborted": True,
                "fatal": False,
                "overflow": True,
                "max_exact_count": 65536,
                "local_overflow": True,
                "download_seconds": 0.0,
                "upload_seconds": 0.0,
                "host_memcpy_seconds": 0.0,
                "peer_wait_seconds": 0.0,
                "transfer_wall_seconds": 0.0,
                "wall_seconds": 0.0,
                "async_pipeline": False,
                "vulkan_persistent_mapping": True,
            }
        return super().exchange_pair_gradient(rank=rank, generation=generation)

    def optimizer_step_pair_exchange_gpu_merge(
        self, step: int, rank0_timestamp: float, rank1_timestamp: float,
    ):
        self.optimizer_calls += 1
        return super().optimizer_step_pair_exchange_gpu_merge(step, rank0_timestamp, rank1_timestamp)


class _StalePairExchangeFakeBackend(_PairExchangeFakeBackend):
    def create_pair_exchange_gpu_merge_owner(self, **kwargs):
        report = dict(super().create_pair_exchange_gpu_merge_owner(**kwargs))
        report["abi"] = "ftgspp-host-mailbox-v1"
        return report


class DistributedRuntimeTests(unittest.TestCase):
    def test_phase_boundaries_recreate_session_and_preserve_timeout(self):
        from unittest.mock import Mock
        replicas = [_PairExchangeFakeBackend(0), _PairExchangeFakeBackend(1)]
        for replica in replicas:
            replica.begin_hard_separation = lambda: {"marginalized": 1, "kept": 1}
            replica.resume_hard_separation = replica.begin_hard_separation
        with SynchronousBackendGroup(replicas, collective_timeout_seconds=0.25) as group:
            create = replicas[0].create_pair_exchange_gpu_merge_owner
            replicas[0].create_pair_exchange_gpu_merge_owner = Mock(wraps=create)
            group.begin_hard_separation()
            group.resume_hard_separation()
            calls = replicas[0].create_pair_exchange_gpu_merge_owner.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(call.kwargs["timeout_seconds"] == 0.25 for call in calls))

    def test_main_phase_seed_is_the_train_seed(self):
        seed, train_iterations = 17, 30_000
        main_seed = sampling_phase_seed(
            seed=seed, phase="main", train_iterations=train_iterations,
        )
        self.assertEqual(main_seed, seed)
        with self.assertRaisesRegex(ValueError, "sampling phase"):
            sampling_phase_seed(seed=seed, phase="unknown",
                                train_iterations=train_iterations)
        self.assertEqual(
            sampling_phase_seed(
                seed=seed, phase="hard_separation",
                train_iterations=train_iterations,
            ),
            seed + train_iterations,
        )

    def test_flat_epoch_covers_the_complete_global_flat_view_stream(self):
        seed, world_size, num_views = 31, 2, 5
        actual = [
            view
            for step in range(7)
            for view in sample_rank_views(
                seed=seed, step=step, world_size=world_size,
                num_views=num_views, sampling_policy="flat_epoch",
            )
        ]
        expected = np.concatenate([
            np.random.Generator(np.random.PCG64(np.random.SeedSequence([
                seed, epoch, 0x4645504D,
            ]))).permutation(num_views)
            for epoch in range(3)
        ]).tolist()[:len(actual)]
        self.assertEqual(actual, expected)
        self.assertCountEqual(actual[:num_views], range(num_views))
        self.assertCountEqual(actual[num_views:2 * num_views], range(num_views))

        # Step 2 straddles epochs (global slots 4 and 5), and a resumed call
        # reconstructs the same boundary directly without mutable RNG state.
        boundary = sample_rank_views(
            seed=seed, step=2, world_size=world_size,
            num_views=num_views, sampling_policy="flat_epoch",
        )
        self.assertEqual(boundary, tuple(expected[4:6]))
        self.assertEqual(
            boundary,
            sample_rank_views(
                seed=seed, step=2, world_size=world_size,
                num_views=num_views, sampling_policy="flat_epoch",
            ),
        )

    def test_single_view_sampler_returns_one_rank_slot(self):
        actual = sample_rank_views(
            seed=31, step=2, world_size=2,
            num_views=11, sampling_policy="flat_epoch",
        )
        stream = np.concatenate([
            _flat_epoch_permutation(31, epoch, 11) for epoch in range(2)
        ])
        self.assertEqual(actual, tuple(int(value) for value in stream[4:6]))
        self.assertEqual(len(actual), 2)

    def test_flat_epoch_permutation_is_cached_and_immutable(self):
        first = _flat_epoch_permutation(7, 3, 11)
        second = _flat_epoch_permutation(7, 3, 11)
        self.assertIs(first, second)
        self.assertFalse(first.flags.writeable)
        self.assertFalse(np.array_equal(
            _flat_epoch_permutation(7, 1, 101),
            _flat_epoch_permutation(8, 0, 101),
        ))

    def test_flat_epoch_is_the_sync_schema_default(self):
        self.assertEqual(DistributedConfig().sampling_policy, "flat_epoch")
        self.assertIn("flat_epoch", DISTRIBUTED_SAMPLING_POLICIES)

    def test_distributed_plan_resolves_auto_only_to_pair_host(self):
        config = AppConfig()
        config.distributed.mode = "sync"
        config.distributed.backend = "auto"
        config.distributed.device_ids = [0, 1]
        plan = DistributedPlan.from_config(config)
        self.assertEqual(plan.backend, "pair_host")
        self.assertEqual(
            plan.collective_algorithm, "pair_exchange_gpu_merge",
        )

    def test_view_sampling_is_step_addressable_and_rank_distinct(self):
        first = sample_rank_views(
            seed=17, step=23, world_size=4, num_views=20,
            sampling_policy="unique",
        )
        second = sample_rank_views(
            seed=17, step=23, world_size=4, num_views=20,
            sampling_policy="unique",
        )
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 4)
        expected_single = int(np.random.default_rng(17 + 23).integers(20))
        self.assertEqual(
            sample_rank_views(
                seed=17, step=23, world_size=1, num_views=20,
                sampling_policy="unique",
            ),
            (expected_single,),
        )

    def test_with_replacement_policy_matches_keyed_generator(self):
        actual = sample_rank_views(
            seed=17, step=23, world_size=8, num_views=20,
            sampling_policy="with_replacement",
        )
        entropy = [17, 23, 0x46544753]
        generator = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(entropy))
        )
        expected = tuple(int(value) for value in generator.choice(
            20, size=8, replace=True,
        ))
        self.assertEqual(actual, expected)

    def test_torch_random_sampler_matches_ftgspp_reference_stream(self):
        expected = [
            3644, 2439, 5033, 2760, 2563, 3579, 727, 703,
            4897, 5283, 2501, 4466,
        ]
        stream = _torch_random_sampler_stream(0, 6300, 1102)
        self.assertEqual([int(value) for value in stream[:12]], expected)
        self.assertEqual(
            [int(stream[index]) for index in [
                50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000,
            ]],
            [1840, 2705, 5772, 1369, 135, 81, 3789, 6194, 6212, 4567, 1795],
        )
        self.assertFalse(stream.flags.writeable)
        self.assertIs(stream, _torch_random_sampler_stream(0, 6300, 1102))
        self.assertEqual(
            sample_rank_views(
                seed=0, step=2, world_size=2, num_views=6300,
                sampling_policy="torch_random_sampler", sample_count=10,
            ),
            (2563, 3579),
        )

    def test_torch_random_sampler_requires_phase_sample_count(self):
        with self.assertRaisesRegex(ValueError, "sample_count"):
            sample_rank_views(
                seed=0, step=0, world_size=1, num_views=20,
                sampling_policy="torch_random_sampler",
            )
        with self.assertRaisesRegex(ValueError, "outside sample_count"):
            sample_rank_views(
                seed=0, step=10, world_size=1, num_views=20,
                sampling_policy="torch_random_sampler", sample_count=10,
            )

    def test_single_device_step_local_policies_keep_legacy_keyed_history(self):
        expected = (int(np.random.default_rng(40).integers(20)),)
        for policy in ("unique", "with_replacement"):
            self.assertEqual(
                sample_rank_views(
                    seed=17, step=23, world_size=1, num_views=20,
                    sampling_policy=policy,
                ),
                expected,
            )

    def test_single_device_flat_epoch_covers_every_view_before_repeating(self):
        seed, num_views = 17, 20
        actual = [
            sample_rank_views(
                seed=seed, step=step, world_size=1, num_views=num_views,
                sampling_policy="flat_epoch",
            )[0]
            for step in range(2 * num_views)
        ]

        self.assertCountEqual(actual[:num_views], range(num_views))
        self.assertCountEqual(actual[num_views:], range(num_views))
        self.assertEqual(
            actual[:num_views],
            [int(value) for value in _flat_epoch_permutation(seed, 0, num_views)],
        )

    def test_flat_epoch_keeps_every_prefix_globally_balanced(self):
        for world_size, steps, num_views in (
            (1, 53, 17),
            (2, 29, 17),
        ):
            with self.subTest(
                world_size=world_size,
            ):
                counts = np.zeros(num_views, dtype=np.int64)
                for step in range(steps):
                    selected = sample_rank_views(
                        seed=31, step=step, world_size=world_size,
                        num_views=num_views,
                        sampling_policy="flat_epoch",
                    )
                    np.add.at(
                        counts, np.asarray(selected, dtype=np.int64), 1,
                    )
                    self.assertLessEqual(
                        int(counts.max() - counts.min()), 1,
                    )

                self.assertTrue(np.all(counts > 0))

    def test_single_device_co_frame_uses_its_metadata_contract(self):
        with self.assertRaisesRegex(ValueError, "co_frame sampling requires"):
            sample_rank_views(
                seed=17, step=23, world_size=1, num_views=20,
                sampling_policy="co_frame",
            )

    def test_unknown_sampling_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sampling_policy"):
            sample_rank_views(
                seed=1, step=2, world_size=2, num_views=3,
                sampling_policy="invalid",
            )

    def test_sync_step_averages_raw_gradients_then_updates_all_replicas(self):
        replicas = [_PairExchangeFakeBackend(0), _PairExchangeFakeBackend(1)]
        with SynchronousBackendGroup(replicas) as group:
            result = group.train_step((0, 2), step=5)
            # Raw gradients are [1,2] and [3,6], hence global mean [2,4].
            for replica in replicas:
                np.testing.assert_array_equal(replica.parameter, [-2.0, -4.0])
                np.testing.assert_array_equal(replica.moment, [2.0, 4.0])
            self.assertAlmostEqual(result["loss"]["l1"], 2.0)
            self.assertAlmostEqual(result["loss"]["reg_opacity"], 0.2)
            self.assertAlmostEqual(
                result["loss"]["total_loss"],
                sum(result["loss"][key] for key in (
                    "l1", "ssim", "reg_opacity", "reg_scale", "gate",
                    "color_cc",
                )),
            )
            self.assertEqual(result["views"], (0, 2))
            self.assertIsNotNone(replicas[0].prefetch_executor)
        self.assertTrue(all(replica.closed for replica in replicas))

    def test_rank0_overflow_skips_adam_then_replays_same_views(self):
        replicas = [
            _OverflowOncePairExchangeFakeBackend(0),
            _OverflowOncePairExchangeFakeBackend(1),
        ]
        with SynchronousBackendGroup(replicas) as group:
            result = group.train_step((0, 2), step=5)
            self.assertEqual(result["sync_abort_retries"], 1)
            for replica in replicas:
                self.assertEqual(replica.exchange_calls, 2)
                self.assertEqual(replica.optimizer_calls, 1)
                self.assertEqual(replica.apply_sync_capacity_calls, [65536])
                np.testing.assert_array_equal(replica.parameter, [-2.0, -4.0])
                np.testing.assert_array_equal(replica.moment, [2.0, 4.0])

    def test_zero_bucket_runs_pair_calibration_and_selects_slowest_rank_minimum(self):
        replicas = [_PairExchangeFakeBackend(0), _PairExchangeFakeBackend(1)]
        with SynchronousBackendGroup(
            replicas, collective_bucket_bytes=0,
        ) as group:
            self.assertEqual(
                group.collective_bucket_bytes, 32 * 1024 * 1024,
            )
            report = group.collective_bucket_benchmark
            self.assertTrue(report["executed"])
            self.assertEqual(
                report["selected_bucket_bytes"], 32 * 1024 * 1024,
            )
            self.assertEqual(len(report["rank_results"]), 2)

    def test_non_log_sync_step_skips_scalar_readback(self):
        replicas = [_PairExchangeFakeBackend(0), _PairExchangeFakeBackend(1)]
        with SynchronousBackendGroup(replicas) as group:
            result = group.train_step((0, 1), step=2, collect_loss=False)
            self.assertIsNone(result["loss"])
            self.assertIn("data_seconds", result)
            self.assertTrue(all(replica.loss_reads == 0 for replica in replicas))

    def test_next_batch_decode_is_scheduled_on_a_separate_executor(self):
        replicas = [_PairExchangeFakeBackend(0), _PairExchangeFakeBackend(1)]
        with SynchronousBackendGroup(replicas) as group:
            group.train_step(
                (0, 1), step=2, next_view_indices=(2, 3),
            )
            self.assertEqual(replicas[0].scheduled_views, (2, 3))
            self.assertIs(
                replicas[0].schedule_executor, replicas[0].prefetch_executor,
            )
            self.assertIsNot(replicas[0].schedule_executor, group._executor)

    def test_evaluation_items_are_partitioned_once_by_rank_stride(self):
        replicas = [_PairExchangeFakeBackend(0), _PairExchangeFakeBackend(1)]
        with SynchronousBackendGroup(replicas) as group:
            shards = group.parallel_shards(
                [10, 11, 12, 13, 14],
                lambda rank, _backend, values: (rank, values),
            )
        self.assertEqual(shards, [(0, (10, 12, 14)), (1, (11, 13))])

    def test_auto_maps_only_to_pair_exchange(self):
        replicas = [_PairExchangeFakeBackend(0), _PairExchangeFakeBackend(1)]
        with SynchronousBackendGroup(replicas, transport="auto") as group:
            result = group.train_step((0, 2), step=0)
            self.assertEqual(group.transport, "pair_host")
            self.assertEqual(
                group.collective_algorithm, "pair_exchange_gpu_merge",
            )
            self.assertEqual(result["reduction_seconds"], 0.0)
            self.assertTrue(result["gradient_async_pipeline"])
            self.assertTrue(result["gradient_persistent_host_mapping"])
            self.assertAlmostEqual(
                float(result["gradient_host_memcpy_seconds"]), 0.02)
            self.assertAlmostEqual(
                float(result["gradient_exchange_wall_seconds"]), 0.15)
            self.assertAlmostEqual(
                float(result["gradient_peer_wait_seconds"]), 0.03)
            self.assertAlmostEqual(
                float(result["gradient_transfer_wall_seconds"]), 0.12)
            # (0.1+0.1)/0.12 ≈ 1.67 transfer overlap; /0.15 ≈ 1.33 exchange overlap
            self.assertGreater(
                float(result["gradient_transfer_overlap"]),
                float(result["gradient_exchange_overlap"]),
            )
            self.assertIn("backward_seconds_max", result)
            self.assertIn("wave_backward_download_seconds", result)
            for replica in replicas:
                np.testing.assert_array_equal(replica.moment, [2.0, 4.0])

    def test_stale_native_pair_abi_is_rejected_before_training(self):
        replicas = [
            _StalePairExchangeFakeBackend(0),
            _StalePairExchangeFakeBackend(1),
        ]
        with self.assertRaisesRegex(RuntimeError, "stale native pair-exchange ABI"):
            SynchronousBackendGroup(replicas)

    def test_pair_relocation_sum_is_available_on_both_replicas(self):
        replicas = [_PairExchangeFakeBackend(0), _PairExchangeFakeBackend(1)]
        replicas[0].stats[:] = [[1.0, 2.0], [3.0, 4.0]]
        replicas[1].stats[:] = [[10.0, 20.0], [30.0, 40.0]]
        with SynchronousBackendGroup(replicas) as group:
            group.relocate(100, "3d_mcmc", True, 7)
            np.testing.assert_array_equal(
                replicas[0].assert_global_stats,
                [[11.0, 22.0], [33.0, 44.0]],
            )

    def test_removed_runtime_transport_values_are_rejected(self):
        for transport in ("packed_host", "host_mailbox", "device_group"):
            with self.subTest(transport=transport), self.assertRaisesRegex(
                ValueError, "expected 'auto' or 'pair_host'",
            ):
                SynchronousBackendGroup(
                    [
                        _PairExchangeFakeBackend(0),
                        _PairExchangeFakeBackend(1),
                    ],
                    transport=transport,
                )

    def test_relocation_sums_pairs_plans_once_and_broadcasts(self):
        replicas = [_PairExchangeFakeBackend(0), _PairExchangeFakeBackend(1)]
        replicas[0].stats[:] = [[1.0, 2.0], [3.0, 4.0]]
        replicas[1].stats[:] = [[10.0, 20.0], [30.0, 40.0]]
        with SynchronousBackendGroup(replicas) as group:
            result = group.relocate(100, "3d_mcmc", True, 7)
            np.testing.assert_array_equal(
                replicas[0].assert_global_stats,
                [[11.0, 22.0], [33.0, 44.0]],
            )
            self.assertEqual(replicas[0].plan_calls, 1)
            self.assertEqual(replicas[1].plan_calls, 0)
            self.assertEqual(result["relocated"], 1)
            self.assertEqual(replicas[0].apply_planned_calls, ["3d_mcmc"])
            self.assertEqual(len(replicas[0].relocation_calls), 0)
            self.assertEqual(len(replicas[1].relocation_calls), 1)
            _mapping, _counts, mode, kwargs = replicas[1].relocation_calls[0]
            self.assertEqual(mode, "3d_mcmc")
            self.assertEqual(kwargs.get("require_dead_destinations"), False)
            self.assertEqual(kwargs.get("require_alive_donors"), False)
            for replica in replicas:
                np.testing.assert_array_equal(replica.stats, np.zeros((2, 2)))

    def test_relocation_caps_oversubscribed_donor_before_broadcast(self):
        class _OversubscribedPlanBackend(_PairExchangeFakeBackend):
            def plan_relocation_native(self, step, include_gate, seed):
                del include_gate, seed
                self.plan_calls += 1
                self.assert_global_stats = self.stats.copy()
                n = 53
                mapping = np.full(n, -1, np.int32)
                mapping[1:] = 0
                counts = np.zeros(n, np.int32)
                counts[0] = 52
                return {
                    "mapping": mapping,
                    "counts": counts,
                    "relocated": 52,
                    "alive": 1,
                    "all_dead": False,
                }

        replicas = [_OversubscribedPlanBackend(0), _OversubscribedPlanBackend(1)]
        with SynchronousBackendGroup(replicas) as group:
            result = group.relocate(100, "3d_mcmc", True, 7)
            self.assertEqual(result["relocated"], 50)
            self.assertEqual(result["alive"], 1)
            for replica in replicas:
                mapping, counts, mode, kwargs = replica.relocation_calls[0]
                self.assertEqual(mode, "3d_mcmc")
                self.assertEqual(kwargs.get("require_dead_destinations"), False)
                self.assertEqual(kwargs.get("require_alive_donors"), False)
                self.assertEqual(int(counts.max()), 50)
                self.assertEqual(int(np.count_nonzero(mapping >= 0)), 50)
                np.testing.assert_array_equal(mapping[1:51], 0)
                np.testing.assert_array_equal(mapping[51:], -1)

    def test_checkpoint_stats_have_one_owner(self):
        replicas = [_PairExchangeFakeBackend(0), _PairExchangeFakeBackend(1)]
        replicas[0].stats[:] = [[1.0, 2.0], [3.0, 4.0]]
        replicas[1].stats[:] = [[10.0, 20.0], [30.0, 40.0]]
        with SynchronousBackendGroup(replicas) as group:
            state = group.checkpoint_optimizer_state()
            expected = np.array([[11.0, 22.0], [33.0, 44.0]], np.float32)
            np.testing.assert_array_equal(state["relocation_stats"], expected)
            np.testing.assert_array_equal(replicas[0].stats, expected)
            np.testing.assert_array_equal(replicas[1].stats, np.zeros((2, 2)))

    def test_train_step_accepts_hard_separation_phase(self):
        replicas = [_PairExchangeFakeBackend(0), _PairExchangeFakeBackend(1)]
        with SynchronousBackendGroup(replicas) as group:
            result = group.train_step((0, 1), 0, phase="hard_separation")
            self.assertEqual(result["views"], (0, 1))


if __name__ == "__main__":
    unittest.main()
