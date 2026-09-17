from __future__ import annotations

import unittest

import numpy as np

from ftgspp.training.distributed_contract import (
    DistributedContractError,
    ReplicaDivergenceError,
    assert_replicas_consistent,
    cap_relocation_donor_copies,
    shard_resumed_relocation_statistics,
    sum_relocation_statistics,
    validate_checkpoint_optimizer_state,
)
from  tests.reference.distributed import (
    add_shared_gradients,
    consolidate_optimizer_state_for_checkpoint,
    deterministic_relocation_plan,
    mean_raw_gradients,
    reduce_color_gradients,
    relocation_scores,
    state_digest,
)


def _base_optimizer_state(n: int) -> dict[str, np.ndarray]:
    padded_n = ((n + 31) // 32) * 32
    return {
        "g_xyz_ws": np.zeros(6 * n, np.float32),
        "g_sh_coeffs_1": np.zeros(48 * padded_n, np.float32),
        "g_sh_coeffs_2": np.zeros(48 * padded_n, np.float32),
        "g_rotations": np.zeros(8 * n, np.float32),
        "g_scales_opacs": np.zeros(8 * n, np.float32),
        "g_times_durations": np.zeros(4 * n, np.float32),
        "g_velocities_gates": np.zeros(8 * n, np.float32),
        "relocation_stats": np.zeros((n, 2), np.float32),
    }


class DistributedGradientContractTests(unittest.TestCase):
    def test_raw_gradients_are_meaned_before_optimizer(self):
        gradients = mean_raw_gradients([
            {"means": np.array([-1.0, 2.0], np.float32)},
            {"means": np.array([3.0, 6.0], np.float32)},
        ])
        np.testing.assert_array_equal(gradients["means"], [1.0, 4.0])

        # A first-step Adam update is approximately sign(g).  Updating each
        # rank and averaging deltas would give zero for [-1, +3], whereas
        # averaging raw gradients first correctly gives a negative update.
        local_parameter_delta_mean = (-np.sign(-1.0) - np.sign(3.0)) / 2.0
        global_parameter_delta = -np.sign((-1.0 + 3.0) / 2.0)
        self.assertEqual(local_parameter_delta_mean, 0.0)
        self.assertEqual(global_parameter_delta, -1.0)

    def test_raw_gradient_layout_and_finiteness_are_strict(self):
        with self.assertRaisesRegex(DistributedContractError, "keys differ"):
            mean_raw_gradients([
                {"hash": np.zeros(2, np.float32)},
                {"mlp": np.zeros(2, np.float32)},
            ])
        with self.assertRaisesRegex(DistributedContractError, "non-finite"):
            mean_raw_gradients([{"hash": np.array([np.nan], np.float32)}])

    def test_shared_regularizer_is_added_once_after_data_mean(self):
        data = mean_raw_gradients([
            {"gate": np.array([2.0], np.float32)},
            {"gate": np.array([6.0], np.float32)},
            {"gate": np.array([10.0], np.float32)},
            {"gate": np.array([14.0], np.float32)},
        ])
        combined = add_shared_gradients(
            data, {"gate": np.array([3.0], np.float32)}
        )
        np.testing.assert_array_equal(combined["gate"], [11.0])

    def test_generic_shared_gradient_is_added_once(self):
        reduced = mean_raw_gradients([
            {
                "parameter_a": np.array([1.0, 3.0], np.float32),
                "parameter_b": np.array([2.0], np.float32),
            },
            {
                "parameter_a": np.array([5.0, 7.0], np.float32),
                "parameter_b": np.array([6.0], np.float32),
            },
        ])
        total = add_shared_gradients(reduced, {
            "parameter_a": np.array([0.25, -0.5], np.float32),
            "parameter_b": np.array([1.5], np.float32),
        })
        np.testing.assert_allclose(total["parameter_a"], [3.25, 4.5])
        np.testing.assert_allclose(total["parameter_b"], [5.5])

    def test_color_sparse_camera_gradients_are_view_mean_plus_l2_derivative(self):
        params = np.arange(24, dtype=np.float32).reshape(2, 12) / 10.0
        rank0 = np.zeros_like(params)
        rank1 = np.zeros_like(params)
        rank0[0] = 2.0
        rank1[1] = 4.0
        result = reduce_color_gradients([rank0, rank1], params, 0.1)
        expected = np.zeros_like(params)
        expected[0] = 1.0
        expected[1] = 2.0
        # FTGSPP-main uses 0.1 * sum(params**2), whose derivative is 0.2 * params.
        expected += 0.2 * params
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-7)


class DistributedRelocationContractTests(unittest.TestCase):
    def test_statistics_sum_magnitudes_and_counts_not_vectors_or_means(self):
        rank0 = np.array([[2.0, 1.0], [0.0, 0.0]], np.float32)
        rank1 = np.array([[0.0, 0.0], [4.0, 1.0]], np.float32)
        total = sum_relocation_statistics([rank0, rank1])
        np.testing.assert_array_equal(total, [[2.0, 1.0], [4.0, 1.0]])
        np.testing.assert_array_equal(relocation_scores(total), [2.0, 4.0])

        incorrectly_meaned = (rank0 + rank1) / 2.0
        np.testing.assert_array_equal(relocation_scores(incorrectly_meaned), [1.0, 2.0])
        # Opposite projected directions still contribute two units of score.
        projected_vectors = np.array([[1.0, 0.0], [-1.0, 0.0]], np.float32)
        self.assertEqual(float(np.linalg.norm(projected_vectors, axis=1).sum()), 2.0)
        self.assertEqual(float(np.linalg.norm(projected_vectors.sum(axis=0))), 0.0)

    def test_relocation_plan_is_deterministic_and_only_uses_alive_donors(self):
        opacity_logits = np.array([-10.0, 2.0, 1.0, -9.0, 3.0], np.float32)
        gate_raw = np.array([-1.0, -0.5, 0.5, 0.0, 1.0], np.float32)
        stats = np.array([
            [0.0, 0.0], [4.0, 2.0], [1.0, 1.0], [0.0, 0.0], [9.0, 3.0],
        ], np.float32)
        kwargs = dict(
            opacity_threshold=5e-3, include_gate=True, seed=17, step=500,
        )
        first = deterministic_relocation_plan(opacity_logits, gate_raw, stats, **kwargs)
        second = deterministic_relocation_plan(opacity_logits, gate_raw, stats, **kwargs)
        np.testing.assert_array_equal(first.mapping, second.mapping)
        np.testing.assert_array_equal(first.counts, second.counts)
        np.testing.assert_array_equal(first.donor_cdf, second.donor_cdf)
        self.assertEqual(first.relocated, 2)
        alive = set(np.flatnonzero(first.raw_scores[:, 3] > 0.5).tolist())
        donors = set(first.mapping[first.mapping >= 0].tolist())
        self.assertTrue(donors <= alive)
        self.assertEqual(int(first.counts.sum()), first.relocated)

    def test_degenerate_enabled_score_falls_back_to_whole_alive_uniform(self):
        # The gradient component is enabled but has a zero total.  FTGSPP's
        # final isfinite check replaces the *whole* probability vector, so the
        # very different opacity scores below must not survive in the CDF.
        plan = deterministic_relocation_plan(
            np.array([-10.0, -1.0, 1.0, 6.0], np.float32),
            np.zeros(4, np.float32),
            np.zeros((4, 2), np.float32),
            opacity_threshold=5e-3,
            include_gate=False,
            seed=19,
            step=700,
        )
        np.testing.assert_allclose(
            plan.donor_cdf,
            np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], np.float32),
            rtol=0.0,
            atol=1e-7,
        )

        # Gate degeneracy is relevant only when gate scoring is enabled.  With
        # saturated dynamic gates its zero total must trigger the same global
        # fallback even though opacity and gradient totals remain valid.
        gate_plan = deterministic_relocation_plan(
            np.array([-10.0, -1.0, 1.0, 6.0], np.float32),
            np.full(4, 100.0, np.float32),
            np.array([[0.0, 0.0], [1.0, 1.0], [4.0, 1.0], [9.0, 1.0]], np.float32),
            opacity_threshold=5e-3,
            include_gate=True,
            seed=19,
            step=700,
        )
        np.testing.assert_allclose(
            gate_plan.donor_cdf,
            np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], np.float32),
            rtol=0.0,
            atol=1e-7,
        )

        gate_disabled_plan = deterministic_relocation_plan(
            np.array([-10.0, -1.0, 1.0, 6.0], np.float32),
            np.full(4, 100.0, np.float32),
            np.array([[0.0, 0.0], [1.0, 1.0], [4.0, 1.0], [9.0, 1.0]], np.float32),
            opacity_threshold=5e-3,
            include_gate=False,
            seed=19,
            step=700,
        )
        self.assertFalse(np.allclose(
            gate_disabled_plan.donor_cdf,
            np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], np.float32),
            rtol=0.0,
            atol=1e-7,
        ))

    def test_all_dead_plan_matches_native_no_relocation_semantics(self):
        plan = deterministic_relocation_plan(
            np.full(3, -100.0, np.float32),
            np.zeros(3, np.float32),
            np.zeros((3, 2), np.float32),
            opacity_threshold=5e-3,
            include_gate=False,
            seed=0,
            step=100,
        )
        self.assertTrue(plan.all_dead)
        self.assertEqual(plan.alive, 0)
        self.assertEqual(plan.relocated, 0)
        np.testing.assert_array_equal(plan.mapping, [-1, -1, -1])
        np.testing.assert_array_equal(plan.counts, [0, 0, 0])


class DistributedStateContractTests(unittest.TestCase):
    def test_digest_and_replica_check_cover_optimizer_and_color_state(self):
        state0 = {
            "g_xyz_ws": np.array([1.0, 2.0], np.float32),
            "color_params": np.zeros((1, 12), np.float32),
            "color_m": np.zeros((1, 12), np.float32),
            "color_v": np.zeros((1, 12), np.float32),
        }
        state1 = {key: value.copy() for key, value in state0.items()}
        self.assertEqual(state_digest(state0), state_digest(state1))
        assert_replicas_consistent([state0, state1])
        state1["g_xyz_ws"][1] += 1e-3
        with self.assertRaisesRegex(ReplicaDivergenceError, "g_xyz_ws"):
            assert_replicas_consistent([state0, state1])

    def test_checkpoint_rejects_legacy_field_state(self):
        n = 2
        state = _base_optimizer_state(n)
        state["field_hash_params"] = np.zeros(3, np.float32)
        with self.assertRaisesRegex(DistributedContractError, "explicit-only"):
            validate_checkpoint_optimizer_state(
                state, num_gaussians=n, color_num_cameras=0,
            )

    def test_checkpoint_consolidates_stats_and_resume_has_one_history_owner(self):
        n = 2
        rank0 = _base_optimizer_state(n)
        rank1 = {key: value.copy() for key, value in rank0.items()}
        rank0["relocation_stats"] = np.array([[3.0, 1.0], [0.0, 0.0]], np.float32)
        rank1["relocation_stats"] = np.array([[0.0, 0.0], [8.0, 2.0]], np.float32)
        checkpoint = consolidate_optimizer_state_for_checkpoint([rank0, rank1])
        np.testing.assert_array_equal(
            checkpoint["relocation_stats"], [[3.0, 1.0], [8.0, 2.0]]
        )

        resumed = shard_resumed_relocation_statistics(
            checkpoint["relocation_stats"], world_size=2
        )
        np.testing.assert_array_equal(resumed[0], checkpoint["relocation_stats"])
        np.testing.assert_array_equal(resumed[1], np.zeros((n, 2), np.float32))

        new0 = resumed[0] + np.array([[1.0, 1.0], [0.0, 0.0]], np.float32)
        new1 = resumed[1] + np.array([[0.0, 0.0], [2.0, 1.0]], np.float32)
        next_global = sum_relocation_statistics([new0, new1])
        np.testing.assert_array_equal(next_global, [[4.0, 2.0], [10.0, 3.0]])

        incorrectly_broadcast = sum_relocation_statistics([
            checkpoint["relocation_stats"], checkpoint["relocation_stats"]
        ])
        np.testing.assert_array_equal(incorrectly_broadcast, [[6.0, 2.0], [16.0, 4.0]])

    def test_checkpoint_rejects_negative_relocation_statistics(self):
        state = _base_optimizer_state(1)
        state["relocation_stats"][0, 1] = -1.0
        with self.assertRaisesRegex(DistributedContractError, "non-negative"):
            validate_checkpoint_optimizer_state(
                state,
                num_gaussians=1,
                color_num_cameras=0,
            )

    def test_checkpoint_requires_exact_native_sh_subgroup_padding(self):
        state = _base_optimizer_state(33)
        state["g_sh_coeffs_1"] = np.zeros(48 * 33, np.float32)
        with self.assertRaisesRegex(
            DistributedContractError, r"48\*ceil\(33/32\)\*32",
        ):
            validate_checkpoint_optimizer_state(
                state,
                num_gaussians=33,
                color_num_cameras=0,
            )

    def test_cap_relocation_donor_copies_keeps_lowest_destinations(self):
        mapping = np.full(6, -1, np.int32)
        mapping[1:] = 0
        counts = np.array([5, 0, 0, 0, 0, 0], np.int32)
        capped_map, capped_counts = cap_relocation_donor_copies(
            mapping, counts, max_copies=2,
        )
        np.testing.assert_array_equal(capped_map, [-1, 0, 0, -1, -1, -1])
        np.testing.assert_array_equal(capped_counts, [2, 0, 0, 0, 0, 0])
        unchanged_map, unchanged_counts = cap_relocation_donor_copies(
            capped_map, capped_counts, max_copies=2,
        )
        np.testing.assert_array_equal(unchanged_map, capped_map)
        np.testing.assert_array_equal(unchanged_counts, capped_counts)

    def test_cap_relocation_donor_copies_is_stable_across_two_donors(self):
        mapping = np.array([-1, 0, 1, 0, 1, 0, 1, 0], np.int32)
        counts = np.bincount(mapping[mapping >= 0], minlength=8).astype(np.int32)
        capped_map, capped_counts = cap_relocation_donor_copies(
            mapping, counts, max_copies=2,
        )
        np.testing.assert_array_equal(
            capped_map, [-1, 0, 1, 0, 1, -1, -1, -1],
        )
        np.testing.assert_array_equal(capped_counts, [2, 2, 0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
