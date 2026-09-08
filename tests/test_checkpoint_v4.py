from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ftgspp.native_runtime.checkpoint import (
    CheckpointSnapshot,
    ResumeCompatibilityError,
    SAMPLING_PHASE_SEED_FORMULA,
    SINGLE_VIEW_SAMPLING,
    SYNC_FLAT_EPOCH_VIEW_SAMPLING,
    SYNC_VIEW_SAMPLING,
    TORCH_RANDOM_SAMPLER_VIEW_SAMPLING,
    capture_checkpoint_snapshot,
    checkpoint_sampler_abi,
    collect_runtime_provenance,
    latest_checkpoint,
    load_checkpoint_full,
    publish_checkpoint,
    validate_or_infer_training_clocks,
    validate_resume_compatibility,
)
from ftgspp.native_runtime.model import NativeGaussians


def _model(n: int = 2) -> NativeGaussians:
    rotations = np.zeros((n, 4), np.float32)
    rotations[:, 0] = 1.0
    return NativeGaussians(
        means=np.zeros((n, 3), np.float32),
        log_scales=np.zeros((n, 3), np.float32),
        quaternions=rotations,
        opacity_logits=np.zeros((n, 1), np.float32),
        sh_coefficients=np.zeros((n, 16, 3), np.float32),
        times=np.zeros((n, 1), np.float32),
        duration_raw=np.zeros((n, 1), np.float32),
        velocities=np.zeros((n, 3), np.float32),
        gate_raw=np.zeros((n, 1), np.float32),
        max_duration=5.0,
    )


class _SnapshotBackend:
    def __init__(self) -> None:
        self.model_calls = 0
        self.optimizer_calls = 0

    def model(self) -> NativeGaussians:
        self.model_calls += 1
        return _model()

    def optimizer_state(self) -> dict[str, np.ndarray]:
        self.optimizer_calls += 1
        return {
            "moment": np.arange(4, dtype=np.float32),
            "relocation_stats": np.zeros((2, 2), np.float32),
        }


def _metadata(
    *, world_size: int = 2, policy: str = "unique", seed: int = 0,
) -> dict:
    return {
        "checkpoint_schema": "ftgspp-vulkan-v4",
        "phase": "main",
        "phase_step": 7,
        "training": {
            "seed": seed,
            "world_size": world_size,
            "global_batch_size": world_size,
            "per_device_batch_size": 1,
            "reduction": "mean",
            "requested_backend": "hierarchical_host",
            "transport": "hierarchical_host",
            "collective": {
                "abi": "ftgspp-hierarchical-host-v1",
                "algorithm": "pair_exchange_gpu_merge",
                "P": world_size,
                "K": 1,
                "reduction_tree": "fixed_rank_order_v1",
                "codec_policy": "auto_exact",
                "topology": {
                    "policy": "device_uuid_pci_numa_v1",
                    "fingerprint": "topology-a",
                },
            },
            "clocks": {
                "abi": "ftgspp-four-training-clocks-v1",
                "optimizer_steps": 8,
                "microsteps": 8,
                "views_seen": 8 * world_size,
                "wall_seconds": 1.0,
            },
            "sampling_policy": policy,
            "sampler_abi": checkpoint_sampler_abi(
                synchronous=True, sampling_policy=policy,
            ),
            "phase_seed_formula": SAMPLING_PHASE_SEED_FORMULA,
            "phase_seeds": {
                "main": seed,
            },
        },
        "dataset": {
            "fingerprint": "dataset-a",
            "frames": [0, 1],
            "fps": 60.0,
            "time_offsets": {"0": 0.0},
            "camera_dense_mapping": {"0": 0},
            "train_cameras": [0],
            "eval_cameras": [],
        },
        "model_config": {
            "velocity_model": "explicit",
            "marginal_gating": True,
            "max_duration": 1.0,
            "sh_degree": 3,
            "color_correction": True,
        },
        "algorithm": {
            "sha256": "algorithm-a",
            "config": {"optimizer": {"iterations": 30_000}},
        },
        "runtime": {
            "native_abi": "abi-1",
            "native_reported_abi": "native-abi-1",
            "devices": [
                {
                    "uuid": f"uuid-{rank}", "name": "test-gpu",
                    "driver_version": "1",
                }
                for rank in range(world_size)
            ],
            "shaders": {"set_sha256": "shader-a"},
            "git": {"commit": "commit-a"},
        },
    }


class CheckpointV4Tests(unittest.TestCase):
    def test_shader_provenance_recurses_with_portable_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            shader_root = Path(directory)
            generated = shader_root / "generated"
            radix_sort = shader_root / "radix_sort"
            generated.mkdir()
            radix_sort.mkdir()
            (generated / "render.spv").write_bytes(b"render-v1")
            (radix_sort / "upsweep.spv").write_bytes(b"radix-v1")
            (shader_root / "ignored.txt").write_text("not a shader", "utf-8")

            with mock.patch.dict(
                os.environ, {"FTGSPP_VK_SHADER_DIR": str(shader_root)},
            ):
                first = collect_runtime_provenance(device_ids=[0])
                (radix_sort / "upsweep.spv").write_bytes(b"radix-v2")
                second = collect_runtime_provenance(device_ids=[0])

        self.assertEqual(
            set(first["shaders"]["files"]),
            {"generated/render.spv", "radix_sort/upsweep.spv"},
        )
        self.assertNotEqual(
            first["shaders"]["files"]["radix_sort/upsweep.spv"],
            second["shaders"]["files"]["radix_sort/upsweep.spv"],
        )
        self.assertNotEqual(
            first["shaders"]["set_sha256"],
            second["shaders"]["set_sha256"],
        )

    def test_capture_downloads_model_and_optimizer_once_then_latest_is_pointer(self):
        backend = _SnapshotBackend()
        rng = np.random.default_rng(3)
        snapshot = capture_checkpoint_snapshot(
            backend, 7, rng, metadata=_metadata()
        )
        self.assertEqual(backend.model_calls, 1)
        self.assertEqual(backend.optimizer_calls, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = publish_checkpoint(root, snapshot)
            self.assertEqual(target.name, "000007.npz")
            self.assertFalse((root / "latest.npz").exists())
            pointer = json.loads((root / "latest.json").read_text("utf-8"))
            self.assertEqual(pointer["checkpoint"], "000007.npz")
            self.assertEqual(latest_checkpoint(root), target)
            with self.assertRaisesRegex(FileExistsError, "immutable checkpoint"):
                publish_checkpoint(root, snapshot)
            restored = load_checkpoint_full(target)
        self.assertEqual(restored.iteration, 7)
        self.assertEqual(restored.metadata["phase"], "main")
        np.testing.assert_array_equal(
            restored.optimizer["moment"], snapshot.optimizer["moment"]
        )
        self.assertEqual(backend.model_calls, 1)
        self.assertEqual(backend.optimizer_calls, 1)

    def test_converter_resolves_atomic_latest_pointer(self):
        from ftgspp.checkpoint_convert import convert

        metadata = _metadata()
        metadata["phase"] = "main_complete"
        snapshot = capture_checkpoint_snapshot(
            _SnapshotBackend(), 7, np.random.default_rng(3),
            metadata=metadata,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = publish_checkpoint(root, snapshot)
            source_metadata = load_checkpoint_full(source).metadata
            converted = root / "converted.npz"
            convert(root / "latest.json", converted)
            restored = load_checkpoint_full(converted)
        self.assertEqual(restored.iteration, 7)
        self.assertEqual(restored.metadata["phase"], "main_complete")
        self.assertEqual(
            restored.metadata["checkpoint_schema"], "ftgspp-vulkan-v5-converted"
        )
        self.assertEqual(
            restored.metadata["runtime"]["devices"],
            source_metadata["runtime"]["devices"],
        )
        self.assertEqual(
            restored.metadata["runtime"]["git"],
            source_metadata["runtime"]["git"],
        )

    def test_converter_records_gate_and_phase_transforms_without_losing_provenance(self):
        metadata = _metadata()
        snapshot = capture_checkpoint_snapshot(
            _SnapshotBackend(), 7, np.random.default_rng(3), metadata=metadata,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = publish_checkpoint(root, snapshot)
            destination = root / "transformed.npz"
            from ftgspp.checkpoint_convert import convert
            convert(
                source, destination, marginal_gating=False,
                phase="main_complete",
            )
            restored = load_checkpoint_full(destination)
        self.assertEqual(
            restored.metadata["runtime"]["devices"],
            snapshot.metadata["runtime"]["devices"],
        )
        self.assertEqual(
            restored.metadata["runtime"]["git"],
            snapshot.metadata["runtime"]["git"],
        )
        self.assertEqual(
            restored.metadata["runtime"]["native_abi"],
            "ftgspp-vulkan-community-explicit-v5-abi-v1",
        )
        self.assertEqual(restored.metadata["dataset"], snapshot.metadata["dataset"])
        self.assertEqual(restored.metadata["phase"], "main_complete")
        self.assertEqual(restored.metadata["phase_step"], -1)
        self.assertFalse(restored.metadata["model_config"]["marginal_gating"])
        record = restored.metadata["migration_history"][-1]
        self.assertEqual(record["source"]["runtime"], snapshot.metadata["runtime"])
        self.assertEqual(
            [item["kind"] for item in record["transforms"]],
            ["checkpoint_schema", "marginal_gating", "phase"],
        )

    def test_field_sampling_metadata_marks_explicit_transform_and_rehashes_algorithm(self):
        from ftgspp.checkpoint_convert import _metadata_for_conversion

        metadata = _metadata()
        metadata["model_config"]["velocity_model"] = "field"
        metadata["algorithm"] = {
            "sha256": "source-field-hash",
            "config": {"model": {
                "velocity_model": "field", "marginal_gating": True,
            }},
        }
        transformed = _metadata_for_conversion(
            metadata, _model(), {}, phase="main",
            transformations=[{
                "kind": "velocity_field", "operation": "sample",
                "from": "field", "to": "explicit",
            }],
        )
        self.assertEqual(transformed["model_config"]["velocity_model"], "explicit")
        self.assertEqual(
            transformed["algorithm"]["config"]["model"]["velocity_model"],
            "explicit",
        )
        self.assertNotEqual(
            transformed["algorithm"]["sha256"], "source-field-hash",
        )
        record = transformed["migration_history"][-1]
        self.assertEqual(record["source"]["algorithm"], metadata["algorithm"])
        self.assertEqual(record["transforms"][0]["kind"], "velocity_field")

    def test_numbered_fallback_sorts_iteration_numerically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "999999.npz").write_bytes(b"")
            expected = root / "1000000.npz"
            expected.write_bytes(b"")
            self.assertEqual(latest_checkpoint(root), expected)

    def test_latest_pointer_must_match_target_format_and_iteration(self):
        snapshot = capture_checkpoint_snapshot(
            _SnapshotBackend(), 7, np.random.default_rng(3),
            metadata=_metadata(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publish_checkpoint(root, snapshot)
            pointer_path = root / "latest.json"
            pointer = json.loads(pointer_path.read_text("utf-8"))
            pointer["iteration"] = 8
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "iteration does not match"):
                latest_checkpoint(root)

    def test_supplied_distributed_optimizer_avoids_second_optimizer_download(self):
        backend = _SnapshotBackend()
        supplied = {
            "moment": np.ones(4, np.float32),
            "relocation_stats": np.zeros((2, 2), np.float32),
        }
        capture_checkpoint_snapshot(
            backend, 1, np.random.default_rng(0),
            optimizer_state=supplied, metadata=_metadata(),
        )
        self.assertEqual(backend.model_calls, 1)
        self.assertEqual(backend.optimizer_calls, 0)

    def test_world_size_change_requires_explicit_acknowledgement(self):
        saved = _metadata(world_size=2)
        current = _metadata(world_size=4)
        current["training"]["global_batch_size"] = 4
        with self.assertRaisesRegex(
            ResumeCompatibilityError, "allow_resume_world_size_change"
        ):
            validate_resume_compatibility(
                saved, current, allow_world_size_change=False,
            )
        warnings = validate_resume_compatibility(
            saved, current, allow_world_size_change=True,
        )
        self.assertTrue(any("optimizer trajectory" in value for value in warnings))

    def test_legacy_production_collectives_are_explicitly_unsupported(self):
        for value in ("packed_host", "host_mailbox", "device_group"):
            with self.subTest(value=value):
                saved = _metadata()
                saved["training"]["transport"] = value
                with self.assertRaisesRegex(
                    ResumeCompatibilityError, "checkpoint is unsupported",
                ):
                    validate_resume_compatibility(
                        saved, _metadata(), allow_world_size_change=False,
                    )

    def test_collective_abi_algorithm_k_tree_and_topology_are_resume_contracts(self):
        fields = {
            "abi": "ftgspp-hierarchical-host-v2",
            "algorithm": "central_hierarchical_reduce_broadcast",
            "K": 2,
            "reduction_tree": "dual_numa_v1",
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                saved = _metadata()
                saved["training"]["collective"][field] = value
                with self.assertRaisesRegex(
                    ResumeCompatibilityError,
                    rf"training\.collective\.{field}",
                ):
                    validate_resume_compatibility(
                        saved, _metadata(), allow_world_size_change=False,
                    )
        saved = _metadata()
        saved["training"]["collective"]["topology"]["fingerprint"] = "other"
        with self.assertRaisesRegex(
            ResumeCompatibilityError, "topology fingerprint",
        ):
            validate_resume_compatibility(
                saved, _metadata(), allow_world_size_change=False,
            )

    def test_sampling_policy_and_sampler_abi_are_resume_contracts(self):
        saved = _metadata(policy="unique")
        current = _metadata(policy="with_replacement")
        with self.assertRaisesRegex(ResumeCompatibilityError, "sampling_policy"):
            validate_resume_compatibility(
                saved, current, allow_world_size_change=False,
            )

    def test_phase_seed_bumps_every_sampler_abi_and_rejects_old_v1(self):
        single_v2 = (
            f"{SINGLE_VIEW_SAMPLING};{SAMPLING_PHASE_SEED_FORMULA};policy=unique;"
            "replacement_fallback=views_lt_world_size"
        )
        unique_v2 = (
            f"{SYNC_VIEW_SAMPLING};{SAMPLING_PHASE_SEED_FORMULA};policy=unique;"
            "replacement_fallback=views_lt_world_size"
        )
        self.assertEqual(
            checkpoint_sampler_abi(
                synchronous=False, sampling_policy="unique",
            ),
            single_v2,
        )
        self.assertEqual(
            checkpoint_sampler_abi(
                synchronous=True, sampling_policy="unique",
            ),
            unique_v2,
        )
        self.assertEqual(
            checkpoint_sampler_abi(
                synchronous=True, sampling_policy="flat_epoch",
            ),
            SYNC_FLAT_EPOCH_VIEW_SAMPLING,
        )
        self.assertEqual(
            checkpoint_sampler_abi(
                synchronous=False, sampling_policy="flat_epoch",
            ),
            SYNC_FLAT_EPOCH_VIEW_SAMPLING,
        )
        self.assertEqual(
            checkpoint_sampler_abi(
                synchronous=False, sampling_policy="torch_random_sampler",
            ),
            TORCH_RANDOM_SAMPLER_VIEW_SAMPLING
            + ";mode=off;microsteps=1",
        )
        self.assertEqual(
            checkpoint_sampler_abi(
                synchronous=True, sampling_policy="torch_random_sampler",
                microsteps=2,
            ),
            TORCH_RANDOM_SAMPLER_VIEW_SAMPLING
            + ";mode=sync;microsteps=2",
        )
        self.assertIn(
            "step=phase_local_optimizer_step;slot=step*world_size+rank",
            SYNC_FLAT_EPOCH_VIEW_SAMPLING,
        )
        k2 = checkpoint_sampler_abi(
            synchronous=True, sampling_policy="flat_epoch", microsteps=2,
        )
        self.assertIn(
            "slot=step*(world_size*microsteps)+rank*microsteps+microstep",
            k2,
        )
        self.assertIn("microsteps=2", k2)
        self.assertIn("epoch=slot//num_train_views", SYNC_FLAT_EPOCH_VIEW_SAMPLING)
        self.assertIn(SAMPLING_PHASE_SEED_FORMULA, unique_v2)
        self.assertIn(SAMPLING_PHASE_SEED_FORMULA, SYNC_FLAT_EPOCH_VIEW_SAMPLING)

        # Old V4 checkpoints name the pre-phase-domain sampler ABI. The normal
        # compatibility check rejects them instead of silently replaying main's
        # early view stream in hard separation.
        saved = _metadata(policy="unique")
        current = _metadata(policy="unique")
        saved["training"]["sampler_abi"] = (
            "numpy-seedsequence(seed,optimizer_step,FTGS)-choice-v1;"
            "policy=unique;replacement_fallback=views_lt_world_size"
        )
        with self.assertRaisesRegex(ResumeCompatibilityError, "sampler_abi"):
            validate_resume_compatibility(
                saved, current, allow_world_size_change=False,
            )

    def test_phase_seed_metadata_is_a_resume_contract(self):
        saved = _metadata()
        current = _metadata()
        current["training"]["phase_seeds"] = {
            "main": 1,
        }
        with self.assertRaisesRegex(ResumeCompatibilityError, "phase_seeds"):
            validate_resume_compatibility(
                saved, current, allow_world_size_change=False,
            )

    def test_four_clocks_validate_against_iteration_p_and_k(self):
        metadata = _metadata(world_size=2)
        clocks, warning = validate_or_infer_training_clocks(
            metadata, iteration=7,
        )
        self.assertIsNone(warning)
        self.assertEqual(clocks["views_seen"], 16)
        metadata["training"]["clocks"]["microsteps"] = 7
        with self.assertRaisesRegex(
            ResumeCompatibilityError, "inconsistent with iteration/P/K",
        ):
            validate_or_infer_training_clocks(metadata, iteration=7)

    def test_clockless_legacy_checkpoint_uses_explicit_k1_inference(self):
        metadata = _metadata(world_size=4)
        del metadata["training"]["clocks"]
        clocks, warning = validate_or_infer_training_clocks(
            metadata, iteration=7,
        )
        self.assertEqual(clocks, {
            "optimizer_steps": 8,
            "microsteps": 8,
            "views_seen": 32,
            "wall_seconds": 0.0,
        })
        self.assertIn("explicit K=1 legacy inference", warning)
        self.assertEqual(
            metadata["training"]["clocks"]["abi"],
            "ftgspp-four-training-clocks-v1",
        )
        current = _metadata(world_size=4)
        self.assertEqual(
            validate_resume_compatibility(
                metadata, current, allow_world_size_change=False,
            ),
            [],
        )

    def test_seed_and_algorithm_configuration_are_resume_contracts(self):
        with self.assertRaisesRegex(ResumeCompatibilityError, "training.seed"):
            validate_resume_compatibility(
                _metadata(seed=1), _metadata(seed=2),
                allow_world_size_change=False,
            )
        current = _metadata()
        current["algorithm"] = {
            "sha256": "algorithm-b",
            "config": {"optimizer": {"iterations": 15_000}},
        }
        with self.assertRaisesRegex(
            ResumeCompatibilityError, "algorithm configuration"
        ):
            validate_resume_compatibility(
                _metadata(), current, allow_world_size_change=False,
            )

    def test_device_change_warns_but_does_not_block_resume(self):
        current = _metadata()
        current["runtime"]["devices"][0]["driver_version"] = "2"
        warnings = validate_resume_compatibility(
            _metadata(), current, allow_world_size_change=False,
        )
        self.assertTrue(any("device provenance changed" in item for item in warnings))

    def test_v4_loader_rejects_a_forged_layout_manifest(self):
        snapshot = CheckpointSnapshot(
            model=_model(),
            optimizer={"moment": np.zeros(4, np.float32)},
            iteration=1,
            rng_state=np.random.default_rng(0).bit_generator.state,
            metadata={**_metadata(), "layout": {
                "model": {}, "optimizer": {}, "field": {}, "color": {},
            }},
        )
        # publish_checkpoint/save_checkpoint_snapshot would preserve the forged
        # manifest; load must compare it with actual arrays and reject it.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000001.npz"
            from ftgspp.native_runtime.checkpoint import save_checkpoint_snapshot
            save_checkpoint_snapshot(path, snapshot)
            with self.assertRaisesRegex(ValueError, "layout manifest"):
                load_checkpoint_full(path)


if __name__ == "__main__":
    unittest.main()
