from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ftgspp.io.checkpoint import (
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
    validate_training_clocks,
    validate_resume_compatibility,
)
from ftgspp.io.model import NativeGaussians


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
        "phase": "main",
        "phase_step": 7,
        "training": {
            "seed": seed,
            "world_size": world_size,
            "reduction": "mean",
            "requested_backend": "pair_host",
            "transport": "pair_host",
            "collective": {
                "abi": "ftgspp-pair-host-v1",
                "algorithm": "pair_exchange_gpu_merge",
                "P": world_size,
                "reduction_tree": "fixed_rank_order_v1",
                "codec_policy": "auto_exact",
                "topology": {
                    "policy": "device_uuid_pci_numa_v1",
                    "fingerprint": "topology-a",
                },
            },
            "clocks": {
                "abi": "ftgspp-training-clocks-v2",
                "optimizer_steps": 8,
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
            "hard_separation": False,
        },
        "algorithm": {
            "sha256": "algorithm-a",
            "config": {
                "optimizer": {"iterations": 30_000},
                "hard_separation": False,
            },
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


class CheckpointV6Tests(unittest.TestCase):
    def test_source_tree_change_warns_on_resume(self):
        saved = _metadata()
        current = _metadata()
        saved["runtime"]["source_tree"] = {
            "schema": "ftgspp-source-tree-v1", "set_sha256": "source-a",
        }
        current["runtime"]["source_tree"] = {
            "schema": "ftgspp-source-tree-v1", "set_sha256": "source-b",
        }
        warnings = validate_resume_compatibility(
            saved, current, allow_world_size_change=False,
        )
        self.assertTrue(any("source-tree" in value for value in warnings))

    def test_shader_provenance_recurses_and_hashes_complete_set(self):
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

        self.assertEqual(first["shaders"]["file_count"], 2)
        self.assertEqual(second["shaders"]["file_count"], 2)
        self.assertNotEqual(
            first["shaders"]["set_sha256"],
            second["shaders"]["set_sha256"],
        )

    def test_capture_downloads_model_and_optimizer_once_then_latest_is_pointer(self):
        backend = _SnapshotBackend()
        snapshot = capture_checkpoint_snapshot(
            backend, 7, metadata=_metadata()
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

    def test_numbered_fallback_sorts_iteration_numerically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "999999.npz").write_bytes(b"")
            expected = root / "1000000.npz"
            expected.write_bytes(b"")
            self.assertEqual(latest_checkpoint(root), expected)

    def test_latest_pointer_must_match_target_format_and_iteration(self):
        snapshot = capture_checkpoint_snapshot(
            _SnapshotBackend(), 7,
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
            backend, 1,
            optimizer_state=supplied, metadata=_metadata(),
        )
        self.assertEqual(backend.model_calls, 1)
        self.assertEqual(backend.optimizer_calls, 0)

    def test_supplied_model_avoids_model_download_for_selected_handoff(self):
        backend = _SnapshotBackend()
        selected = _model()
        snapshot = capture_checkpoint_snapshot(
            backend, 29_999, model=selected, optimizer_state={},
            metadata={**_metadata(), "checkpoint_role": "selected"},
        )
        self.assertIs(snapshot.model, selected)
        self.assertEqual(backend.model_calls, 0)
        self.assertEqual(backend.optimizer_calls, 0)

    def test_selected_main_handoff_only_allows_enabling_hs(self):
        saved = _metadata()
        current = copy.deepcopy(saved)
        current["model_config"]["hard_separation"] = True
        current["algorithm"]["sha256"] = "algorithm-hs"
        current["algorithm"]["config"]["hard_separation"] = True
        with self.assertRaisesRegex(
            ResumeCompatibilityError, "hard_separation",
        ):
            validate_resume_compatibility(
                saved, current, allow_world_size_change=False,
            )
        self.assertEqual(
            validate_resume_compatibility(
                saved, current, allow_world_size_change=False,
                allow_hard_separation_enable=True,
            ),
            [],
        )
        current["algorithm"]["config"]["optimizer"]["iterations"] = 20_000
        with self.assertRaisesRegex(
            ResumeCompatibilityError, "algorithm configuration",
        ):
            validate_resume_compatibility(
                saved, current, allow_world_size_change=False,
                allow_hard_separation_enable=True,
            )

    def test_world_size_change_requires_explicit_acknowledgement(self):
        saved = _metadata(world_size=2)
        current = _metadata(world_size=4)
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

    def test_collective_abi_algorithm_and_tree_are_resume_contracts(self):
        fields = {
            "abi": "ftgspp-pair-host-v2",
            "algorithm": "central_hierarchical_reduce_broadcast",
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
        warnings = validate_resume_compatibility(
            saved, _metadata(), allow_world_size_change=False,
        )
        self.assertTrue(any("topology fingerprint" in value for value in warnings))

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
            + ";mode=off;views_per_rank=1",
        )
        self.assertIn(
            "step=phase_local_optimizer_step;slot=step*world_size+rank",
            SYNC_FLAT_EPOCH_VIEW_SAMPLING,
        )
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

    def test_v6_loader_rejects_a_forged_layout_manifest(self):
        snapshot = CheckpointSnapshot(
            model=_model(),
            optimizer={"moment": np.zeros(4, np.float32)},
            iteration=1,
            metadata={**_metadata(), "layout": {
                "model": {}, "optimizer": {}, "field": {}, "color": {},
            }},
        )
        # publish_checkpoint/save_checkpoint_snapshot would preserve the forged
        # manifest; load must compare it with actual arrays and reject it.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000001.npz"
            from ftgspp.io.checkpoint import save_checkpoint_snapshot
            save_checkpoint_snapshot(path, snapshot)
            with self.assertRaisesRegex(ValueError, "layout manifest"):
                load_checkpoint_full(path)


if __name__ == "__main__":
    unittest.main()
