from __future__ import annotations

import copy
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from ftgspp.config.schema import AppConfig, RelocationConfig
from ftgspp.config.validate import compute_derived_fields, validate_static_config


ROOT = Path(__file__).resolve().parents[1]


def _valid_config() -> AppConfig:
    config = AppConfig()
    config.data.frames.start = 0
    config.data.frames.stop = 10
    config.init.keyframe_stride = 1
    config.init.num_gaussians = 16
    config.train.iterations = 100
    config.train.relocation = RelocationConfig(
        start=0,
        stop=90,
        every=10,
        opacity_threshold=5e-3,
    )
    return config


class StaticConfigValidationTests(unittest.TestCase):
    def test_single_device_profile_defaults_to_complete_epoch_sampling(self):
        profile = OmegaConf.load(ROOT / "configs/distributed/off.yaml")

        self.assertEqual(profile.mode, "off")
        self.assertEqual(profile.sampling_policy, "flat_epoch")

    def test_effective_batch_adam_derives_pk_batch_and_sqrt_lr(self):
        config = copy.deepcopy(_valid_config())
        config.distributed.mode = "sync"
        config.distributed.device_ids = [0, 1]
        config.distributed.accumulation.microsteps = 2
        config.distributed.world_size_lr_scaling = "effective_batch_adam"
        original_lrs = dict(config.train.lrs)

        compute_derived_fields(config)

        self.assertEqual(config.distributed.global_batch_size, 4)
        for name, value in original_lrs.items():
            self.assertAlmostEqual(config.train.lrs[name], 2.0 * value)

    def test_zero_collective_bucket_enables_startup_calibration(self):
        config = copy.deepcopy(_valid_config())
        config.distributed.host_collective.bucket_mb = 0
        validate_static_config(config)
        config.distributed.host_collective.bucket_mb = -1
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            validate_static_config(config)

    def test_valid_relocation_contract_is_accepted(self):
        validate_static_config(_valid_config())

    def test_train_seed_must_be_non_negative(self):
        config = copy.deepcopy(_valid_config())
        config.train.seed = -1
        with self.assertRaisesRegex(ValueError, "train.seed must be non-negative"):
            validate_static_config(config)

    def test_train_seed_must_fit_native_uint32(self):
        config = copy.deepcopy(_valid_config())
        config.train.seed = 0x1_0000_0000
        with self.assertRaisesRegex(ValueError, "native uint32 range"):
            validate_static_config(config)

        config.train.seed = 0xFFFFFFFF - config.train.iterations + 1
        with self.assertRaisesRegex(ValueError, "uint32 phase-seed range"):
            validate_static_config(config)

    def test_invalid_relocation_schedule_is_rejected(self):
        cases = (
            ("every", 0, "every must be greater than zero"),
            ("start", -1, "start must not be negative"),
            ("stop", 0, "stop must be greater than"),
            ("stop", 101, "stop must not exceed"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                config = copy.deepcopy(_valid_config())
                setattr(config.train.relocation, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    validate_static_config(config)

    def test_opacity_threshold_must_be_finite_and_non_degenerate(self):
        for threshold in (float("nan"), float("inf"), -0.1, 0.0, 1.0, 1.1):
            with self.subTest(threshold=threshold):
                config = copy.deepcopy(_valid_config())
                config.train.relocation.opacity_threshold = threshold
                with self.assertRaisesRegex(
                    ValueError, "finite and strictly between zero and one",
                ):
                    validate_static_config(config)

    def test_hierarchical_host_accepts_dual_device_world(self):
        for backend in ("auto", "hierarchical_host"):
            with self.subTest(backend=backend):
                config = copy.deepcopy(_valid_config())
                config.distributed.mode = "sync"
                config.distributed.backend = backend
                config.distributed.device_ids = [0, 1]
                config.distributed.global_batch_size = 2
                validate_static_config(config)

    def test_removed_distributed_backends_are_rejected(self):
        for backend in ("packed_host", "host_mailbox", "device_group"):
            with self.subTest(backend=backend):
                config = copy.deepcopy(_valid_config())
                config.distributed.mode = "sync"
                config.distributed.backend = backend
                config.distributed.device_ids = [0, 1]
                config.distributed.global_batch_size = 2
                with self.assertRaisesRegex(
                    ValueError, "unsupported distributed.backend",
                ):
                    validate_static_config(config)

    def test_k_view_schema_accepts_exact_supported_values(self):
        config = copy.deepcopy(_valid_config())
        self.assertEqual(config.distributed.accumulation.microsteps, 1)
        self.assertEqual(
            config.distributed.clocks.optimizer, "optimizer_steps",
        )
        config.distributed.mode = "sync"
        config.distributed.device_ids = [0, 1]
        config.distributed.accumulation.microsteps = 2
        config.distributed.global_batch_size = 4
        validate_static_config(config)
        config.distributed.accumulation.microsteps = 3
        config.distributed.global_batch_size = 6
        with self.assertRaisesRegex(ValueError, "one of 1, 2, or 4"):
            validate_static_config(config)

    def test_three_or_more_ranks_are_rejected(self):
        config = copy.deepcopy(_valid_config())
        config.distributed.mode = "sync"
        config.distributed.backend = "hierarchical_host"
        config.distributed.device_ids = [0, 1, 2]
        config.distributed.global_batch_size = 3
        config.distributed.host_collective.algorithm = (
            "pair_exchange_gpu_merge"
        )
        with self.assertRaisesRegex(ValueError, "at most two"):
            validate_static_config(config)

    def test_more_than_two_ranks_are_rejected(self):
        config = copy.deepcopy(_valid_config())
        config.distributed.mode = "sync"
        config.distributed.device_ids = list(range(3))
        config.distributed.global_batch_size = 3
        with self.assertRaisesRegex(ValueError, "at most two"):
            validate_static_config(config)


if __name__ == "__main__":
    unittest.main()
