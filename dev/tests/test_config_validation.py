from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from ftgspp.config.schema import AppConfig, RelocationConfig
from ftgspp.config.compose import resolve_config
from ftgspp.config.validate import compute_derived_fields, validate_static_config


ROOT = Path(__file__).resolve().parents[2]


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
    compute_derived_fields(config)
    return config


class StaticConfigValidationTests(unittest.TestCase):
    def test_shipped_examples_resolve_and_final_test_template_is_json(self):
        expected = {
            "foreground_fixed_black.yaml": ("foreground", "composite_v1", "carving"),
            "foreground_point_filtering.yaml": (
                "foreground", "composite_v1", "point_filtering",
            ),
            "foreground_scene.yaml": ("foreground", "composite_v1", "carving"),
            "foreground_visual_hull.yaml": ("foreground", "composite_v1", "carving"),
            "scene.yaml": ("scene", "scene_rgb_v1", "auto"),
        }
        for name, contract in expected.items():
            with self.subTest(example=name):
                path = ROOT / "examples" / name
                config = resolve_config(
                    OmegaConf.load(path), base_dir=path.parent,
                ).config
                self.assertEqual(
                    (
                        config.pipeline.mode,
                        config.pipeline.image_objective,
                        config.init.backend,
                    ),
                    contract,
                )
                if name == "foreground_fixed_black.yaml":
                    self.assertEqual(config.train.background.mode, "fixed")
                    self.assertEqual(
                        list(config.train.background.color), [0.0, 0.0, 0.0],
                    )
        manifest = json.loads(
            (ROOT / "examples" / "final_test_manifest.example.json").read_text(
                encoding="utf-8",
            )
        )
        self.assertEqual(manifest["schema"], "ftgspp-final-test-v2")
        self.assertTrue(manifest["independent_from_selection"])
        self.assertTrue(manifest["views"][0]["rgb_sha256"])
        self.assertTrue(manifest["views"][0]["mask_sha256"])

    def test_foreground_pipeline_mode_applies_one_public_profile(self):
        config = resolve_config(OmegaConf.create({
            "pipeline": {"mode": "foreground"},
            "data": {"root": "scene"},
        }), base_dir=ROOT).config
        self.assertEqual(config.data.masks.source, "external")
        self.assertTrue(config.data.masks.required)
        self.assertEqual(
            config.data.masks.manifest_path,
            (ROOT / "scene/manifests/masks.json").resolve(),
        )
        self.assertEqual(config.init.backend, "carving")
        self.assertEqual(
            config.init.points_path,
            (ROOT / "scene/points_vhull_ibr_stride10").resolve(),
        )
        self.assertTrue(config.init.da3.foreground_only)
        self.assertIsNone(config.init.da3.min_consistent_views)
        self.assertIsNone(config.init.da3.use_sky_mask)
        self.assertEqual(
            config.init.da3.cache_path,
            (ROOT / "scene/points_da3_foreground_stride10").resolve(),
        )
        self.assertEqual(config.pipeline.image_objective, "composite_v1")
        self.assertEqual(config.train.background.mode, "random_solid")
        self.assertEqual(config.pipeline.eval_protocol, "foreground_v1")
        self.assertEqual(config.eval.selector, "foreground_psnr")
        self.assertEqual(
            [list(color) for color in config.eval.background_colors[:1]],
            [[0.0, 0.0, 0.0]],
        )
        self.assertTrue(config.model.marginal_gating)
        self.assertEqual(
            config.train.relocation.score_mode, "gate_included",
        )
        self.assertFalse(config.train.color_correction)
        self.assertFalse(config.train.hard_separation)
        self.assertEqual(config.train.hard_separation_steps, 3000)
        self.assertEqual(list(config.train.hard_relocation_steps), [1500])

    def test_foreground_ranking_background_must_stay_black(self):
        with self.assertRaisesRegex(ValueError, "background_colors\\[0\\]"):
            resolve_config(OmegaConf.create({
                "pipeline": {"mode": "foreground"},
                "data": {"root": "scene"},
                "eval": {"background_colors": [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]},
            }), base_dir=ROOT)

    def test_scene_pipeline_enables_gate_cc_and_hs_defaults(self):
        config = resolve_config(OmegaConf.create({
            "pipeline": {"mode": "scene"},
            "data": {"root": "scene"},
        }), base_dir=ROOT).config
        self.assertTrue(config.model.marginal_gating)
        self.assertEqual(
            config.train.relocation.score_mode, "gate_included",
        )
        self.assertTrue(config.train.color_correction)
        self.assertTrue(config.train.hard_separation)
        self.assertEqual(config.train.hard_separation_steps, 3000)
        self.assertEqual(list(config.train.hard_relocation_steps), [1500])

    def test_explicit_settings_override_scene_pipeline_defaults(self):
        config = resolve_config(OmegaConf.create({
            "pipeline": {"mode": "scene"},
            "data": {"root": "scene"},
            "model": {"marginal_gating": False},
            "train": {
                "color_correction": False,
                "hard_separation": False,
                "relocation": {"score_mode": "default"},
            },
        }), base_dir=ROOT).config
        self.assertFalse(config.model.marginal_gating)
        self.assertEqual(config.train.relocation.score_mode, "default")
        self.assertFalse(config.train.color_correction)
        self.assertFalse(config.train.hard_separation)

    def test_foreground_contract_has_no_version_or_loss_strategy(self):
        config = resolve_config(OmegaConf.create({
            "pipeline": {"mode": "foreground"},
            "data": {"root": "scene", "masks": {"source": "alpha"}},
        }), base_dir=ROOT).config
        self.assertEqual(config.pipeline.image_objective, "composite_v1")
        self.assertEqual(config.pipeline.eval_protocol, "foreground_v1")
        self.assertFalse(hasattr(config.pipeline, "foreground_version"))
        self.assertFalse(hasattr(config.pipeline, "foreground_preset"))
        self.assertFalse(hasattr(config.train, "foreground_loss"))
        for field_name in (
            "selector_min_silhouette_iou",
            "selector_max_background_opacity_leakage",
            "selector_max_foreground_holes",
            "selector_min_boundary_fscore",
        ):
            self.assertTrue(hasattr(config.eval, field_name))

    def test_foreground_da3_route_is_an_explicit_override(self):
        config = resolve_config(OmegaConf.create({
            "pipeline": {"mode": "foreground"},
            "data": {"root": "scene"},
            "init": {"backend": "da3"},
        }), base_dir=ROOT).config
        self.assertEqual(config.init.backend, "da3")
        self.assertTrue(config.init.da3.foreground_only)

    def test_foreground_point_filtering_route_is_explicit(self):
        config = resolve_config(OmegaConf.create({
            "pipeline": {"mode": "foreground"},
            "data": {"root": "scene"},
            "init": {
                "backend": "point_filtering",
                "points_path": "scene/points_vhull_filtered_stride10",
            },
        }), base_dir=ROOT).config
        self.assertEqual(config.init.backend, "point_filtering")
        self.assertTrue(config.init.da3.foreground_only)

    def test_scene_da3_geometry_gates_stay_automatic(self):
        config = resolve_config(OmegaConf.create({
            "data": {"root": "scene"},
        }), base_dir=ROOT).config
        self.assertEqual(config.pipeline.mode, "scene")
        self.assertIsNone(config.init.da3.min_consistent_views)
        self.assertIsNone(config.init.da3.use_sky_mask)

    def test_official_training_preset_matches_freetimegs_plus_plus(self):
        config = resolve_config(OmegaConf.create({
            "data": {"root": "scene"},
        }), base_dir=ROOT).config
        self.assertEqual(config.train.relocation.start, 500)
        self.assertEqual(config.train.relocation.stop, 25000)
        self.assertEqual(config.train.relocation.every, 100)
        self.assertAlmostEqual(config.train.lrs["velocity_model"], 1.6e-3)
        short = resolve_config(OmegaConf.create({
            "data": {"root": "scene"},
            "train": {"iterations": 20000},
        }), base_dir=ROOT).config
        self.assertEqual(short.train.relocation.stop, 20000)

    def test_foreground_profile_keeps_alpha_source_and_fixed_background(self):
        config = resolve_config(OmegaConf.create({
            "pipeline": {"mode": "foreground"},
            "data": {"root": "scene", "masks": {"source": "alpha"}},
            "train": {
                "background": {"mode": "fixed"},
            },
        }), base_dir=ROOT).config
        self.assertEqual(config.data.masks.source, "alpha")
        self.assertIsNone(config.data.masks.manifest_path)
        self.assertEqual(config.pipeline.image_objective, "composite_v1")
        self.assertEqual(config.train.background.mode, "fixed")

    def test_foreground_mode_rejects_disabled_derived_contracts(self):
        config = copy.deepcopy(_valid_config())
        config.pipeline.mode = "foreground"
        with self.assertRaisesRegex(ValueError, "mandatory external or alpha"):
            validate_static_config(config)
        config.data.masks.source = "alpha"
        config.data.masks.required = True
        with self.assertRaisesRegex(ValueError, "explicit init.backend"):
            validate_static_config(config)
        config.init.backend = "da3"
        with self.assertRaisesRegex(ValueError, "foreground-aware"):
            validate_static_config(config)
        config.init.da3.foreground_only = True
        validate_static_config(config)

    def test_scene_rejects_foreground_masks(self):
        config = copy.deepcopy(_valid_config())
        config.data.masks.source = "alpha"
        with self.assertRaisesRegex(ValueError, "full-frame RGB"):
            validate_static_config(config)

    def test_unknown_pipeline_mode_is_rejected(self):
        config = copy.deepcopy(_valid_config())
        config.pipeline.mode = "object-ish"
        with self.assertRaisesRegex(ValueError, "unsupported pipeline.mode"):
            validate_static_config(config)

    def test_explicit_init_duration_is_not_overwritten(self):
        config = copy.deepcopy(_valid_config())
        config.init.duration = 0.1
        compute_derived_fields(config)
        self.assertEqual(config.init.duration, 0.1)

    def test_missing_init_duration_uses_keyframe_interval(self):
        config = copy.deepcopy(_valid_config())
        config.init.duration = None
        config.init.keyframe_stride = 3
        config.data.fps = 30.0
        compute_derived_fields(config)
        self.assertEqual(config.init.duration, 0.1)

    def test_nonfinite_fps_is_rejected_before_derived_temporal_fields(self):
        config = copy.deepcopy(_valid_config())
        config.data.fps = float("inf")
        with self.assertRaisesRegex(ValueError, "data.fps must be finite"):
            compute_derived_fields(config)

    def test_single_device_profile_defaults_to_complete_epoch_sampling(self):
        profile = AppConfig().distributed

        self.assertEqual(profile.mode, "off")
        self.assertEqual(profile.sampling_policy, "flat_epoch")

    def test_zero_collective_bucket_enables_startup_calibration(self):
        config = copy.deepcopy(_valid_config())
        config.distributed.host_collective.bucket_mb = 0
        validate_static_config(config)
        config.distributed.host_collective.bucket_mb = -1
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            validate_static_config(config)

    def test_valid_relocation_contract_is_accepted(self):
        validate_static_config(_valid_config())

    def test_checkpoint_export_defaults(self):
        config = AppConfig()
        self.assertEqual(config.train.checkpoint_interval, 2000)
        self.assertEqual(config.train.log_interval, 2000)
        self.assertTrue(config.train.save_checkpoints)
        self.assertTrue(config.train.export_best)

    def test_train_log_interval_must_be_positive(self):
        config = copy.deepcopy(_valid_config())
        config.train.log_interval = 0
        with self.assertRaisesRegex(ValueError, "log_interval must be greater than zero"):
            validate_static_config(config)

    def test_hard_separation_requires_marginal_gating(self):
        config = copy.deepcopy(_valid_config())
        config.train.hard_separation = True
        with self.assertRaisesRegex(ValueError, "requires model.marginal_gating"):
            validate_static_config(config)
        config.model.marginal_gating = True
        validate_static_config(config)

    def test_invalid_hard_separation_schedule_is_rejected(self):
        cases = (
            (0, [1500], "steps must be greater than zero"),
            (3000, [1500, 1500], "strictly increasing"),
            (3000, [1600, 1500], "strictly increasing"),
            (3000, [0], "inside the hard phase"),
            (3000, [3000], "inside the hard phase"),
        )
        for steps, relocations, message in cases:
            with self.subTest(steps=steps, relocations=relocations):
                config = copy.deepcopy(_valid_config())
                config.train.hard_separation_steps = steps
                config.train.hard_relocation_steps = relocations
                with self.assertRaisesRegex(ValueError, message):
                    validate_static_config(config)

        config = copy.deepcopy(_valid_config())
        config.train.hard_relocation_steps = []
        validate_static_config(config)

        config = copy.deepcopy(_valid_config())
        config.train.hard_relocation_steps = None
        with self.assertRaisesRegex(ValueError, "must be a list"):
            validate_static_config(config)

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

    def test_init_seed_may_be_fixed_independently_from_train_seed(self):
        config = _valid_config()
        config.train.seed = 7
        config.init.seed = 0
        validate_static_config(config)
        config.init.seed = -1
        with self.assertRaisesRegex(ValueError, "init.seed must be non-negative"):
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

    def test_pair_host_accepts_dual_device_world(self):
        for backend in ("auto", "pair_host"):
            with self.subTest(backend=backend):
                config = copy.deepcopy(_valid_config())
                config.distributed.mode = "sync"
                config.distributed.backend = backend
                config.distributed.device_ids = [0, 1]
                validate_static_config(config)

    def test_removed_distributed_backends_are_rejected(self):
        for backend in ("packed_host", "host_mailbox", "device_group"):
            with self.subTest(backend=backend):
                config = copy.deepcopy(_valid_config())
                config.distributed.mode = "sync"
                config.distributed.backend = backend
                config.distributed.device_ids = [0, 1]
                with self.assertRaisesRegex(
                    ValueError, "unsupported distributed.backend",
                ):
                    validate_static_config(config)

    def test_single_view_schema_is_fixed(self):
        config = copy.deepcopy(_valid_config())
        self.assertFalse(hasattr(config.distributed, "accumulation"))
        self.assertEqual(
            config.distributed.clocks.optimizer, "optimizer_steps",
        )
        config.distributed.mode = "sync"
        config.distributed.device_ids = [0, 1]
        validate_static_config(config)

    def test_three_or_more_ranks_are_rejected(self):
        config = copy.deepcopy(_valid_config())
        config.distributed.mode = "sync"
        config.distributed.backend = "pair_host"
        config.distributed.device_ids = [0, 1, 2]
        config.distributed.host_collective.algorithm = (
            "pair_exchange_gpu_merge"
        )
        with self.assertRaisesRegex(ValueError, "at most two"):
            validate_static_config(config)

    def test_more_than_two_ranks_are_rejected(self):
        config = copy.deepcopy(_valid_config())
        config.distributed.mode = "sync"
        config.distributed.device_ids = list(range(3))
        with self.assertRaisesRegex(ValueError, "at most two"):
            validate_static_config(config)

    def test_da3_depth_type_and_target_bounds_are_hard_contracts(self):
        config = copy.deepcopy(_valid_config())
        config.init.da3.depth_type = "ray_distance"
        with self.assertRaisesRegex(ValueError, "camera_z"):
            validate_static_config(config)
        config.init.da3.depth_type = "camera_z"
        config.init.da3.target_bounds = [0, 0, 0, 1, 1, 0]
        with self.assertRaisesRegex(ValueError, "maxima"):
            validate_static_config(config)
        config.init.da3.target_bounds = [-1, -2, -3, 1, 2, 3]
        validate_static_config(config)

if __name__ == "__main__":
    unittest.main()
