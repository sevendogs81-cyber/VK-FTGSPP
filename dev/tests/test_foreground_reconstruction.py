from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import ftgspp.optional.birefnet as birefnet
import ftgspp.optional.visual_hull as visual_hull
from ftgspp.data.scene import Scene, View, _read_mask_manifest, load_final_test_scene
from ftgspp.optional.birefnet import (
    BIREFNET_HR_MODEL_NAME,
    BIREFNET_HR_REVISION,
    _infer_probabilities,
    _view_inputs,
)
from ftgspp.optional.da3 import (
    _conditioned_camera_diagnostics,
    depth_to_points,
    geometric_consistency_filter,
)
from ftgspp.optional.lpips import FinalTestLpipsMetric
from ftgspp.optional.point_color import color_points_ibr
from ftgspp.optional.visual_hull import carve_visual_hull, filter_points_by_visual_hull
from ftgspp.training.backend import NativeVulkanBackend
from ftgspp.training.background import sample_background, sample_rank_backgrounds
from ftgspp.training.foreground_metrics import boundary_fscore, foreground_metrics
from ftgspp.training.objective import (
    color_correct_composite_v1,
    color_correct_composite_vjp,
    composite_v1,
    composite_v1_pixel_vjp,
)
from ftgspp.training.runner import (
    NativeRunner,
    _background_context_audit,
    holdout_is_better,
)


def _gpu_holdout(
    height: int,
    width: int,
    *,
    prediction: np.ndarray,
    reference: np.ndarray,
    ssim: float = 0.0,
    foreground_ssim: float | None = None,
    components: dict[str, np.ndarray] | None = None,
    read_components: bool,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    pred = np.asarray(prediction, np.float32)
    ref = np.asarray(reference, np.float32)
    pixels = height * width
    windows = (height - 10) * (width - 10) if height > 10 and width > 10 else 0
    se = float(np.sum((pred - ref) ** 2, dtype=np.float64))
    object_ssim = float(ssim if foreground_ssim is None else foreground_ssim)
    return {
        "squared_error_sum": se,
        "ssim_sum": float(ssim) * float(windows) * 3.0,
        "foreground_ssim_sum": object_ssim * float(windows) * 3.0,
        "foreground_squared_error_sum": se,
        "foreground_samples": float(pixels),
        "valid_pixels": float(pixels),
        "valid_windows": float(max(windows, 0)),
        "foreground_valid_windows": float(max(windows, 0)),
    }, (components if read_components else None)


def _background_options(**values):
    defaults = dict(
        mode="random_solid", color=[0.1, 0.2, 0.3], seed=7,
        rank_policy="shared", hard_phase="inherit",
    )
    defaults.update(values)
    return SimpleNamespace(**defaults)


def test_background_sampler_replays_and_shared_policy_is_exact():
    options = _background_options()
    first = sample_rank_backgrounds(options, phase="main", logical_step=12, world_size=2)
    second = sample_rank_backgrounds(options, phase="main", logical_step=12, world_size=2)
    assert first == second
    assert first[0].color == first[1].color
    assert first[0].color != sample_background(
        options, phase="main", logical_step=13,
    ).color
    independent = _background_options(rank_policy="independent")
    colors = sample_rank_backgrounds(
        independent, phase="main", logical_step=12, world_size=2,
    )
    assert colors[0].color != colors[1].color


def test_hard_phase_background_override_is_black():
    context = sample_background(
        _background_options(hard_phase="fixed_black"),
        phase="hard_separation", logical_step=99,
    )
    assert context.color == (0.0, 0.0, 0.0)


def test_background_audit_records_exact_rank_phase_and_replay_key():
    contexts = sample_rank_backgrounds(
        _background_options(), phase="hard_separation",
        logical_step=23, world_size=2,
    )
    audit = _background_context_audit(contexts)
    assert audit["background_shared"] is True
    assert audit["background_contexts"][0] == {
        "rank": 0,
        "color": list(contexts[0].color),
        "sampler_version": contexts[0].version,
        "phase": "hard_separation",
        "logical_step": 23,
        "rank_policy": "shared",
        "sampled_rank": None,
    }


def test_composite_v1_transmittance_vjp_matches_finite_difference():
    rng = np.random.default_rng(4)
    foreground = rng.uniform(0.1, 0.4, (2, 3, 3))
    transmittance = rng.uniform(0.2, 0.6, (2, 3))
    reference = rng.uniform(0.1, 0.9, (2, 3, 3))
    mask = rng.uniform(0.0, 1.0, (2, 3))
    background = np.asarray([0.2, 0.4, 0.3])
    upstream = rng.normal(size=(2, 3, 3))
    _df, expected_dt = composite_v1_pixel_vjp(upstream, background)

    def objective(value):
        prediction, _target = composite_v1(
            foreground, value, reference, mask, background,
        )
        return float(np.sum(prediction * upstream))

    epsilon = 1e-6
    numeric = np.zeros_like(transmittance)
    for index in np.ndindex(transmittance.shape):
        plus, minus = transmittance.copy(), transmittance.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric[index] = (objective(plus) - objective(minus)) / (2 * epsilon)
    np.testing.assert_allclose(numeric, expected_dt, atol=1e-8, rtol=1e-7)


def test_composite_color_correction_vjp_matches_finite_difference():
    rng = np.random.default_rng(8)
    foreground = rng.uniform(0.1, 0.7, (2, 2, 3))
    transmittance = rng.uniform(0.1, 0.9, (2, 2))
    matrix = np.eye(3) + rng.normal(0, 0.05, (3, 3))
    bias = np.asarray([0.03, -0.04, 0.02])
    background = np.asarray([0.8, 0.2, 0.5])
    upstream = rng.normal(size=(2, 2, 3))
    gradients = color_correct_composite_vjp(
        foreground, transmittance, upstream, matrix, bias, background,
    )

    def objective(t):
        return float(np.sum(color_correct_composite_v1(
            foreground, t, matrix, bias, background,
        ) * upstream))

    epsilon = 1e-6
    numeric = np.zeros_like(transmittance)
    for index in np.ndindex(transmittance.shape):
        plus, minus = transmittance.copy(), transmittance.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric[index] = (objective(plus) - objective(minus)) / (2 * epsilon)
    np.testing.assert_allclose(numeric, gradients["transmittance"], atol=1e-8, rtol=1e-7)


def test_foreground_final_test_streams_optional_lpips_to_separate_result(
    tmp_path, monkeypatch,
):
    height = width = 64
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[..., :3] = [120, 80, 40]
    rgba[..., 3] = 255
    intrinsic = np.asarray(
        [[50, 0, width / 2], [0, 50, height / 2], [0, 0, 1]],
        np.float32,
    )
    view = View(
        path=tmp_path / "unused.png", frame=0, camera_id=9, timestamp=0.0,
        w2c=np.eye(4, dtype=np.float32), source_intrinsic=intrinsic,
        intrinsic=intrinsic, distortion=np.zeros(5, np.float32),
        size=(height, width), mask_source="alpha",
    )
    dataset = Scene(
        frames=[0], cameras=[9], train_cameras=[], eval_cameras=[9],
        camera_split_source="locked final test", train_views=[],
        eval_views=[view], fps=30.0,
        evaluation_identity={"kind": "final_test", "manifest_sha256": "a" * 64},
    )

    class Backend:
        native = SimpleNamespace()
        def view_load_workers(self):
            return 0
        def view_cache_capacity(self):
            return 1
        def cached_view_image(self, *, is_train, index):
            assert not is_train and index == 0
            return rgba
        def evaluate_holdout_view(
            self, index, *, background, bind_validity, read_components,
        ):
            assert index == 0 and bind_validity and read_components
            rgb = rgba[..., :3].astype(np.float32) / 255.0
            foreground = rgb * np.float32(0.97)
            components = {
                "foreground": foreground,
                "transmittance": np.zeros((height, width), np.float32),
                "reference_image": rgba,
                "validity": np.ones((height, width), np.float32),
            }
            return _gpu_holdout(
                height, width, prediction=foreground, reference=rgb, ssim=0.9,
                components=components, read_components=True,
            )
        def device_metadata(self):
            return {"uuid": "fake-vulkan-device"}

    captured = {"pairs": []}
    class FakeEvaluator:
        device_name = "cpu"
        def __init__(self, **options):
            captured["options"] = options
        def evaluate(self, pairs, *, batch_size):
            captured["pairs"].extend(pairs)
            captured["batch_size"] = batch_size
            return [0.375] * len(pairs)
    runner = object.__new__(NativeRunner)
    runner.config = SimpleNamespace(
        pipeline=SimpleNamespace(
            mode="foreground", eval_protocol="foreground_v1",
        ),
        run=SimpleNamespace(output_path=tmp_path),
        train=SimpleNamespace(color_correction=False),
        eval=SimpleNamespace(
            frame_interval=1,
            background_colors=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            opacity_threshold=0.5,
        ),
    )
    extension = FinalTestLpipsMetric(
        pipeline_mode="foreground", device="cpu", batch_size=2, min_size=32,
        evaluator_factory=FakeEvaluator,
    )
    metrics = runner._evaluate(
        Backend(), dataset, 17, metric_extension=extension,
    )

    assert metrics["lpips"] == pytest.approx(0.375)
    assert metrics["lpips_views"] == 1
    assert metrics["lpips_view_metrics"][0]["lpips"] == pytest.approx(0.375)
    assert metrics["lpips_protocol"]["region"] == "reference_foreground_roi"
    assert metrics["evaluation_identity"]["kind"] == "final_test"
    assert len(captured["pairs"]) == 1
    assert captured["pairs"][0][0].shape == (64, 64, 3)
    assert captured["options"]["device"] == "cpu"
    assert captured["batch_size"] == 2
    assert (tmp_path / "final_test_metrics.jsonl").is_file()
    assert not (tmp_path / "metrics.jsonl").exists()


def test_scene_eval_scores_full_rgb_and_ignores_reference_alpha(tmp_path):
    height = width = 16
    rgba = np.zeros((height, width, 4), np.uint8)
    intrinsic = np.asarray(
        [[20, 0, width / 2], [0, 20, height / 2], [0, 0, 1]],
        np.float32,
    )
    view = View(
        path=tmp_path / "unused.png", frame=0, camera_id=4, timestamp=0.0,
        w2c=np.eye(4, dtype=np.float32), source_intrinsic=intrinsic,
        intrinsic=intrinsic, distortion=np.zeros(5, np.float32),
        size=(height, width), mask_source="none",
    )
    dataset = Scene(
        frames=[0], cameras=[4], train_cameras=[], eval_cameras=[4],
        camera_split_source="test", train_views=[], eval_views=[view], fps=30.0,
    )

    class Backend:
        native = SimpleNamespace()
        def view_load_workers(self):
            return 0
        def view_cache_capacity(self):
            return 1
        def cached_view_image(self, *, is_train, index):
            return rgba
        def evaluate_holdout_view(
            self, index, *, background, bind_validity, read_components,
        ):
            assert not bind_validity and not read_components
            return _gpu_holdout(
                height, width,
                prediction=np.ones((height, width, 3), np.float32),
                reference=np.zeros((height, width, 3), np.float32),
                read_components=False,
            )
        def device_metadata(self):
            return {"uuid": "fake-vulkan-device"}

    runner = object.__new__(NativeRunner)
    runner.config = SimpleNamespace(
        pipeline=SimpleNamespace(mode="scene", eval_protocol="scene_rgb_v1"),
        run=SimpleNamespace(output_path=tmp_path),
        train=SimpleNamespace(color_correction=False),
        eval=SimpleNamespace(
            frame_interval=1, background_colors=[[0.0, 0.0, 0.0]],
            opacity_threshold=0.5,
        ),
    )
    metrics = runner._evaluate(Backend(), dataset, 3)
    assert metrics["protocol"] == "scene_rgb_v1"
    assert metrics["psnr"] == pytest.approx(0.0)
    assert metrics["ssim_variant"] == "SSIM_2"
    assert "foreground_ssim" not in metrics
    assert "full_ssim" not in metrics


def test_scene_final_test_optional_lpips_scores_full_frame(tmp_path):
    height = width = 64
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[..., :3] = 128
    rgba[..., 3] = 17
    intrinsic = np.asarray(
        [[50, 0, width / 2], [0, 50, height / 2], [0, 0, 1]],
        np.float32,
    )
    view = View(
        path=tmp_path / "unused.png", frame=4, camera_id=3, timestamp=0.0,
        w2c=np.eye(4, dtype=np.float32), source_intrinsic=intrinsic,
        intrinsic=intrinsic, distortion=np.zeros(5, np.float32),
        size=(height, width), mask_source="none",
    )
    dataset = Scene(
        frames=[4], cameras=[3], train_cameras=[], eval_cameras=[3],
        camera_split_source="locked final test", train_views=[],
        eval_views=[view], fps=30.0,
        evaluation_identity={"kind": "final_test"},
    )
    prediction = np.full((height, width, 3), 0.4, np.float32)

    class Backend:
        native = SimpleNamespace()
        def view_load_workers(self):
            return 0
        def view_cache_capacity(self):
            return 1
        def evaluate_holdout_view(
            self, index, *, background, bind_validity, read_components,
        ):
            assert index == 0 and not bind_validity and read_components
            components = {
                "foreground": prediction,
                "transmittance": np.zeros((height, width), np.float32),
                "reference_image": rgba,
            }
            return _gpu_holdout(
                height, width, prediction=prediction,
                reference=rgba[..., :3].astype(np.float32) / 255.0,
                components=components, read_components=True,
            )
        def device_metadata(self):
            return {"uuid": "fake-vulkan-device"}

    captured = []
    class FakeEvaluator:
        device_name = "cpu"
        def __init__(self, **_options):
            pass
        def evaluate(self, pairs, *, batch_size):
            assert batch_size == 4
            captured.extend(pairs)
            return [0.2] * len(pairs)

    runner = object.__new__(NativeRunner)
    runner.config = SimpleNamespace(
        pipeline=SimpleNamespace(mode="scene", eval_protocol="scene_rgb_v1"),
        run=SimpleNamespace(output_path=tmp_path),
        train=SimpleNamespace(color_correction=False),
        eval=SimpleNamespace(
            frame_interval=1, background_colors=[[0.0, 0.0, 0.0]],
            opacity_threshold=0.5,
        ),
    )
    extension = FinalTestLpipsMetric(
        pipeline_mode="scene", device="cpu", batch_size=4, min_size=32,
        evaluator_factory=FakeEvaluator,
    )
    metrics = runner._evaluate(
        Backend(), dataset, 11, metric_extension=extension,
    )

    assert metrics["lpips"] == pytest.approx(0.2)
    assert metrics["lpips_views"] == 1
    assert metrics["lpips_protocol"]["region"] == "full_frame"
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0][0], prediction)
    np.testing.assert_allclose(captured[0][1], 128.0 / 255.0)


def test_final_test_lpips_streams_bounded_same_shape_batches():
    calls = []
    class FakeEvaluator:
        device_name = "cpu"
        def __init__(self, **_options):
            pass
        def evaluate(self, pairs, *, batch_size):
            calls.append((len(pairs), batch_size))
            return [float(len(calls))] * len(pairs)

    metric = FinalTestLpipsMetric(
        pipeline_mode="scene", batch_size=2, min_size=32,
        evaluator_factory=FakeEvaluator,
    )
    image = np.full((32, 32, 3), 64, np.uint8)
    components = {
        "foreground": np.full((32, 32, 3), 0.25, np.float32),
        "transmittance": np.zeros((32, 32), np.float32),
        "reference_image": image,
    }
    for index in range(5):
        metric.observe(
            view_index=index,
            view=SimpleNamespace(camera_id=0, frame=index),
            background=np.zeros(3, np.float32),
            components=components,
        )
        assert sum(len(items) for items in metric._buffers.values()) < 2
    result, _timings = metric.finalize(vulkan_metadata={})

    assert calls == [(2, 2), (2, 2), (1, 2)]
    assert result["lpips_views"] == 5
    assert result["lpips"] == pytest.approx((1 + 1 + 2 + 2 + 3) / 5)


def test_foreground_in_loop_eval_is_ranking_only(tmp_path):
    height = width = 16
    rgba = np.full((height, width, 4), 128, np.uint8)
    rgba[..., 3] = 255
    intrinsic = np.asarray(
        [[20, 0, width / 2], [0, 20, height / 2], [0, 0, 1]],
        np.float32,
    )
    view = View(
        path=tmp_path / "unused.png", frame=0, camera_id=2, timestamp=0.0,
        w2c=np.eye(4, dtype=np.float32), source_intrinsic=intrinsic,
        intrinsic=intrinsic, distortion=np.zeros(5, np.float32),
        size=(height, width), mask_source="alpha",
    )
    dataset = Scene(
        frames=[0], cameras=[2], train_cameras=[], eval_cameras=[2],
        camera_split_source="test", train_views=[], eval_views=[view], fps=30.0,
    )
    rgb = rgba[..., :3].astype(np.float32) / 255.0

    class Backend:
        native = SimpleNamespace()
        def view_load_workers(self):
            return 0
        def view_cache_capacity(self):
            return 1
        def cached_view_image(self, *, is_train, index):
            return rgba
        def evaluate_holdout_view(
            self, index, *, background, bind_validity, read_components,
        ):
            assert bind_validity and not read_components
            return _gpu_holdout(
                height, width, prediction=rgb, reference=rgb,
                ssim=0.2, foreground_ssim=0.8, read_components=False,
            )
        def device_metadata(self):
            return {"uuid": "fake-vulkan-device"}

    runner = object.__new__(NativeRunner)
    runner.config = SimpleNamespace(
        pipeline=SimpleNamespace(
            mode="foreground", eval_protocol="foreground_v1",
        ),
        run=SimpleNamespace(output_path=tmp_path),
        train=SimpleNamespace(color_correction=False),
        eval=SimpleNamespace(
            frame_interval=1, background_colors=[[0.0, 0.0, 0.0]],
            opacity_threshold=0.5, selector="foreground_psnr",
        ),
    )
    metrics = runner._evaluate(Backend(), dataset, 9)
    assert metrics["foreground_bundle"] == "ranking"
    assert "foreground_views" not in metrics
    assert "silhouette_iou" not in metrics
    assert metrics["foreground_psnr"] is not None
    assert metrics["ssim"] == pytest.approx(0.8)
    assert metrics["foreground_ssim"] == pytest.approx(0.8)
    assert metrics["full_ssim"] == pytest.approx(0.2)
    assert metrics["ssim_variant"] == "foreground_SSIM_2"
    assert metrics["camera_metrics"]["2"]["ssim"] == pytest.approx(0.8)


def test_foreground_full_validation_keeps_full_ssim_on_fixed_backgrounds(tmp_path):
    height = width = 16
    rgba = np.full((height, width, 4), 128, np.uint8)
    rgba[..., 3] = 255
    intrinsic = np.asarray(
        [[20, 0, width / 2], [0, 20, height / 2], [0, 0, 1]],
        np.float32,
    )
    view = View(
        path=tmp_path / "unused.png", frame=0, camera_id=2, timestamp=0.0,
        w2c=np.eye(4, dtype=np.float32), source_intrinsic=intrinsic,
        intrinsic=intrinsic, distortion=np.zeros(5, np.float32),
        size=(height, width), mask_source="alpha",
    )
    dataset = Scene(
        frames=[0], cameras=[2], train_cameras=[], eval_cameras=[2],
        camera_split_source="test", train_views=[], eval_views=[view], fps=30.0,
    )
    rgb = rgba[..., :3].astype(np.float32) / 255.0

    class Backend:
        native = SimpleNamespace()
        def view_load_workers(self):
            return 0
        def view_cache_capacity(self):
            return 1
        def cached_view_image(self, *, is_train, index):
            return rgba
        def evaluate_holdout_view(
            self, index, *, background, bind_validity, read_components,
        ):
            assert bind_validity and read_components
            np.testing.assert_allclose(background, 0.0)
            components = {
                "foreground": rgb,
                "transmittance": np.zeros((height, width), np.float32),
                "reference_image": rgba,
                "validity": np.ones((height, width), np.float32),
            }
            return _gpu_holdout(
                height, width, prediction=rgb, reference=rgb,
                ssim=0.2, foreground_ssim=0.8,
                components=components, read_components=True,
            )
        def device_metadata(self):
            return {"uuid": "fake-vulkan-device"}

    runner = object.__new__(NativeRunner)
    runner.config = SimpleNamespace(
        pipeline=SimpleNamespace(
            mode="foreground", eval_protocol="foreground_v1",
        ),
        run=SimpleNamespace(output_path=tmp_path),
        train=SimpleNamespace(color_correction=False),
        eval=SimpleNamespace(
            frame_interval=1, background_colors=[[0.0, 0.0, 0.0]],
            opacity_threshold=0.5, selector="foreground_psnr",
        ),
    )
    metrics = runner._evaluate(
        Backend(), dataset, 9, full_validation=True,
    )
    assert metrics["foreground_bundle"] == "diagnostic"
    assert metrics["ssim"] == pytest.approx(0.8)
    assert metrics["foreground_ssim"] == pytest.approx(0.8)
    assert metrics["full_ssim"] == pytest.approx(0.2)
    assert metrics["fixed_backgrounds"][0]["ssim"] == pytest.approx(0.2)
    assert holdout_is_better(
        {
            "foreground_psnr": metrics["foreground_psnr"],
            "foreground_ssim": metrics["foreground_ssim"],
            "ssim": metrics["full_ssim"],
        },
        {
            "foreground_psnr": metrics["foreground_psnr"],
            "foreground_ssim": metrics["full_ssim"],
            "ssim": 0.99,
        },
        selector="foreground_psnr",
    )


def test_native_backend_reuses_bound_eval_reference_and_validity():
    image = np.zeros((12, 14, 4), np.uint8)
    image[..., 3] = 255
    calls = {
        "image_load": 0, "validity_load": 0,
        "image_bind": 0, "validity_bind": 0, "bound_copy": 0,
        "drain": 0,
    }

    class EvalView:
        def load_validity(self):
            calls["validity_load"] += 1
            return np.ones((12, 14), np.float32)

    class Native:
        def bind_dynamic_view_image(self, index, is_train, value):
            assert index == 0 and not is_train and value is image
            calls["image_bind"] += 1
        def bind_dynamic_view_validity(self, index, is_train, value):
            assert index == 0 and not is_train
            assert value.dtype == np.uint8
            calls["validity_bind"] += 1
        def drain_eval(self):
            calls["drain"] += 1
        def begin_eval_val_view(self, index, apply_color_correction):
            assert index == 0 and not apply_color_correction
        def complete_eval_metrics(self, _background):
            return {
                "squared_error_sum": 0.0,
                "ssim_sum": 1.0,
                "foreground_ssim_sum": 1.0,
                "foreground_squared_error_sum": 0.0,
                "foreground_samples": 168.0,
                "valid_pixels": 168.0,
                "valid_windows": 8.0,
                "foreground_valid_windows": 8.0,
            }
        def copy_bound_eval_inputs(self, index):
            assert index == 0
            calls["bound_copy"] += 1
            return image.copy(), np.ones((12, 14), np.uint8)
        def submit_eval_rgb_readback(self):
            pass
        def complete_eval_components(self):
            return np.zeros((12, 14, 4), np.float32)

    backend = object.__new__(NativeVulkanBackend)
    backend.native = Native()
    backend._dataset = SimpleNamespace(eval_views=[EvalView()])
    backend._bound_eval_images = set()
    backend._bound_eval_validity = set()
    backend._eval_background_colors = [[0.0, 0.0, 0.0]]
    backend._eval_batch_prepared = False

    def cached_view_image(*, is_train, index):
        assert not is_train and index == 0
        calls["image_load"] += 1
        return image

    backend.cached_view_image = cached_view_image
    backend.begin_evaluation_batch()
    try:
        for _ in range(2):
            backend.evaluate_holdout_view(
                0, background=[0.0, 0.0, 0.0],
                bind_validity=True, read_components=False,
            )
        _gpu, components = backend.evaluate_holdout_view(
            0, background=[0.0, 0.0, 0.0],
            bind_validity=True, read_components=True,
        )
    finally:
        backend.end_evaluation_batch()

    assert components is not None
    assert components["reference_image"].shape == image.shape
    assert calls == {
        "image_load": 1, "validity_load": 1,
        "image_bind": 1, "validity_bind": 1, "bound_copy": 1,
        "drain": 1,
    }


def test_external_16bit_mask_is_soft_and_rgb_is_not_premultiplied():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rgb_path = root / "rgb.png"
        mask_path = root / "mask.png"
        cv2.imwrite(str(rgb_path), np.full((2, 2, 3), (10, 20, 30), np.uint8))
        cv2.imwrite(str(mask_path), np.asarray([[0, 32768], [65535, 16384]], np.uint16))
        view = View(
            path=rgb_path, frame=0, camera_id=0, timestamp=0.0,
            w2c=np.eye(4, dtype=np.float32),
            source_intrinsic=np.eye(3, dtype=np.float32),
            intrinsic=np.eye(3, dtype=np.float32),
            distortion=np.zeros(5, np.float32), size=(2, 2),
            mask_path=mask_path, mask_source="external",
        )
        rgba = view.load_rgb()
        np.testing.assert_array_equal(rgba[..., :3], np.full((2, 2, 3), (30, 20, 10), np.uint8))
        np.testing.assert_array_equal(rgba[..., 3], np.asarray([[0, 128], [255, 64]], np.uint8))


def test_declared_premultiplied_rgb_is_unmultiplied_once():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        # BGR on disk corresponds to straight RGB=(100, 80, 60) times M=.5.
        cv2.imwrite(str(root / "rgb.png"), np.full((1, 1, 3), (30, 40, 50), np.uint8))
        cv2.imwrite(str(root / "mask.png"), np.full((1, 1), 128, np.uint8))
        view = View(
            path=root / "rgb.png", frame=0, camera_id=0, timestamp=0.0,
            w2c=np.eye(4, dtype=np.float32),
            source_intrinsic=np.eye(3, dtype=np.float32),
            intrinsic=np.eye(3, dtype=np.float32),
            distortion=np.zeros(5, np.float32), size=(1, 1),
            mask_path=root / "mask.png", mask_source="external",
            rgb_encoding="premultiplied",
        )
        rgba = view.load_rgb()
        np.testing.assert_allclose(rgba[0, 0, :3], [100, 80, 60], atol=1)
        assert rgba[0, 0, 3] == 128


def test_birefnet_view_manifest_supports_arbitrary_source_names():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        image = root / "capture" / "left-subject.take.png"
        image.parent.mkdir()
        cv2.imwrite(str(image), np.zeros((2, 3, 3), np.uint8))
        manifest = root / "views.json"
        manifest.write_text(json.dumps({"views": [{
            "camera_id": 17,
            "frame": 42,
            "rgb_path": "capture/left-subject.take.png",
        }]}), encoding="utf-8")

        rows = _view_inputs(root, manifest)

        assert rows == [(image.resolve(), 17, 42, Path("17/000042.png"))]


def test_birefnet_hr_preprocessing_and_batch_restore(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(birefnet, "BIREFNET_HR_RESOLUTION", 4)

    class ConstantModel:
        def __init__(self, value: float):
            self.value = value
            self.input = None

        def __call__(self, value):
            self.input = value.detach().cpu().numpy()
            return torch.full(
                (value.shape[0], 1, value.shape[2], value.shape[3]),
                self.value, device=value.device,
            )

    rgb = np.full((3, 5, 3), 255, np.uint8)
    model = ConstantModel(0.25)
    probability = _infer_probabilities(model, torch, "cpu", [rgb])[0]
    np.testing.assert_allclose(
        probability, 1.0 / (1.0 + np.exp(-0.25)), atol=1e-7,
    )
    expected_input = (
        np.ones((1, 3, 4, 4), np.float32)
        - np.asarray([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
    ) / np.asarray([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
    np.testing.assert_allclose(model.input, expected_input, atol=1e-6)
    batched = _infer_probabilities(
        ConstantModel(0.75), torch, "cpu",
        [np.zeros((2, 3, 3), np.uint8), np.zeros((4, 5, 3), np.uint8)],
    )
    assert [value.shape for value in batched] == [(2, 3), (4, 5)]
    expected = 1.0 / (1.0 + np.exp(-0.75))
    np.testing.assert_allclose(batched[0], expected, atol=1e-7)
    np.testing.assert_allclose(batched[1], expected, atol=1e-7)


def test_birefnet_hr_cuda_oom_batch_fallback(
    tmp_path, monkeypatch,
):
    scene = tmp_path / "scene"
    images = scene / "images" / "0"
    images.mkdir(parents=True)
    for frame in range(2):
        assert cv2.imwrite(
            str(images / f"{frame:06d}.png"),
            np.full((3 + frame, 5, 3), 64 + frame, np.uint8),
        )

    class FakeCuda:
        empty_cache_calls = 0

        @classmethod
        def empty_cache(cls):
            cls.empty_cache_calls += 1

    fake_torch = SimpleNamespace(
        float16=object(), bfloat16=object(), cuda=FakeCuda,
    )
    model = SimpleNamespace(config=SimpleNamespace(
        _name_or_path=BIREFNET_HR_MODEL_NAME, _commit_hash="test-commit",
    ))
    monkeypatch.setattr(
        birefnet, "_load_model",
        lambda *_args, **_kwargs: (model, fake_torch, "cuda:0", "float16"),
    )
    calls = []

    def infer(_model, _torch, _device, rgbs, **kwargs):
        calls.append((len(rgbs), kwargs["precision"]))
        if len(rgbs) > 1:
            raise RuntimeError("CUDA out of memory")
        return [np.full(rgb.shape[:2], 0.75, np.float32) for rgb in rgbs]

    monkeypatch.setattr(birefnet, "_infer_probabilities", infer)
    monkeypatch.setattr(birefnet, "_model_state_sha256", lambda _model: "a" * 64)
    monkeypatch.setattr(
        "ftgspp.io.provenance.source_identity",
        lambda: {
            "schema": "test-source-v1", "set_sha256": "b" * 64,
        },
    )

    manifest_path = birefnet.prepare_birefnet_masks(
        images_root=scene / "images",
        output_root=scene / "masks" / "birefnet_hr",
        batch_size=2,
        qa_count=0,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["model"] == BIREFNET_HR_MODEL_NAME
    assert payload["model_identity"]["requested_revision"] == BIREFNET_HR_REVISION
    assert payload["model_identity"]["verification"] == "resolved_commit"
    assert payload["model_identity"]["loaded_state_dict_sha256"] is None
    assert payload["inference_resolution"] == 2048
    assert payload["preprocessing"]["profile"] == "birefnet_hr_imagenet_square_v1"
    assert payload["preprocessing"]["interpolation"] == "opencv_linear"
    assert payload["inference"] == {
        "device": "cuda:0",
        "precision": "float16",
        "requested_batch_size": 2,
        "minimum_batch_size": 1,
        "maximum_successful_batch_size": 1,
        "cuda_oom_batch_reductions": 1,
    }
    assert calls == [(2, "float16"), (1, "float16"), (1, "float16")]
    assert FakeCuda.empty_cache_calls == 1
    assert len(payload["views"]) == 2
    assert all(Path(row["mask_path"]).suffix == ".png" for row in payload["views"])
    assert payload["performance"]["pipeline"]["input_read_and_sha256_passes"] == 1
    assert payload["performance"]["pipeline"]["mask_encode_and_sha256_passes"] == 1


def test_birefnet_manifest_requires_completed_hashed_matching_content():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rgb = root / "rgb.png"
        mask = root / "mask.png"
        cv2.imwrite(str(rgb), np.zeros((4, 5, 3), np.uint8))
        cv2.imwrite(str(mask), np.full((4, 5), 255, np.uint8))
        manifest = root / "masks.json"
        completed = root / "completed.json"
        payload = {
            "schema": "ftgspp-birefnet-mask-v1",
            "completion_marker": "completed.json",
            "views": [{
                "camera_id": 0, "frame": 0,
                "rgb_path": "rgb.png", "mask_path": "mask.png",
                "rgb_sha256": hashlib.sha256(rgb.read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
                "original_size": [4, 5],
            }],
        }
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="completion"):
            _read_mask_manifest(manifest, root)
        completed.write_text(json.dumps({
            "schema": "ftgspp-birefnet-mask-v1",
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "views": 1,
        }), encoding="utf-8")
        parsed = _read_mask_manifest(manifest, root)
        assert parsed.strict and set(parsed.rows) == {(0, 0)}
        cv2.imwrite(str(mask), np.zeros((4, 5), np.uint8))
        with pytest.raises(ValueError, match="mask_sha256 mismatch"):
            _read_mask_manifest(manifest, root)


def test_da3_conditioned_camera_check_rejects_a_second_coordinate_system():
    conditioned = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    conditioned[1, 0, 3] = -1.0
    predicted = conditioned.copy()
    predicted[0, 1, 3] = -1e-3
    diagnostics = _conditioned_camera_diagnostics(predicted, conditioned)
    assert diagnostics["max_center_error_over_baseline"] < 0.02

    predicted[0, 1, 3] = -0.1
    with pytest.raises(RuntimeError, match="inconsistent"):
        _conditioned_camera_diagnostics(predicted, conditioned)


def test_confidence_percentile_is_computed_inside_foreground():
    depth = np.ones((2, 3), np.float32)
    confidence = np.asarray([[100, 100, 100], [0.1, 0.2, 0.9]], np.float32)
    foreground = np.asarray([[0, 0, 0], [1, 1, 1]], np.float32)
    xyz, _rgb = depth_to_points(
        depth, np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32),
        np.zeros((2, 3, 3), np.uint8), confidence=confidence,
        confidence_percentile=50, foreground_mask=foreground,
    )
    assert len(xyz) == 2


def test_geometric_consistency_distinguishes_support_occlusion_and_conflict():
    points = np.asarray([[0, 0, 1], [1, 0, 2], [0, 0, 0.5]], np.float32)
    depth = np.asarray([[1.0, 1.0, 1.0]], np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = intrinsic[1, 1] = 1.0
    keep, support, conflict = geometric_consistency_filter(
        points, source_index=0, depths=[depth, depth],
        extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        intrinsics=np.repeat(intrinsic[None], 2, axis=0),
        foreground_masks=[np.ones_like(depth), np.ones_like(depth)],
        validities=[np.ones_like(depth), np.ones_like(depth)],
        min_consistent_views=2, relative_tolerance=0.01, absolute_tolerance=0.01,
    )
    assert support[0] == 2 and keep[0]
    assert support[1] == 1 and not keep[1]  # behind the observed surface: occluded
    assert conflict[2] == 1 and not keep[2]  # in front: contradictory


def test_foreground_metrics_expose_background_opacity_leakage():
    mask = np.zeros((16, 16), np.float32)
    mask[5:11, 5:11] = 1.0
    alpha = mask.copy()
    alpha[:2, :] = 0.25
    report = foreground_metrics(
        np.zeros((16, 16, 3), np.float32), 1.0 - alpha,
        np.zeros((16, 16, 3), np.float32), mask,
    )
    assert report["background_opacity_leakage"] > 0.0
    assert report["silhouette_iou"] == 1.0
    assert report["foreground_roi"] == {
        "y0": 4, "y1": 12, "x0": 4, "x1": 12,
        "pixels": 64, "padding_pixels": 1,
        "defined_by": "reference_mask_and_validity",
    }


def test_boundary_fscore_does_not_create_a_boundary_at_invalid_pixels():
    target = np.zeros((24, 24), bool)
    target[4:20, 4:20] = True
    prediction = target.copy()
    # The prediction differs only inside the ignored band. Masking before
    # extracting boundaries would manufacture a vertical contour at x=12.
    prediction[:, 12:15] = False
    validity = np.ones_like(target)
    validity[:, 10:15] = False
    score = boundary_fscore(prediction, target, 1, validity)
    assert score == pytest.approx(1.0)


def test_visual_hull_carves_mask_supported_voxels():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        views = []
        intrinsic = np.asarray([[8, 0, 4], [0, 8, 4], [0, 0, 1]], np.float32)
        for camera in range(2):
            rgb_path = root / f"rgb{camera}.png"
            mask_path = root / f"mask{camera}.png"
            cv2.imwrite(str(rgb_path), np.zeros((8, 8, 3), np.uint8))
            mask = np.zeros((8, 8), np.uint8)
            mask[2:6, 2:6] = 255
            cv2.imwrite(str(mask_path), mask)
            views.append(View(
                path=rgb_path, frame=0, camera_id=camera, timestamp=0,
                w2c=np.eye(4, dtype=np.float32),
                source_intrinsic=intrinsic, intrinsic=intrinsic,
                distortion=np.zeros(5, np.float32), size=(8, 8),
                mask_path=mask_path, mask_source="external",
            ))
        points, diagnostics = carve_visual_hull(
            views, np.asarray([[-0.4, -0.4, 1.5], [0.4, 0.4, 2.5]]),
            resolution=12, min_views=2, support_fraction=1.0, device="cpu",
        )
        assert len(points) > 0
        assert diagnostics["output_count"] == len(points)
        assert diagnostics["device"] == "cpu"


def test_carving_bounds_are_derived_robustly_from_da3_clouds():
    rng = np.random.default_rng(9)
    points = rng.uniform(-1.0, 1.0, (2_000, 3)).astype(np.float32)
    points = np.concatenate((points, np.asarray([[500.0, 0.0, 0.0]], np.float32)))
    bounds, diagnostics = visual_hull._bounds_from_point_clouds([
        (points, np.zeros((len(points), 3), np.uint8)),
    ])
    assert bounds.shape == (2, 3)
    assert np.all(bounds[1] > bounds[0])
    assert bounds[1, 0] < 2.0
    assert diagnostics["source_points"] == len(points)


def test_default_carving_reuses_da3_for_automatic_bounds(tmp_path, monkeypatch):
    cloud_root = tmp_path / "da3"
    cloud_root.mkdir()
    xyz = np.asarray([
        [-1.0, -0.5, 1.0], [1.0, 0.5, 2.0], [0.0, 0.0, 1.5],
    ], np.float32)
    rgb = np.full((len(xyz), 3), 127, np.uint8)

    class Generated(list):
        cache_root = cloud_root
        generated = True

    views = [
        SimpleNamespace(camera_id=camera, frame=0, timestamp=0.0)
        for camera in range(2)
    ]

    class FakeScene:
        frames = [0]
        train_cameras = [0, 1]
        fps = 30.0
        calibration_available = True

        def view(self, _frame, camera):
            return views[camera]

        def views_nearest_timestamp(self, *_args, **_kwargs):
            return views

        def checkpoint_metadata(self):
            return {"fingerprint": "dataset"}

    config = SimpleNamespace(
        init=SimpleNamespace(
            keyframe_stride=1,
            da3=SimpleNamespace(sync_tolerance_fraction=0.25),
        ),
        train=SimpleNamespace(seed=4),
    )
    monkeypatch.setattr(visual_hull, "load_config", lambda _path: config)
    monkeypatch.setattr(
        visual_hull.Scene, "from_config", lambda _config: FakeScene(),
    )
    monkeypatch.setattr(
        "ftgspp.optional.da3.generate_da3_points",
        lambda *_args, **_kwargs: Generated([(xyz, rgb)]),
    )
    monkeypatch.setattr(
        visual_hull, "carve_visual_hull",
        lambda _views, _bounds, **_kwargs: (
            xyz,
            {"device": "cpu", "output_count": len(xyz)},
        ),
    )
    monkeypatch.setattr(
        visual_hull, "color_points_ibr",
        lambda points, *_args, **_kwargs: SimpleNamespace(
            keep=np.ones(len(points), bool),
            rgb=rgb,
            source_count=np.ones(len(points), np.int64),
        ),
    )

    manifest_path = visual_hull.prepare_visual_hull_points(
        tmp_path / "config.yaml", tmp_path / "prepared", None,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["method"] == "carving"
    assert manifest["point_source"]["kind"] == "da3_generated"
    assert manifest["bounds_source"]["kind"] == "point_cloud_quantiles"
    assert manifest["bounds_source"]["point_source"]["path"] == str(cloud_root)


def test_visual_hull_cli_makes_both_geometry_methods_explicit(
    tmp_path, monkeypatch,
):
    calls = []

    def prepare(config, output, bounds, **kwargs):
        calls.append((config, output, bounds, kwargs))
        return tmp_path / "manifest.json"

    monkeypatch.setattr(visual_hull, "prepare_visual_hull_points", prepare)
    visual_hull.main([
        "--config", str(tmp_path / "scene.yaml"),
        "--output", str(tmp_path / "carved"),
    ])
    assert calls[-1][3]["method"] == "carving"
    assert calls[-1][3]["device"] == "auto"
    assert calls[-1][2] is None

    visual_hull.main([
        "--config", str(tmp_path / "scene.yaml"),
        "--output", str(tmp_path / "filtered"),
        "--method", "point_filtering",
    ])
    assert calls[-1][2] is None
    assert calls[-1][3]["method"] == "point_filtering"
    assert calls[-1][3]["input_points"] is None

    with pytest.raises(SystemExit):
        visual_hull.main([
            "--config", str(tmp_path / "scene.yaml"),
            "--output", str(tmp_path / "invalid"),
            "--method", "point_filtering",
            "--bounds", "-1", "-1", "-1", "1", "1", "1",
        ])


def _synthetic_view(
    root: Path, camera: int, rgb: tuple[int, int, int], *,
    w2c: np.ndarray | None = None,
) -> View:
    rgb_path = root / f"color-{camera}.png"
    mask_path = root / f"color-mask-{camera}.png"
    cv2.imwrite(str(rgb_path), np.full((9, 9, 3), rgb[::-1], np.uint8))
    cv2.imwrite(str(mask_path), np.full((9, 9), 255, np.uint8))
    intrinsic = np.asarray([[10, 0, 4], [0, 10, 4], [0, 0, 1]], np.float32)
    return View(
        path=rgb_path, frame=0, camera_id=camera, timestamp=0,
        w2c=np.eye(4, dtype=np.float32) if w2c is None else w2c,
        source_intrinsic=intrinsic, intrinsic=intrinsic,
        distortion=np.zeros(5, np.float32), size=(9, 9),
        mask_path=mask_path, mask_source="external",
    )


def test_visual_hull_cuda_matches_cpu_when_available(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    views = [
        _synthetic_view(tmp_path, camera, (50, 60, 70))
        for camera in range(2)
    ]
    bounds = np.asarray([[-0.2, -0.2, 2.0], [0.2, 0.2, 2.4]])
    cpu, _cpu_diagnostics = carve_visual_hull(
        views, bounds, resolution=8, surface_band=False, device="cpu",
    )
    cuda, cuda_diagnostics = carve_visual_hull(
        views, bounds, resolution=8, surface_band=False, device="cuda",
    )
    np.testing.assert_allclose(cuda, cpu, atol=0.0)
    assert cuda_diagnostics["device"] == "cuda"


def test_point_filter_and_ibr_cuda_match_cpu_when_available(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    views = []
    for camera, value in enumerate((10, 20, 30, 200)):
        w2c = np.eye(4, dtype=np.float32)
        w2c[2, 3] = float(camera)
        views.append(
            _synthetic_view(
                tmp_path, camera, (value, value, value), w2c=w2c,
            )
        )
    points = np.asarray([[0, 0, 4], [0.6, 0, 4], [100, 0, 2]], np.float32)
    cpu_keep, cpu_diagnostics = filter_points_by_visual_hull(
        points, views, device="cpu",
    )
    cuda_keep, cuda_diagnostics = filter_points_by_visual_hull(
        points, views, device="cuda",
    )
    default_keep, default_diagnostics = filter_points_by_visual_hull(
        points, views,
    )
    np.testing.assert_array_equal(cuda_keep, cpu_keep)
    np.testing.assert_array_equal(default_keep, cuda_keep)
    assert default_diagnostics["device"].startswith("cuda")
    assert cuda_diagnostics["visible_histogram"] == cpu_diagnostics["visible_histogram"]
    assert cuda_diagnostics["support_histogram"] == cpu_diagnostics["support_histogram"]
    for mode in ("visibility_robust", "depth_rank_top3"):
        cpu = color_points_ibr(
            points, views, mode=mode, source_search_radius=0, device="cpu",
        )
        cuda = color_points_ibr(
            points, views, mode=mode, source_search_radius=0, device="cuda",
        )
        default = color_points_ibr(
            points, views, mode=mode, source_search_radius=0,
        )
        assert default.device.startswith("cuda")
        np.testing.assert_array_equal(cuda.keep, cpu.keep)
        np.testing.assert_array_equal(default.keep, cuda.keep)
        np.testing.assert_array_equal(cuda.rgb, cpu.rgb)
        np.testing.assert_array_equal(cuda.source_count, cpu.source_count)
        np.testing.assert_allclose(
            cuda.disagreement, cpu.disagreement, atol=1e-5, equal_nan=True,
        )


def test_ibr_depth_rank_top3_and_neighbor_z_buffer_are_explicit():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        views = []
        for camera, value in enumerate((10, 20, 30, 200)):
            w2c = np.eye(4, dtype=np.float32)
            w2c[2, 3] = float(camera)
            views.append(_synthetic_view(root, camera, (value, value, value), w2c=w2c))
        ranked = color_points_ibr(
            np.asarray([[0, 0, 4]], np.float32), views,
            mode="depth_rank_top3", depth_rank_views=3,
            source_search_radius=0,
        )
        assert ranked.keep.tolist() == [True]
        np.testing.assert_array_equal(ranked.rgb[0], [20, 20, 20])

        one_view = views[:1]
        points = np.asarray([[0, 0, 2], [0.6, 0, 4]], np.float32)
        strict = color_points_ibr(
            points, one_view, z_buffer_radius=0, source_search_radius=0,
        )
        robust = color_points_ibr(
            points, one_view, z_buffer_radius=1, source_search_radius=0,
        )
        assert strict.keep.tolist() == [True, True]
        assert robust.keep.tolist() == [True, False]


def test_visual_hull_surface_thickness_and_existing_cloud_filter():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        views = [
            _synthetic_view(root, camera, (50, 60, 70))
            for camera in range(2)
        ]
        bounds = np.asarray([[-0.2, -0.2, 2.0], [0.2, 0.2, 2.4]])
        thin, thin_diag = carve_visual_hull(
            views, bounds, resolution=8, surface_band_layers=1, device="cpu",
        )
        thick, thick_diag = carve_visual_hull(
            views, bounds, resolution=8, surface_band_layers=2, device="cpu",
        )
        assert len(thick) > len(thin)
        assert thick_diag["surface_band_layers"] == 2
        candidates = np.asarray([[0, 0, 2.2], [100, 0, 2.2], [0, 0, -1]], np.float32)
        keep, diagnostics = filter_points_by_visual_hull(candidates, views)
        assert keep.tolist() == [True, False, False]
        assert diagnostics["kept_count"] == 1


def test_coarse_carving_only_projects_sparse_fine_candidates(tmp_path):
    views = [
        _synthetic_view(tmp_path, camera, (50, 60, 70))
        for camera in range(2)
    ]
    mask = np.zeros((9, 9), np.uint8)
    mask[3:6, 3:6] = 255
    for view in views:
        assert cv2.imwrite(str(view.mask_path), mask)
    _points, diagnostics = carve_visual_hull(
        views,
        np.asarray([[-2.0, -2.0, 2.0], [2.0, 2.0, 3.0]]),
        resolution=32,
        coarse_resolution=8,
        surface_band=False,
        min_views=2,
        support_fraction=1.0,
        mask_dilation_pixels=0,
        chunk_points=257,
        device="cpu",
    )
    assert diagnostics["coarse_candidate_count"] < diagnostics["candidate_count"]
    assert (
        diagnostics["evaluated_candidate_count"]
        == diagnostics["coarse_candidate_count"]
    )


def test_visual_hull_rejects_all_points_behind_cameras():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        views = [
            _synthetic_view(root, camera, (0, 0, 0))
            for camera in range(2)
        ]
        with pytest.raises(RuntimeError, match="empty volume"):
            carve_visual_hull(
                views, np.asarray([[-1, -1, -3], [1, 1, -1]], float),
                resolution=8, device="cpu",
            )


def test_locked_final_test_manifest_uses_training_identity_and_explicit_time():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cv2.imwrite(str(root / "rgb.png"), np.full((4, 5, 3), 64, np.uint8))
        cv2.imwrite(str(root / "mask.png"), np.full((4, 5), 255, np.uint8))
        training = SimpleNamespace(
            fps=30.0,
            time_offsets={0: 0.0},
            checkpoint_metadata=lambda: {"fingerprint": "locked-training"},
        )
        manifest = {
            "schema": "ftgspp-final-test-v1",
            "training_dataset_fingerprint": "locked-training",
            "fps": 30.0,
            "rgb_encoding": "straight",
            "cameras": {
                "9": {
                    "w2c": np.eye(4).tolist(),
                    "intrinsic": [[4, 0, 2.5], [0, 4, 2], [0, 0, 1]],
                    "size": [4, 5],
                },
            },
            "views": [{
                "camera_id": 9,
                "frame": 3,
                "timestamp": 0.125,
                "rgb_path": "rgb.png",
                "mask_path": "mask.png",
                "rgb_sha256": hashlib.sha256((root / "rgb.png").read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256((root / "mask.png").read_bytes()).hexdigest(),
                "original_size": [4, 5],
            }],
        }
        path = root / "final_test.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        scene = load_final_test_scene(training, path)

        assert scene.train_views == []
        assert scene.eval_cameras == [9]
        assert scene.eval_views[0].timestamp == 0.125
        np.testing.assert_array_equal(scene.eval_views[0].load_mask(), 1.0)
        assert scene.evaluation_identity["manifest_sha256"]
        assert scene.evaluation_identity["content_sha256"]

        (root / "mask.png").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="hash mismatch"):
            load_final_test_scene(training, path)
        cv2.imwrite(str(root / "mask.png"), np.full((4, 5), 255, np.uint8))

        manifest["training_dataset_fingerprint"] = "wrong"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            load_final_test_scene(training, path)
        except ValueError as exc:
            assert "fingerprint" in str(exc)
        else:  # pragma: no cover - makes the security boundary explicit
            raise AssertionError("mismatched final test identity was accepted")


def test_locked_final_test_rejects_content_used_by_checkpoint_selection(tmp_path):
    source = tmp_path / "selection.png"
    duplicate = tmp_path / "final.png"
    assert cv2.imwrite(
        str(source), np.full((4, 5, 3), 64, np.uint8),
    )
    duplicate.write_bytes(source.read_bytes())
    training = SimpleNamespace(
        fps=30.0,
        time_offsets={},
        train_views=[SimpleNamespace(frame=0, camera_id=0, path=source)],
        eval_views=[],
        checkpoint_metadata=lambda: {"fingerprint": "locked-training"},
    )
    manifest = {
        "schema": "ftgspp-final-test-v2",
        "split": "final_test",
        "independent_from_selection": True,
        "training_dataset_fingerprint": "locked-training",
        "fps": 30.0,
        "mask_source": "alpha",
        "cameras": {
            "9": {
                "w2c": np.eye(4).tolist(),
                "intrinsic": [[4, 0, 2.5], [0, 4, 2], [0, 0, 1]],
                "size": [4, 5],
            },
        },
        "views": [{
            "camera_id": 9,
            "frame": 3,
            "timestamp": 0.1,
            "rgb_path": duplicate.name,
            "rgb_sha256": hashlib.sha256(duplicate.read_bytes()).hexdigest(),
            "original_size": [4, 5],
        }],
    }
    path = tmp_path / "final.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="content overlaps"):
        load_final_test_scene(training, path)


def test_locked_final_test_cli_uses_selected_model_identity_without_mutation(
    tmp_path, monkeypatch,
):
    from ftgspp import cli as cli_module

    run = tmp_path / "run"
    run.mkdir()
    model_path = run / "gaussians.npz"
    model_path.write_bytes(b"locked-model")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    selection_path = run / "export_selection.json"
    selection_path.write_text(json.dumps({
        "step": 7,
        "model_sha256": model_hash,
        "training_dataset_fingerprint": "locked-training",
        # A final-test camera must not consume this training-camera state.
        "color_correction": {"artifact": "must-not-be-opened.npz"},
    }), encoding="utf-8")
    selection_before = selection_path.read_bytes()
    test_manifest = tmp_path / "locked-test.json"
    test_manifest.write_text("{}", encoding="utf-8")
    config = SimpleNamespace(
        run=SimpleNamespace(
            output_path=run, view_rgb_cache=1, view_load_workers=0,
        ),
        pipeline=SimpleNamespace(mode="scene"),
        train=SimpleNamespace(
            iterations=20, color_correction=True,
        ),
        data=SimpleNamespace(test_manifest=None),
    )
    training_scene = SimpleNamespace(
        name="training",
        checkpoint_metadata=lambda: {"fingerprint": "locked-training"},
    )
    locked_scene = SimpleNamespace(evaluation_identity={
        "kind": "final_test",
        "schema": "ftgspp-final-test-v2",
        "manifest_path": str(test_manifest),
        "manifest_sha256": "b" * 64,
    })
    observed = {}

    class Runner:
        def __init__(self, actual_config):
            assert actual_config is config
        def evaluate(self, backend, dataset, step, *, scope, metric_extension=None):
            assert scope == "final_test"
            observed["dataset"] = dataset
            observed["step"] = step
            observed["color_correction"] = config.train.color_correction
            observed["lpips"] = {
                "mode": metric_extension.pipeline_mode,
                "device": metric_extension.device,
                "batch_size": metric_extension.batch_size,
                "min_size": metric_extension.min_size,
            }

    class Backend:
        def __init__(self, device, *, honor_device_env):
            observed["device"] = device
            assert honor_device_env is False
        def configure(self, actual_config, extent):
            assert actual_config is config and extent == 1.0
        def upload_dataset(self, dataset, **options):
            assert dataset is locked_scene
        def upload_model(self, model):
            assert model == "model"
        def device_metadata(self):
            return {"uuid": "test-device"}
        def restore_optimizer(self, _state):
            raise AssertionError("final test must not restore training optimizer")
        def set_color_params(self, _params):
            raise AssertionError("final test must not install learned camera rows")
        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(cli_module, "load_run_config", lambda path: config)
    monkeypatch.setattr("ftgspp.training.evaluation.Evaluator", Runner)
    monkeypatch.setattr("ftgspp.training.evaluation.load_evaluation_scene", lambda config: training_scene)
    monkeypatch.setattr("ftgspp.training.backend.NativeVulkanBackend", Backend)
    monkeypatch.setattr("ftgspp.training.backend.camera_extent", lambda scene: 1.0)
    monkeypatch.setattr(
        "ftgspp.data.scene.load_final_test_scene",
        lambda scene, path: (
            locked_scene
            if scene is training_scene and Path(path) == test_manifest
            else (_ for _ in ()).throw(AssertionError("wrong final-test inputs"))
        ),
    )
    monkeypatch.setattr("ftgspp.io.checkpoint.load_model", lambda path: "model")
    monkeypatch.setattr(
        "ftgspp.io.checkpoint.latest_checkpoint",
        lambda path: (_ for _ in ()).throw(
            AssertionError("final test must not load a training checkpoint")
        ),
    )
    cli_module.main([
        "eval", "--run", str(run), "--device", "2",
        "--test-manifest", str(test_manifest),
        "--lpips", "--lpips-device", "cpu",
        "--lpips-batch-size", "2", "--lpips-min-size", "32",
    ])

    identity = locked_scene.evaluation_identity
    assert observed == {
        "device": 2,
        "dataset": locked_scene,
        "step": 7,
        "color_correction": False,
        "lpips": {
            "mode": "scene", "device": "cpu",
            "batch_size": 2, "min_size": 32,
        },
        "closed": True,
    }
    assert identity["model_sha256"] == model_hash
    assert identity["model_step"] == 7
    assert identity["selection_manifest"] == str(selection_path.resolve())
    assert identity["color_correction"] == "disabled_for_novel_final_test_cameras"
    assert selection_path.read_bytes() == selection_before


def test_lpips_cli_rejects_non_final_holdout_eval(tmp_path, monkeypatch):
    from ftgspp import cli as cli_module

    config = SimpleNamespace(
        run=SimpleNamespace(output_path=tmp_path),
        data=SimpleNamespace(test_manifest=None),
    )
    monkeypatch.setattr(cli_module, "load_run_config", lambda _path: config)
    with pytest.raises(SystemExit) as exc:
        cli_module.main([
            "eval", "--run", str(tmp_path), "--lpips",
        ])
    assert exc.value.code == 2


@pytest.mark.parametrize("quick", [False, True])
def test_holdout_eval_cli_does_not_apply_trained_color_corrector(
    tmp_path, monkeypatch, quick,
):
    from ftgspp import cli as cli_module

    run = tmp_path / "run"
    run.mkdir()
    model_path = run / "gaussians.npz"
    model_path.write_bytes(b"holdout-model")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (run / "export_selection.json").write_text(json.dumps({
        "step": 11,
        "model_sha256": model_hash,
        "color_correction": {"artifact": "must-not-be-opened.npz"},
    }), encoding="utf-8")
    config = SimpleNamespace(
        run=SimpleNamespace(
            output_path=run, view_rgb_cache=1, view_load_workers=0,
        ),
        train=SimpleNamespace(
            iterations=20, color_correction=True,
        ),
        data=SimpleNamespace(test_manifest=None),
    )
    holdout_scene = SimpleNamespace(
        name="holdout",
        eval_views=[object()],
        evaluation_identity=None,
        checkpoint_metadata=lambda: {"fingerprint": "training"},
    )
    observed = {}

    class Runner:
        def __init__(self, actual_config):
            assert actual_config is config
        def evaluate(self, backend, dataset, step, *, scope):
            assert scope == ("candidate" if quick else "validation")
            observed["dataset"] = dataset
            observed["step"] = step
            observed["color_correction"] = config.train.color_correction

    class Backend:
        def __init__(self, device, *, honor_device_env):
            observed["device"] = device
            assert honor_device_env is False
        def configure(self, actual_config, extent):
            assert actual_config is config and extent == 1.0
        def upload_dataset(self, dataset, **options):
            assert dataset is holdout_scene
        def upload_model(self, model):
            assert model == "model"
        def restore_optimizer(self, _state):
            observed["restored_optimizer"] = True
        def set_color_params(self, _params):
            raise AssertionError("hold-out eval must not install learned camera rows")
        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(cli_module, "load_run_config", lambda path: config)
    monkeypatch.setattr("ftgspp.training.evaluation.Evaluator", Runner)
    monkeypatch.setattr("ftgspp.training.evaluation.load_evaluation_scene", lambda config: holdout_scene)
    monkeypatch.setattr("ftgspp.training.backend.NativeVulkanBackend", Backend)
    monkeypatch.setattr("ftgspp.training.backend.camera_extent", lambda scene: 1.0)
    monkeypatch.setattr("ftgspp.io.checkpoint.load_model", lambda path: "model")
    monkeypatch.setattr(
        "ftgspp.io.checkpoint.latest_checkpoint",
        lambda path: (_ for _ in ()).throw(
            AssertionError("hold-out eval must not load a training checkpoint")
        ),
    )
    cli_module.main(["eval", "--run", str(run), "--device", "1"] + (["--quick"] if quick else []))

    assert observed == {
        "device": 1,
        "dataset": holdout_scene,
        "step": 11,
        "color_correction": True,
        "closed": True,
    }
