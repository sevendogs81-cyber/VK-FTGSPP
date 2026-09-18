from types import SimpleNamespace
import hashlib
import json

import numpy as np

from ftgspp.io.export_selection import (
    load_export_selection,
    restore_exported_color_correction,
)
from ftgspp.optional.lpips import resolve_torch_device
from ftgspp.training.runner import (
    holdout_is_better,
    joint_quality_score,
    primary_metric_name,
    snapshot_cadence,
)


def test_snapshot_cadence_skips_zero_and_step_zero():
    assert snapshot_cadence(2000, 2000)
    assert not snapshot_cadence(0, 2000)
    assert not snapshot_cadence(2000, 0)
    assert not snapshot_cadence(1999, 2000)


def test_holdout_compares_psnr_and_ssim_jointly():
    first = {"psnr": 22.4, "ssim": 0.84}
    assert holdout_is_better(first, None)
    # Equal PSNR: higher SSIM still wins.
    assert holdout_is_better({"psnr": 22.4, "ssim": 0.85}, first)
    assert not holdout_is_better({"psnr": 22.4, "ssim": 0.83}, first)
    assert not holdout_is_better({"psnr": 22.4, "ssim": 0.84}, first)
    # Both metrics improve.
    assert holdout_is_better({"psnr": 22.6, "ssim": 0.85}, first)
    # Small PSNR gain does not beat a larger SSIM drop.
    assert not holdout_is_better({"psnr": 22.6, "ssim": 0.83}, first)
    # Large SSIM gain can beat a small PSNR drop; lexicographic PSNR-first
    # used to reject this candidate.
    assert holdout_is_better({"psnr": 22.3, "ssim": 0.99}, first)
    assert not holdout_is_better({}, first)
    assert joint_quality_score(22.4, 0.85) > joint_quality_score(22.4, 0.84)
    assert joint_quality_score(22.6, 0.83) < joint_quality_score(22.4, 0.84)


def test_foreground_joint_score_uses_object_psnr_not_composite_psnr():
    current = {
        "foreground_psnr": 23.0,
        "psnr": 28.0,
        "ssim": 0.90,
    }
    # Composite PSNR is higher, but the joint score uses object PSNR.
    weaker_object = {
        "foreground_psnr": 22.5,
        "psnr": 29.0,
        "ssim": 0.90,
    }
    assert not holdout_is_better(
        weaker_object, current, selector="foreground_psnr",
    )
    stronger_object = {
        "foreground_psnr": 23.2,
        "psnr": 27.0,
        "ssim": 0.91,
    }
    assert holdout_is_better(
        stronger_object, current, selector="foreground_psnr",
    )


def test_primary_metric_name_matches_selector_contract():
    assert primary_metric_name("psnr") == "psnr"
    assert primary_metric_name("foreground_psnr") == "foreground_psnr"


def test_foreground_selector_honors_non_degradation_gates():
    candidate = {
        "foreground_psnr": 30.0,
        "psnr": 25.0,
        "ssim": 0.9,
        "silhouette_iou": 0.81,
        "background_opacity_leakage": 0.04,
    }
    assert holdout_is_better(
        candidate, None, selector="foreground_psnr",
        min_silhouette_iou=0.8, max_background_opacity_leakage=0.05,
    )
    assert not holdout_is_better(
        {**candidate, "silhouette_iou": 0.79}, None,
        selector="foreground_psnr", min_silhouette_iou=0.8,
    )
    assert not holdout_is_better(
        {**candidate, "background_opacity_leakage": 0.06}, None,
        selector="foreground_psnr", max_background_opacity_leakage=0.05,
    )


def test_ranking_metrics_do_not_apply_diagnostic_gates():
    ranking = {
        "foreground_psnr": 30.0,
        "psnr": 25.0,
        "ssim": 0.9,
        "foreground_bundle": "ranking",
    }
    assert holdout_is_better(
        ranking, None, selector="foreground_psnr",
        min_silhouette_iou=0.8, max_background_opacity_leakage=0.05,
        max_foreground_holes=0.1, min_boundary_fscore=0.8,
    )


def test_resolve_torch_device_honors_cpu():
    assert resolve_torch_device("cpu") == "cpu"
    assert resolve_torch_device("CPU") == "cpu"


def test_load_export_selection_returns_empty_when_missing(tmp_path):
    assert load_export_selection(tmp_path) == {}


def test_restore_exported_color_correction(tmp_path):
    artifact = tmp_path / "color.npz"
    np.savez(
        artifact,
        snapshot_step=np.int64(12),
        params=np.ones((2, 4), np.float32),
        camera_ids=np.asarray([1, 3], np.int64),
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "export_selection.json").write_text(
        json.dumps({
            "step": 12,
            "color_correction": {"artifact": "color.npz", "sha256": digest},
        }),
        encoding="utf-8",
    )
    selection = load_export_selection(tmp_path)
    backend = SimpleNamespace(params=None)
    backend.set_color_params = lambda params: setattr(backend, "params", params)
    restored = restore_exported_color_correction(
        backend, tmp_path, selection, SimpleNamespace(cameras=[3, 1]),
    )
    assert restored == digest
    assert backend.params.shape == (2, 4)
