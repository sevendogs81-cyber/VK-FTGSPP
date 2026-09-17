from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from omegaconf import OmegaConf

from ftgspp.config.compose import resolve_config
from ftgspp.data.multiview import load_image_manifest
from ftgspp.data.prepare_dataset import main as prepare_dataset


def _write_grid(root: Path) -> None:
    for camera in (2, 7):
        directory = root / str(camera)
        directory.mkdir(parents=True)
        for frame in (3, 4):
            assert cv2.imwrite(
                str(directory / f"{frame:06d}.png"),
                np.full((4, 5, 3), camera+frame, np.uint8),
            )


def test_prepare_dataset_creates_manifest_and_resolvable_starter_config(tmp_path):
    images = tmp_path / "images"
    _write_grid(images)
    prepare_dataset([
        "--images", str(images),
        "--fps", "30",
        "--pipeline", "foreground",
        "--mask-source", "alpha",
        "--eval-cameras", "7",
    ])
    manifest = json.loads(
        (tmp_path / "manifests/dataset.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "ftgspp-dataset-v1"
    assert len(manifest["views"]) == 4
    config_path = tmp_path / "ftgspp.yaml"
    config = resolve_config(
        OmegaConf.load(config_path), base_dir=config_path.parent,
    ).config
    assert config.data.view_manifest == tmp_path / "manifests/dataset.json"
    assert config.data.train_cameras == [2]
    assert config.data.eval_cameras == [7]
    assert config.data.frames.start == 3
    assert config.data.frames.stop == 5


def test_manifest_only_supports_arbitrary_paths_and_explicit_timestamps(tmp_path):
    paths = []
    rows = []
    for frame in (10, 11):
        for camera in (4, 9):
            path = tmp_path / f"arbitrary-{frame}-{camera}.png"
            assert cv2.imwrite(str(path), np.zeros((2, 3, 3), np.uint8))
            paths.append(path)
            rows.append({
                "frame": frame,
                "camera_id": camera,
                "rgb_path": path.name,
                "timestamp": frame/30.0 + camera*1e-4,
            })
    manifest = tmp_path / "views.json"
    manifest.write_text(json.dumps({
        "schema": "ftgspp-dataset-v1", "views": rows,
    }), encoding="utf-8")
    frames, cameras, grid, timestamps = load_image_manifest(
        manifest, start=10, stop=12,
    )
    assert frames == [10, 11]
    assert cameras == [4, 9]
    assert grid == [[paths[0], paths[1]], [paths[2], paths[3]]]
    assert timestamps[(11, 9)] == pytest.approx(11/30.0 + 9e-4)


def test_manifest_rejects_partial_timestamp_and_non_rectangular_grid(tmp_path):
    image = tmp_path / "image.png"
    assert cv2.imwrite(str(image), np.zeros((2, 2, 3), np.uint8))
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({
        "schema": "ftgspp-dataset-v1",
        "views": [
            {"frame": 0, "camera_id": 0, "rgb_path": image.name, "timestamp": 0},
            {"frame": 0, "camera_id": 1, "rgb_path": image.name},
            {"frame": 1, "camera_id": 0, "rgb_path": image.name},
        ],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent camera coverage"):
        load_image_manifest(manifest, start=0, stop=2)
