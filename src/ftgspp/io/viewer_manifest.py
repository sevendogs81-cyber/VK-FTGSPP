"""Publish rendering metadata beside a final model, after camera initialization."""

from pathlib import Path
from ftgspp.viewer.asset import (
    SCHEMA,
    atomic_json,
    camera_hash,
    inferred_metadata,
    validate_metadata,
)


def write_manifest(model_path, metadata, model_sha256):
    model_path = Path(model_path)
    metadata = validate_metadata(metadata)
    atomic_json(
        model_path.parent / "viewer_manifest.json",
        {
            "schema": SCHEMA,
            "model": {"file": model_path.name, "sha256": model_sha256},
            "camera_sha256": camera_hash(metadata["cameras"]),
            "metadata": metadata,
            "coordinate_convention": "OpenCV w2c; +X right, +Y down, +Z forward; time in training seconds",
        },
    )


def write_training_manifest(model_path, model, dataset, config, model_sha256):
    meta = inferred_metadata(model)
    cameras, seen, times = [], set(), set()
    for view in (*dataset.train_views, *dataset.eval_views):
        times.add(float(view.timestamp))
        # Keep distinct poses/intrinsics, without duplicating static rigs for every time sample.
        key = (view.camera_id, view.w2c.tobytes(), view.intrinsic.tobytes(), view.size)
        if key in seen:
            continue
        seen.add(key)
        cameras.append(
            dict(
                name=f"{view.camera_id} / {view.frame}",
                K=view.intrinsic,
                w2c=view.w2c,
                width=view.size[1],
                height=view.size[0],
                timestamp=float(view.timestamp),
            )
        )
    if cameras:
        meta.update(cameras=cameras, initial_camera=cameras[0])
    if times:
        meta.update(
            sample_times=sorted(times), time_start=min(times), time_end=max(times)
        )
    meta.update(
        fps=float(dataset.fps),
        sh_degree=int(config.init.sh_degree),
        marginal_threshold=float(config.model.marginal_threshold),
        inferred=["bounds"],
        camera_source="actual post-initialization training views",
    )
    write_manifest(model_path, meta, model_sha256)
