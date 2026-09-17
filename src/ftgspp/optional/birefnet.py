"""Offline BiRefNet mask preparation; Torch is imported only on invocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


PREPARATION_SCHEMA = "ftgspp-birefnet-mask-v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PREPROCESSING_PROFILES = {
    "birefnet_imagenet_square_v1",
    "unit_square_v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_png(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    if not cv2.imwrite(str(temporary), value):
        raise OSError(f"failed to encode mask: {temporary}")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _view_inputs(
    images_root: Path,
    view_manifest: Path | None,
) -> list[tuple[Path, int, int, Path]]:
    """Resolve physical images and logical camera/frame keys.

    Without a manifest the conventional ``camera/numeric-frame`` layout is
    retained.  A preparation manifest removes that naming restriction: each
    row supplies ``camera_id``, ``frame`` and one of ``rgb_path``,
    ``image_path`` or ``path``.  Its paths are relative to the manifest.
    """

    if view_manifest is None:
        result: list[tuple[Path, int, int, Path]] = []
        for image_path in _iter_images(images_root):
            relative = image_path.relative_to(images_root)
            try:
                camera_id = int(relative.parts[-2])
                frame = int(image_path.stem)
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    "non-conventional image names require --view-manifest; "
                    f"cannot infer camera/frame from {image_path}"
                ) from exc
            result.append((image_path, camera_id, frame, relative))
        return result

    view_manifest = view_manifest.expanduser().resolve()
    try:
        payload = json.loads(view_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid preparation view manifest: {view_manifest}") from exc
    rows = payload.get("views") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("preparation view manifest must contain a non-empty views list")
    result = []
    seen: set[tuple[int, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"invalid preparation view row {index}")
        try:
            camera_id = int(row["camera_id"])
            frame = int(row["frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"preparation view row {index} requires camera_id and frame"
            ) from exc
        key = (camera_id, frame)
        if key in seen:
            raise ValueError(f"duplicate preparation camera/frame row: {key}")
        seen.add(key)
        raw_path = next(
            (row[name] for name in ("rgb_path", "image_path", "path")
             if row.get(name) is not None),
            None,
        )
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"preparation view row {index} requires an RGB path")
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            image_path = view_manifest.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        # Output naming is independent of the source filename. The generated
        # mask manifest remains the authoritative mapping used by training.
        relative = Path(str(camera_id)) / f"{frame:06d}{image_path.suffix.lower()}"
        result.append((image_path, camera_id, frame, relative))
    return result


def _view_manifest_metadata(
    view_manifest: Path | None,
) -> dict[tuple[int, int], dict[str, object]]:
    """Retain camera/time/pixel-transform provenance from a view manifest."""

    if view_manifest is None:
        return {}
    path = view_manifest.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("views") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("preparation view manifest must contain a views list")
    retained = (
        "timestamp", "intrinsic", "source_intrinsic", "w2c", "distortion",
        "pixel_transform", "source_to_output", "original_size",
    )
    result: dict[tuple[int, int], dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (int(row["camera_id"]), int(row["frame"]))
        result[key] = {name: row[name] for name in retained if name in row}
    return result


def _local_model_identity(model_name_or_path: str) -> dict[str, object]:
    """Fingerprint a local weight file/directory without touching remote IDs."""

    path = Path(model_name_or_path).expanduser()
    if not path.exists():
        return {"kind": "remote_id", "identifier": model_name_or_path}
    path = path.resolve()
    if path.is_file():
        return {
            "kind": "local_file", "path": str(path), "sha256": _sha256(path),
        }
    candidates = sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in {
            ".safetensors", ".bin", ".pt", ".pth", ".json",
        }
    )
    files = {
        item.relative_to(path).as_posix(): _sha256(item) for item in candidates
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "kind": "local_directory",
        "path": str(path),
        "files": files,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _model_state_sha256(model: Any) -> str:
    """Hash the loaded tensors so a remote identifier resolves to actual weights."""

    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _prediction_tensor(output: Any):
    value = getattr(output, "logits", None)
    if value is None:
        value = output[-1] if isinstance(output, (tuple, list)) else output
    while isinstance(value, (tuple, list)):
        value = value[-1]
    if getattr(value, "ndim", 0) == 3:
        value = value[:, None]
    if getattr(value, "ndim", 0) != 4:
        raise RuntimeError(f"BiRefNet returned an unsupported output shape: {getattr(value, 'shape', None)}")
    return value[:, :1]


def _load_model(model: str, device: str):
    try:
        import torch
        from transformers import AutoModelForImageSegmentation
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "BiRefNet preparation requires the 'birefnet' optional dependencies: "
            "pip install 'ftgspp-vulkan-community[birefnet]'"
        ) from exc
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto" else device
    )
    # Transformers 5+ `from_pretrained` can pick fp16 on CUDA. Mask
    # preparation is written in float32; keep the two sides aligned.
    load_kwargs = {"trust_remote_code": True}
    try:
        instance = AutoModelForImageSegmentation.from_pretrained(
            model, dtype=torch.float32, **load_kwargs,
        )
    except TypeError:
        instance = AutoModelForImageSegmentation.from_pretrained(
            model, torch_dtype=torch.float32, **load_kwargs,
        )
    instance.to(device=resolved_device, dtype=torch.float32).eval()
    return instance, torch, resolved_device


def _infer_probability(
    model,
    torch,
    device: str,
    rgb: np.ndarray,
    resolution: int,
    *,
    output_activation: str,
    preprocessing_profile: str = "birefnet_imagenet_square_v1",
) -> np.ndarray:
    image = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_AREA)
    if preprocessing_profile not in PREPROCESSING_PROFILES:
        raise ValueError(
            f"unsupported BiRefNet preprocessing profile: {preprocessing_profile}"
        )
    tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1)[None] / 255.0
    if preprocessing_profile == "birefnet_imagenet_square_v1":
        mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
        model_input = (tensor - mean) / std
    else:
        model_input = tensor
    with torch.inference_mode():
        prediction = _prediction_tensor(model(model_input))
        if output_activation == "sigmoid_logits":
            probability = prediction.sigmoid()
        elif output_activation == "probability":
            minimum = float(prediction.min().item())
            maximum = float(prediction.max().item())
            if minimum < 0.0 or maximum > 1.0:
                raise RuntimeError(
                    "BiRefNet output_activation=probability received values outside [0,1]: "
                    f"minimum={minimum:.8g} maximum={maximum:.8g}"
                )
            probability = prediction
        else:
            raise ValueError(
                "output_activation must be 'sigmoid_logits' or 'probability'"
            )
        probability = torch.nn.functional.interpolate(
            probability, size=rgb.shape[:2], mode="bilinear", align_corners=False,
        )[0, 0]
    return probability.detach().cpu().numpy().astype(np.float32)


def _mask_statistics(probability: np.ndarray) -> dict[str, object]:
    value = np.asarray(probability, np.float32)
    hard = value >= 0.5
    count, _labels = cv2.connectedComponents(hard.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(hard)
    return {
        "soft_area_fraction": float(np.mean(value)),
        "hard_area_fraction": float(np.mean(hard)),
        "centroid_xy_normalized": (
            None if not len(xs) else [
                float((np.mean(xs) + 0.5) / value.shape[1]),
                float((np.mean(ys) + 0.5) / value.shape[0]),
            ]
        ),
        "connected_components": int(max(0, count - 1)),
        "hard_pixels": int(np.count_nonzero(hard)),
    }


def _qa_anomalies(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Locate abrupt changes for review without declaring them segmentation errors."""

    candidates: list[dict[str, object]] = []
    by_camera: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_camera.setdefault(int(row["camera_id"]), []).append(row)
    for camera, camera_rows in by_camera.items():
        ordered = sorted(camera_rows, key=lambda row: int(row["frame"]))
        transitions: list[tuple[dict[str, object], float, float, int]] = []
        for previous, current in zip(ordered, ordered[1:]):
            prev_stats = previous["statistics"]
            curr_stats = current["statistics"]
            area_delta = abs(
                float(curr_stats["hard_area_fraction"])
                - float(prev_stats["hard_area_fraction"])
            )
            prev_centroid = prev_stats["centroid_xy_normalized"]
            curr_centroid = curr_stats["centroid_xy_normalized"]
            centroid_delta = (
                float(np.linalg.norm(
                    np.asarray(curr_centroid, float) - np.asarray(prev_centroid, float)
                ))
                if prev_centroid is not None and curr_centroid is not None else 0.0
            )
            component_delta = abs(
                int(curr_stats["connected_components"])
                - int(prev_stats["connected_components"])
            )
            transitions.append((current, area_delta, centroid_delta, component_delta))
        if not transitions:
            continue
        areas = np.asarray([item[1] for item in transitions], np.float64)
        centers = np.asarray([item[2] for item in transitions], np.float64)

        def threshold(values: np.ndarray, floor: float) -> float:
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            return max(floor, median + 6.0 * max(mad, 1e-6))

        area_threshold = threshold(areas, 0.08)
        center_threshold = threshold(centers, 0.12)
        for row, area_delta, centroid_delta, component_delta in transitions:
            reasons = []
            if area_delta > area_threshold:
                reasons.append("area_jump")
            if centroid_delta > center_threshold:
                reasons.append("centroid_jump")
            if component_delta >= 3:
                reasons.append("component_count_jump")
            if bool(row.get("empty_foreground")):
                reasons.append("empty_foreground")
            if reasons:
                candidates.append({
                    "camera_id": camera, "frame": int(row["frame"]),
                    "reasons": reasons, "area_delta": area_delta,
                    "centroid_delta": centroid_delta,
                    "component_delta": component_delta,
                })
    return candidates


def _write_qa_contact_sheets(
    overlays: list[tuple[int, int, np.ndarray]], qa_root: Path,
) -> dict[str, list[str]]:
    """Write compact temporal strips and simultaneous multi-view mosaics."""

    def thumbnail(image: np.ndarray, height: int = 160) -> np.ndarray:
        width = max(1, round(image.shape[1] * height / image.shape[0]))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    outputs = {"temporal_strips": [], "multiview_mosaics": []}
    by_camera: dict[int, list[tuple[int, np.ndarray]]] = {}
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    for camera, frame, overlay in overlays:
        by_camera.setdefault(camera, []).append((frame, overlay))
        by_frame.setdefault(frame, []).append((camera, overlay))
    for camera, values in sorted(by_camera.items()):
        ordered = sorted(values)
        for chunk_index, offset in enumerate(range(0, len(ordered), 8)):
            sheet = np.concatenate(
                [thumbnail(image) for _frame, image in ordered[offset:offset + 8]],
                axis=1,
            )
            path = qa_root / "temporal_strips" / f"camera_{camera}_{chunk_index:03d}.png"
            _atomic_png(path, sheet)
            outputs["temporal_strips"].append(str(path))
    for frame, values in sorted(by_frame.items()):
        if len(values) < 2:
            continue
        sheet = np.concatenate(
            [thumbnail(image) for _camera, image in sorted(values)], axis=1,
        )
        path = qa_root / "multiview_mosaics" / f"frame_{frame:06d}.png"
        _atomic_png(path, sheet)
        outputs["multiview_mosaics"].append(str(path))
    return outputs


def prepare_birefnet_masks(
    *,
    images_root: Path,
    output_root: Path,
    model_name_or_path: str = "ZhengPeng7/BiRefNet",
    device: str = "auto",
    resolution: int = 1024,
    raw_probabilities: bool = False,
    qa_count: int = 40,
    overwrite: bool = False,
    view_manifest: Path | None = None,
    output_activation: str = "sigmoid_logits",
    preprocessing_profile: str = "birefnet_imagenet_square_v1",
) -> Path:
    images_root = images_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    scene_root = (
        output_root.parent.parent
        if output_root.parent.name == "masks"
        else output_root.parent
    )
    manifest_path = scene_root / "manifests" / "masks.json"
    inputs = _view_inputs(images_root, view_manifest)
    input_metadata = _view_manifest_metadata(view_manifest)
    if not inputs:
        raise FileNotFoundError(f"no images found below {images_root}")
    completed = output_root / "completed.json"
    if completed.exists() and not overwrite:
        raise FileExistsError(f"mask output is already complete: {completed}")
    if overwrite:
        # Invalidate the old transaction before replacing any individual mask.
        # A crash can therefore never leave a stale completed marker blessing a
        # mixture of old and new files.
        completed.unlink(missing_ok=True)
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if output_activation not in {"sigmoid_logits", "probability"}:
        raise ValueError(
            "output_activation must be 'sigmoid_logits' or 'probability'"
        )
    if preprocessing_profile not in PREPROCESSING_PROFILES:
        raise ValueError(
            f"unsupported BiRefNet preprocessing profile: {preprocessing_profile}"
        )
    model, torch, resolved_device = _load_model(model_name_or_path, device)
    rows: list[dict[str, object]] = []
    qa_indices = set(np.linspace(0, len(inputs) - 1, min(qa_count, len(inputs)), dtype=int).tolist())
    qa_root = scene_root / "qa" / "mask_overlays"
    qa_overlays: list[tuple[int, int, np.ndarray]] = []
    for index, (image_path, camera_id, frame, relative) in enumerate(inputs):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        probability = _infer_probability(
            model, torch, resolved_device, rgb, resolution,
            output_activation=output_activation,
            preprocessing_profile=preprocessing_profile,
        )
        mask_path = (output_root / relative).with_suffix(".png")
        encoded = np.clip(np.rint(probability * 255.0), 0, 255).astype(np.uint8)
        _atomic_png(mask_path, encoded)
        raw_path: Path | None = None
        if raw_probabilities:
            raw_path = (output_root / "raw" / relative).with_suffix(".npy")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = raw_path.with_name(f".{raw_path.name}.{os.getpid()}.tmp.npy")
            np.save(temporary, probability.astype(np.float16))
            os.replace(temporary, raw_path)
        if index in qa_indices:
            contour = cv2.Canny(encoded, 64, 128) > 0
            overlay = bgr.copy()
            overlay[contour] = (0, 0, 255)
            _atomic_png((qa_root / relative).with_suffix(".png"), overlay)
            qa_overlays.append((camera_id, frame, overlay))
        row = {
            "camera_id": camera_id,
            "frame": frame,
            "rgb_path": os.path.relpath(image_path, manifest_path.parent),
            "mask_path": os.path.relpath(mask_path, manifest_path.parent),
            "rgb_sha256": _sha256(image_path),
            "mask_sha256": _sha256(mask_path),
            "original_size": [int(rgb.shape[0]), int(rgb.shape[1])],
            "empty_foreground": bool(encoded.max() == 0),
            "raw_probability_path": (
                os.path.relpath(raw_path, manifest_path.parent) if raw_path else None
            ),
            "statistics": _mask_statistics(probability),
        }
        row.update(input_metadata.get((camera_id, frame), {}))
        rows.append(row)
    qa_products = _write_qa_contact_sheets(qa_overlays, scene_root / "qa")
    qa_summary = {
        "schema": PREPARATION_SCHEMA,
        "interpretation": "review_candidates_not_automatic_errors",
        "views": len(rows),
        "anomaly_candidates": _qa_anomalies(rows),
        "products": {
            name: [os.path.relpath(Path(path), scene_root) for path in paths]
            for name, paths in qa_products.items()
        },
    }
    qa_summary_path = scene_root / "qa" / "mask_summary.json"
    _atomic_json(qa_summary_path, qa_summary)
    model_config = getattr(model, "config", None)
    model_identity = _local_model_identity(model_name_or_path)
    model_identity.update({
        "resolved_name_or_path": str(
            getattr(model_config, "_name_or_path", model_name_or_path)
        ),
        "commit_hash": getattr(model_config, "_commit_hash", None),
        "loaded_state_dict_sha256": _model_state_sha256(model),
    })
    from ftgspp.io.provenance import source_identity
    source = source_identity()
    manifest = {
        "schema": PREPARATION_SCHEMA,
        "completion_marker": os.path.relpath(completed, manifest_path.parent),
        "model": model_name_or_path,
        "model_identity": model_identity,
        "source_identity": {
            "schema": source["schema"],
            "set_sha256": source["set_sha256"],
            "git_patch_sha256": source["git_patch_sha256"],
        },
        "input_manifest": (
            str(view_manifest.expanduser().resolve()) if view_manifest is not None else None
        ),
        "input_manifest_sha256": (
            _sha256(view_manifest.expanduser().resolve())
            if view_manifest is not None else None
        ),
        "inference_resolution": resolution,
        "preprocessing": {
            "profile": preprocessing_profile,
            "resize": "square",
            "interpolation": "opencv_area",
            "channel_order": "RGB",
            "value_range": [0.0, 1.0],
            "normalization": (
                "imagenet_mean_std"
                if preprocessing_profile == "birefnet_imagenet_square_v1"
                else "none"
            ),
        },
        "activation": output_activation,
        "output": "uint8_probability_png",
        "postprocess": {"blur": False, "largest_component": False},
        "foreground_target": "primary dynamic subject and carried objects",
        "manual_revisions": [],
        "qa_summary": os.path.relpath(qa_summary_path, manifest_path.parent),
        "qa_summary_sha256": _sha256(qa_summary_path),
        "views": rows,
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(completed, {
        "schema": PREPARATION_SCHEMA,
        "manifest": os.path.relpath(manifest_path, output_root),
        "manifest_sha256": _sha256(manifest_path),
        "views": len(rows),
    })
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ftgspp prepare masks")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="ZhengPeng7/BiRefNet")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--raw-probabilities", action="store_true")
    parser.add_argument("--qa-count", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output-activation",
        choices=("sigmoid_logits", "probability"),
        default="sigmoid_logits",
        help="explicit model-output convention; never inferred independently per frame",
    )
    parser.add_argument(
        "--preprocessing-profile",
        choices=tuple(sorted(PREPROCESSING_PROFILES)),
        default="birefnet_imagenet_square_v1",
        help="versioned input preprocessing; recorded in the mask manifest",
    )
    parser.add_argument(
        "--view-manifest", type=Path,
        help="optional JSON mapping arbitrary RGB paths to camera_id/frame",
    )
    args = parser.parse_args(argv)
    manifest = prepare_birefnet_masks(
        images_root=args.images,
        output_root=args.output,
        model_name_or_path=args.model,
        device=args.device,
        resolution=args.resolution,
        raw_probabilities=args.raw_probabilities,
        qa_count=args.qa_count,
        overwrite=args.overwrite,
        view_manifest=args.view_manifest,
        output_activation=args.output_activation,
        preprocessing_profile=args.preprocessing_profile,
    )
    print(manifest)
