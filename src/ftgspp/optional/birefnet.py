"""Offline BiRefNet-HR mask preparation; Torch is imported only on invocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

PREPARATION_SCHEMA = "ftgspp-birefnet-mask-v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
BIREFNET_HR_MODEL_NAME = "ZhengPeng7/BiRefNet_HR"
BIREFNET_HR_REVISION = "a7a562f6fd16021180f2f4348f4de003a2d3d1e1"
BIREFNET_HR_RESOLUTION = 2048
BIREFNET_HR_PREPROCESSING = "birefnet_hr_imagenet_square_v1"
INFERENCE_PRECISIONS = {"auto", "float32", "float16", "bfloat16"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_png(path: Path, value: np.ndarray) -> str:
    """Encode once, atomically publish, and return the encoded-byte digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    encoded_ok, encoded = cv2.imencode(".png", value)
    if not encoded_ok:
        raise OSError(f"failed to encode PNG: {path}")
    payload = memoryview(encoded).cast("B")
    with temporary.open("wb") as stream:
        stream.write(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _load_image(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Read, hash, and decode an RGB input in one filesystem pass."""

    payload = path.read_bytes()
    bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), hashlib.sha256(payload).hexdigest()


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
        seen: set[tuple[int, int]] = set()
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
            key = (camera_id, frame)
            if key in seen:
                raise ValueError(
                    f"duplicate input camera/frame from image names: {key}"
                )
            seen.add(key)
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


def _model_state_sha256(model: Any) -> str:
    """Hash the loaded tensors so a remote identifier resolves to actual weights."""

    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        try:
            array = tensor.numpy()
        except TypeError:
            # NumPy has no native bfloat16 representation.
            array = tensor.float().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
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
        raise RuntimeError(
            "BiRefNet-HR returned an unsupported output shape: "
            f"{getattr(value, 'shape', None)}"
        )
    return value[:, :1]


def _load_model(
    device: str,
    *,
    precision: str,
    revision: str = BIREFNET_HR_REVISION,
):
    try:
        import torch
        from transformers import AutoModelForImageSegmentation
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "BiRefNet-HR preparation requires the 'birefnet' optional dependencies: "
            "pip install 'ftgspp-vulkan-community[birefnet]'"
        ) from exc
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto" else device
    )
    resolved_precision, inference_dtype = _resolve_precision(
        torch, resolved_device, precision,
    )
    model_dtype = torch.float32 if inference_dtype is None else inference_dtype
    load_kwargs: dict[str, object] = {"trust_remote_code": True}
    load_kwargs["revision"] = revision
    try:
        instance = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_HR_MODEL_NAME, dtype=model_dtype, **load_kwargs,
        )
    except TypeError:
        instance = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_HR_MODEL_NAME, torch_dtype=model_dtype, **load_kwargs,
        )
    instance.to(device=resolved_device, dtype=model_dtype).eval()
    return instance, torch, resolved_device, resolved_precision


def _resolve_precision(torch, device: str, precision: str) -> tuple[str, object | None]:
    requested = str(precision).lower()
    if requested not in INFERENCE_PRECISIONS:
        raise ValueError(
            f"unsupported BiRefNet-HR inference precision: {precision}; "
            f"expected one of {sorted(INFERENCE_PRECISIONS)}"
        )
    device_type = str(device).split(":", 1)[0]
    resolved = "float16" if requested == "auto" and device_type == "cuda" else (
        "float32" if requested == "auto" else requested
    )
    if device_type != "cuda" and resolved != "float32":
        raise ValueError(
            f"BiRefNet-HR precision={resolved} requires a CUDA device; got {device}"
        )
    dtype = {
        "float32": None,
        "float16": getattr(torch, "float16", None),
        "bfloat16": getattr(torch, "bfloat16", None),
    }[resolved]
    if resolved != "float32" and dtype is None:
        raise RuntimeError(f"installed Torch does not expose {resolved}")
    return resolved, dtype


def _infer_probabilities(
    model,
    torch,
    device: str,
    rgbs: list[np.ndarray],
    *,
    precision: str = "float32",
) -> list[np.ndarray]:
    """Run the fixed BiRefNet-HR recipe and restore every source image shape."""

    if not rgbs:
        return []
    resized = np.stack([
        cv2.resize(
            rgb, (BIREFNET_HR_RESOLUTION, BIREFNET_HR_RESOLUTION),
            interpolation=cv2.INTER_LINEAR,
        )
        for rgb in rgbs
    ])
    nchw = np.ascontiguousarray(resized.transpose(0, 3, 1, 2))
    tensor = torch.from_numpy(nchw)
    non_blocking = str(device).startswith("cuda")
    if non_blocking and hasattr(tensor, "pin_memory"):
        try:
            tensor = tensor.pin_memory()
        except RuntimeError:
            # Pinning may be unavailable in CPU-only Torch builds even when a
            # caller supplies a custom device backend.
            non_blocking = False
    tensor = tensor.to(
        device=device, dtype=torch.float32, non_blocking=non_blocking,
    ) / 255.0
    mean = torch.tensor(
        (0.485, 0.456, 0.406), device=device, dtype=torch.float32,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        (0.229, 0.224, 0.225), device=device, dtype=torch.float32,
    ).view(1, 3, 1, 1)
    model_input = (tensor - mean) / std

    resolved_precision, autocast_dtype = _resolve_precision(
        torch, device, precision,
    )
    device_type = str(device).split(":", 1)[0]
    autocast = (
        torch.autocast(device_type=device_type, dtype=autocast_dtype)
        if resolved_precision != "float32" else nullcontext()
    )
    with torch.inference_mode(), autocast:
        probabilities = _prediction_tensor(model(model_input)).sigmoid()
        if probabilities.shape[0] != len(rgbs):
            raise RuntimeError(
                "BiRefNet-HR output batch does not match its input batch: "
                f"{probabilities.shape[0]} vs {len(rgbs)}"
            )
        shape_groups: dict[tuple[int, int], list[int]] = {}
        for index, rgb in enumerate(rgbs):
            shape_groups.setdefault(tuple(rgb.shape[:2]), []).append(index)
        restored: list[np.ndarray | None] = [None] * len(rgbs)
        for shape, indices in shape_groups.items():
            group = torch.nn.functional.interpolate(
                probabilities[indices].float(), size=shape,
                mode="bilinear", align_corners=False,
            )[:, 0].detach().cpu().numpy().astype(np.float32)
            for local_index, input_index in enumerate(indices):
                restored[input_index] = group[local_index]
    return [value for value in restored if value is not None]


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


def _write_mask_product(
    *,
    absolute_index: int,
    image_path: Path,
    camera_id: int,
    frame: int,
    relative: Path,
    bgr: np.ndarray,
    rgb_sha256: str,
    probability: np.ndarray,
    output_root: Path,
    manifest_root: Path,
    qa_root: Path,
    write_raw: bool,
    write_qa: bool,
    metadata: dict[str, object],
) -> tuple[int, dict[str, object], tuple[int, int, np.ndarray] | None]:
    """Encode one inference result without blocking the next GPU batch."""

    encoded = np.clip(np.rint(probability * 255.0), 0, 255).astype(np.uint8)
    mask_path = (output_root / relative).with_suffix(".png")
    mask_sha256 = _atomic_png(mask_path, encoded)
    raw_path: Path | None = None
    if write_raw:
        raw_path = (output_root / "raw" / relative).with_suffix(".npy")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw_path.with_name(f".{raw_path.name}.{os.getpid()}.tmp.npy")
        np.save(temporary, probability.astype(np.float16))
        os.replace(temporary, raw_path)
    qa_overlay: tuple[int, int, np.ndarray] | None = None
    if write_qa:
        contour = cv2.Canny(encoded, 64, 128) > 0
        overlay = bgr.copy()
        overlay[contour] = (0, 0, 255)
        _atomic_png((qa_root / relative).with_suffix(".png"), overlay)
        qa_overlay = (camera_id, frame, overlay)
    row: dict[str, object] = {
        "camera_id": camera_id,
        "frame": frame,
        "rgb_path": os.path.relpath(image_path, manifest_root),
        "mask_path": os.path.relpath(mask_path, manifest_root),
        "rgb_sha256": rgb_sha256,
        "mask_sha256": mask_sha256,
        "original_size": [int(bgr.shape[0]), int(bgr.shape[1])],
        "empty_foreground": bool(encoded.max() == 0),
        "raw_probability_path": (
            os.path.relpath(raw_path, manifest_root) if raw_path else None
        ),
        "statistics": _mask_statistics(probability),
    }
    row.update(metadata)
    return absolute_index, row, qa_overlay


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
    device: str = "auto",
    batch_size: int = 2,
    precision: str = "auto",
    model_revision: str = BIREFNET_HR_REVISION,
    raw_probabilities: bool = False,
    qa_count: int = 40,
    overwrite: bool = False,
    view_manifest: Path | None = None,
) -> Path:
    started = time.perf_counter()
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
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(qa_count) < 0:
        raise ValueError("qa_count must be nonnegative")
    model, torch, resolved_device, resolved_precision = _load_model(
        device, precision=precision, revision=model_revision,
    )
    qa_indices = set(np.linspace(0, len(inputs) - 1, min(qa_count, len(inputs)), dtype=int).tolist())
    qa_root = scene_root / "qa" / "mask_overlays"
    active_batch_size = min(int(batch_size), len(inputs))
    minimum_batch_size = active_batch_size
    maximum_successful_batch = 0
    oom_reductions = 0
    inference_seconds = 0.0
    index = 0
    input_workers = min(4, max(1, os.cpu_count() or 1), len(inputs))
    output_workers = min(4, max(1, os.cpu_count() or 1), len(inputs))
    prefetch_views = min(len(inputs), max(4, 2 * active_batch_size))
    maximum_pending_outputs = max(4, 2 * active_batch_size)
    rows_by_index: dict[int, dict[str, object]] = {}
    qa_by_index: dict[int, tuple[int, int, np.ndarray]] = {}

    def collect_output(future: Future) -> None:
        output_index, row, overlay = future.result()
        rows_by_index[int(output_index)] = row
        if overlay is not None:
            qa_by_index[int(output_index)] = overlay

    progress = tqdm(total=len(inputs), desc="Predicting foreground masks")
    try:
        with (
            ThreadPoolExecutor(max_workers=input_workers) as input_pool,
            ThreadPoolExecutor(max_workers=output_workers) as output_pool,
        ):
            input_futures: dict[int, Future] = {}
            loaded_inputs: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
            output_futures: list[Future] = []
            next_submit = 0

            def fill_input_prefetch(target: int) -> None:
                nonlocal next_submit
                while next_submit < min(target, len(inputs)):
                    input_futures[next_submit] = input_pool.submit(
                        _load_image, inputs[next_submit][0],
                    )
                    next_submit += 1

            fill_input_prefetch(prefetch_views)
            while index < len(inputs):
                batch_end = min(len(inputs), index + active_batch_size)
                batch_indices = list(range(index, batch_end))
                for input_index in batch_indices:
                    if input_index not in loaded_inputs:
                        loaded_inputs[input_index] = input_futures.pop(input_index).result()
                fill_input_prefetch(batch_end + prefetch_views)
                batch = [inputs[input_index] for input_index in batch_indices]
                loaded = [loaded_inputs[input_index] for input_index in batch_indices]
                rgbs = [item[1] for item in loaded]
                inference_started = time.perf_counter()
                try:
                    probabilities = _infer_probabilities(
                        model, torch, resolved_device, rgbs,
                        precision=resolved_precision,
                    )
                except RuntimeError as exc:
                    message = str(exc).lower()
                    is_cuda_oom = (
                        str(resolved_device).startswith("cuda")
                        and "out of memory" in message
                    )
                    if not is_cuda_oom or active_batch_size <= 1:
                        raise
                    active_batch_size = max(1, active_batch_size // 2)
                    minimum_batch_size = min(minimum_batch_size, active_batch_size)
                    oom_reductions += 1
                    cuda = getattr(torch, "cuda", None)
                    if cuda is not None and hasattr(cuda, "empty_cache"):
                        cuda.empty_cache()
                    continue
                inference_seconds += time.perf_counter() - inference_started
                minimum_batch_size = min(minimum_batch_size, len(batch))
                maximum_successful_batch = max(maximum_successful_batch, len(batch))
                for input_index, entry, loaded_image, probability in zip(
                    batch_indices, batch, loaded, probabilities, strict=True,
                ):
                    image_path, camera_id, frame, relative = entry
                    bgr, _rgb, rgb_sha256 = loaded_image
                    output_futures.append(output_pool.submit(
                        _write_mask_product,
                        absolute_index=input_index,
                        image_path=image_path,
                        camera_id=camera_id,
                        frame=frame,
                        relative=relative,
                        bgr=bgr,
                        rgb_sha256=rgb_sha256,
                        probability=probability,
                        output_root=output_root,
                        manifest_root=manifest_path.parent,
                        qa_root=qa_root,
                        write_raw=bool(raw_probabilities),
                        write_qa=input_index in qa_indices,
                        metadata=input_metadata.get((camera_id, frame), {}),
                    ))
                    del loaded_inputs[input_index]
                while len(output_futures) > maximum_pending_outputs:
                    collect_output(output_futures.pop(0))
                index += len(batch)
                progress.update(len(batch))
            for future in output_futures:
                collect_output(future)
    finally:
        progress.close()
    rows = [rows_by_index[row_index] for row_index in range(len(inputs))]
    qa_overlays = [qa_by_index[row_index] for row_index in sorted(qa_by_index)]
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
    commit_hash = getattr(model_config, "_commit_hash", None)
    model_identity = {
        "kind": "remote_id",
        "identifier": BIREFNET_HR_MODEL_NAME,
        "requested_revision": model_revision,
        "resolved_name_or_path": str(
            getattr(model_config, "_name_or_path", BIREFNET_HR_MODEL_NAME)
        ),
        "commit_hash": commit_hash,
        # A resolved immutable Hub commit already identifies every weight
        # shard.  Hash tensors only for custom/local models without that proof.
        "loaded_state_dict_sha256": (
            None if commit_hash else _model_state_sha256(model)
        ),
        "verification": (
            "resolved_commit" if commit_hash else "loaded_state_dict_sha256"
        ),
    }
    from ftgspp.io.provenance import source_identity
    source = source_identity()
    manifest = {
        "schema": PREPARATION_SCHEMA,
        "completion_marker": os.path.relpath(completed, manifest_path.parent),
        "model": BIREFNET_HR_MODEL_NAME,
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
        "inference_resolution": BIREFNET_HR_RESOLUTION,
        "inference": {
            "device": str(resolved_device),
            "precision": resolved_precision,
            "requested_batch_size": int(batch_size),
            "minimum_batch_size": int(minimum_batch_size),
            "maximum_successful_batch_size": int(maximum_successful_batch),
            "cuda_oom_batch_reductions": int(oom_reductions),
        },
        "preprocessing": {
            "profile": BIREFNET_HR_PREPROCESSING,
            "resize": "square",
            "interpolation": "opencv_linear",
            "channel_order": "RGB",
            "value_range": [0.0, 1.0],
            "normalization": "imagenet_mean_std",
        },
        "activation": "sigmoid_logits",
        "output": "uint8_probability_png",
        "postprocess": {"blur": False, "largest_component": False},
        "foreground_target": "primary dynamic subject and carried objects",
        "manual_revisions": [],
        "performance": {
            "wall_seconds": float(time.perf_counter() - started),
            "inference_seconds": float(inference_seconds),
            "inference_views_per_second": (
                float(len(rows) / inference_seconds)
                if inference_seconds > 0.0 else None
            ),
            "pipeline": {
                "input_workers": int(input_workers),
                "output_workers": int(output_workers),
                "prefetch_views": int(prefetch_views),
                "maximum_pending_outputs": int(maximum_pending_outputs),
                "input_read_and_sha256_passes": 1,
                "mask_encode_and_sha256_passes": 1,
            },
        },
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
    parser.add_argument(
        "--model-revision",
        default=BIREFNET_HR_REVISION,
        help="pinned Hugging Face revision recorded in the mask manifest",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="initial inference batch; CUDA OOM automatically retries with smaller batches",
    )
    parser.add_argument(
        "--precision", choices=tuple(sorted(INFERENCE_PRECISIONS)), default="auto",
        help="auto uses float16 autocast on CUDA and float32 on CPU",
    )
    parser.add_argument("--raw-probabilities", action="store_true")
    parser.add_argument("--qa-count", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--view-manifest", type=Path,
        help="optional JSON mapping arbitrary RGB paths to camera_id/frame",
    )
    args = parser.parse_args(argv)
    manifest = prepare_birefnet_masks(
        images_root=args.images,
        output_root=args.output,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        precision=args.precision,
        raw_probabilities=args.raw_probabilities,
        qa_count=args.qa_count,
        overwrite=args.overwrite,
        view_manifest=args.view_manifest,
    )
    print(manifest)
