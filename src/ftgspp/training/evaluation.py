"""Shared evaluation engine for training selection and standalone validation.

Callers explicitly choose candidate, validation, or locked final-test scope.
This module does not depend on the training runner or restore optimizer state.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

import cv2
import numpy as np

from ftgspp.config.schema import AppConfig
from ftgspp.config.validate import validate_resolved_config
from ftgspp.data.scene import Scene
from ftgspp.training.backend import NativeVulkanBackend
from ftgspp.training.distributed import SynchronousBackendGroup
from ftgspp.training.foreground_metrics import foreground_metrics
from ftgspp.training.objective import composite_v1

logger = logging.getLogger("ftgspp.native")


def load_evaluation_scene(config: AppConfig) -> Scene:
    """Load and validate existing evaluation inputs without initializing a run."""
    dataset = Scene.from_config(config)
    config.data.train_cameras = dataset.train_cameras
    config.data.eval_cameras = dataset.eval_cameras
    config.data.camera_split_source = dataset.camera_split_source
    validate_resolved_config(config, dataset.cameras)
    return dataset


class EvaluationMetricExtension(Protocol):
    """Single-device, offline metric hook used by the explicit eval command."""

    requires_components: bool

    def observe(
        self,
        *,
        view_index: int,
        view: object,
        background: np.ndarray,
        components: dict[str, np.ndarray],
    ) -> None: ...

    def finalize(
        self,
        *,
        vulkan_metadata: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, float]]: ...

def _psnr(prediction: np.ndarray, reference: np.ndarray) -> float:
    mse = float(np.mean((prediction-reference)**2))
    return -10.0*np.log10(max(mse, 1e-12))

def _gpu_psnr(squared_error_sum: float, valid_pixels: float) -> float:
    samples = float(valid_pixels) * 3.0
    if samples <= 0.0:
        raise RuntimeError("GPU eval produced no valid PSNR pixels")
    mse = float(squared_error_sum) / samples
    return float(-10.0 * math.log10(max(mse, 1e-12)))

def _optional_metric(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _gpu_ssim(ssim_sum: float, valid_windows: float) -> float | None:
    if float(valid_windows) <= 0.0:
        return None
    return float(ssim_sum) / (float(valid_windows) * 3.0)

def _gpu_foreground_psnr(
    squared_error_sum: float, samples: float,
) -> float | None:
    count = float(samples)
    if count <= 0.0:
        return None
    mse = float(squared_error_sum) / (count * 3.0)
    return float(-10.0 * math.log10(max(mse, 1e-12)))

def _ssim(prediction: np.ndarray, reference: np.ndarray) -> float:
    return _EvalSsimWorkspace().ssim(prediction, reference)

class _EvalSsimWorkspace:
    """Paper SSIM₂ eval with reusable OpenCV/NumPy scratch buffers."""

    def __init__(self) -> None:
        self._bufs: dict[str, np.ndarray] = {}

    def _buf(self, key: str, shape: tuple[int, ...]) -> np.ndarray:
        array = self._bufs.get(key)
        if array is None or array.shape != shape:
            array = np.empty(shape, dtype=np.float32)
            self._bufs[key] = array
        return array

    def ssim(
        self,
        prediction: np.ndarray,
        reference: np.ndarray,
        validity: np.ndarray | None = None,
    ) -> float | None:
        # FreeTimeGS/FreeTimeGS++ evaluate [0, 1] RGB with data_range=2.
        # This does not affect the native training SSIM loss.
        data_range = 2.0
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        shape = prediction.shape
        mu_x = self._buf("mu_x", shape)
        mu_y = self._buf("mu_y", shape)
        sigma_x = self._buf("sigma_x", shape)
        sigma_y = self._buf("sigma_y", shape)
        sigma_xy = self._buf("sigma_xy", shape)
        sq_x = self._buf("sq_x", shape)
        sq_y = self._buf("sq_y", shape)
        cross = self._buf("cross", shape)
        cv2.GaussianBlur(prediction, (11, 11), 1.5, dst=mu_x)
        cv2.GaussianBlur(reference, (11, 11), 1.5, dst=mu_y)
        np.multiply(prediction, prediction, out=sq_x)
        np.multiply(reference, reference, out=sq_y)
        np.multiply(prediction, reference, out=cross)
        cv2.GaussianBlur(sq_x, (11, 11), 1.5, dst=sigma_x)
        cv2.GaussianBlur(sq_y, (11, 11), 1.5, dst=sigma_y)
        cv2.GaussianBlur(cross, (11, 11), 1.5, dst=sigma_xy)
        np.subtract(sigma_x, mu_x * mu_x, out=sigma_x)
        np.subtract(sigma_y, mu_y * mu_y, out=sigma_y)
        np.subtract(sigma_xy, mu_x * mu_y, out=sigma_xy)
        value = (
            (2*mu_x*mu_y+c1)*(2*sigma_xy+c2)
        ) / (
            (mu_x*mu_x+mu_y*mu_y+c1)*(sigma_x+sigma_y+c2)
        )
        if validity is None:
            return float(np.mean(value))
        valid = np.asarray(validity).squeeze() >= 0.5
        if valid.shape != prediction.shape[:2]:
            raise ValueError("SSIM validity shape does not match RGB")
        centers = cv2.erode(
            valid.astype(np.uint8), np.ones((11, 11), np.uint8),
            borderType=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(bool)
        # Training excludes the five-pixel image boundary independently of V.
        centers[:5] = False
        centers[-5:] = False
        centers[:, :5] = False
        centers[:, -5:] = False
        return float(np.mean(value[centers])) if centers.any() else None

def _fixed_background_diagnostics(
    colors: Sequence[Sequence[float]],
    gpu: dict[str, object],
    foreground: np.ndarray,
    transmittance: np.ndarray,
    reference: np.ndarray,
    alpha: np.ndarray,
    validity: np.ndarray,
    ssim_workspace: _EvalSsimWorkspace,
) -> list[dict[str, object]]:
    """Build fixed-background rows from native reductions with a legacy fallback."""

    native_backgrounds = gpu.get("background_metrics")
    native_rows = native_backgrounds if isinstance(native_backgrounds, list) else []
    valid_mask = np.asarray(validity).squeeze() >= 0.5
    valid_pixels = int(np.count_nonzero(valid_mask))
    reports: list[dict[str, object]] = []
    for color_index, raw_color in enumerate(colors):
        color = np.asarray(raw_color, np.float32)
        metric_row = (
            native_rows[color_index]
            if color_index < len(native_rows)
            and isinstance(native_rows[color_index], dict)
            else (gpu if color_index == 0 else None)
        )
        if metric_row is not None:
            squared_error_sum = float(metric_row["squared_error_sum"])
            row_pixels = float(metric_row["valid_pixels"])
            reports.append({
                "background": [float(value) for value in color],
                "psnr": _gpu_psnr(squared_error_sum, row_pixels),
                "ssim": _gpu_ssim(
                    float(metric_row["ssim_sum"]),
                    float(metric_row["valid_windows"]),
                ),
                "valid_pixels": int(row_pixels),
                "squared_error_sum": squared_error_sum,
                "samples": int(row_pixels) * 3,
            })
            continue

        # Compatibility fallback for an older native extension.
        prediction, target = composite_v1(
            foreground, transmittance, reference, alpha, color,
        )
        difference = prediction[valid_mask].astype(np.float64)
        difference -= target[valid_mask]
        np.square(difference, out=difference)
        squared_error_sum = float(np.sum(difference, dtype=np.float64))
        samples = valid_pixels * 3
        mse = squared_error_sum / samples if samples else None
        reports.append({
            "background": [float(value) for value in color],
            "psnr": (
                None if mse is None else (
                    float("inf") if mse == 0.0
                    else float(-10.0 * np.log10(mse))
                )
            ),
            "ssim": ssim_workspace.ssim(prediction, target, validity),
            "valid_pixels": valid_pixels,
            "squared_error_sum": squared_error_sum,
            "samples": samples,
        })
    return reports

def _foreground_roi_diagnostics(
    roi: dict[str, object],
    gpu: dict[str, object],
    foreground: np.ndarray,
    transmittance: np.ndarray,
    reference: np.ndarray,
    alpha: np.ndarray,
    validity: np.ndarray,
    ssim_workspace: _EvalSsimWorkspace,
) -> float | None:
    """Return Vulkan ROI SSIM, with a compatibility CPU fallback."""

    y0, y1 = int(roi["y0"]), int(roi["y1"])
    x0, x1 = int(roi["x0"]), int(roi["x1"])
    roi_ssim = (
        _gpu_ssim(
            float(gpu["roi_ssim_sum"]),
            float(gpu["roi_valid_windows"]),
        )
        if (
            gpu.get("roi_ssim_sum") is not None
            and gpu.get("roi_valid_windows") is not None
        ) else None
    )
    if roi_ssim is None:
        prediction, target = composite_v1(
            foreground[y0:y1, x0:x1],
            transmittance[y0:y1, x0:x1],
            reference[y0:y1, x0:x1],
            alpha[y0:y1, x0:x1],
            np.zeros(3, np.float32),
        )
        roi_ssim = ssim_workspace.ssim(
            prediction, target, validity[y0:y1, x0:x1],
        )
    return roi_ssim

def primary_metric_name(selector: str) -> str:
    """Return the task PSNR that enters the joint hold-out score."""

    return "foreground_psnr" if str(selector) == "foreground_psnr" else "psnr"

def stratified_evaluation_indices(
    views: Sequence[object],
    eligible: Sequence[int],
    max_views: int,
) -> list[int]:
    """Choose a deterministic camera/time-stratified candidate subset."""

    indices = [int(value) for value in eligible]
    limit = int(max_views)
    if limit <= 0 or len(indices) <= limit:
        return indices
    by_camera: dict[int, list[int]] = {}
    for index in indices:
        camera = int(getattr(views[index], "camera_id"))
        by_camera.setdefault(camera, []).append(index)
    camera_ids = sorted(by_camera)
    if limit < len(camera_ids):
        positions = np.linspace(0, len(camera_ids)-1, limit, dtype=np.int64)
        camera_ids = [camera_ids[int(value)] for value in np.unique(positions)]
    base, remainder = divmod(limit, len(camera_ids))
    selected: list[int] = []
    for position, camera in enumerate(camera_ids):
        rows = by_camera[camera]
        quota = min(len(rows), base + int(position < remainder))
        if quota:
            temporal = np.linspace(0, len(rows)-1, quota, dtype=np.int64)
            selected.extend(rows[int(value)] for value in np.unique(temporal))
    if len(selected) < limit:
        chosen = set(selected)
        remaining = [index for index in indices if index not in chosen]
        quota = min(limit-len(selected), len(remaining))
        temporal = np.linspace(0, len(remaining)-1, quota, dtype=np.int64)
        selected.extend(remaining[int(value)] for value in np.unique(temporal))
    return sorted(selected[:limit])


@dataclass
class Evaluator:
    config: AppConfig

    @property
    def output_path(self) -> Path:
        return self.config.run.output_path

    def evaluate(
        self,
        backend: NativeVulkanBackend,
        dataset: Scene,
        step: int,
        *,
        group: SynchronousBackendGroup | None = None,
        wall_equalized: bool = False,
        wall_seconds: float | None = None,
        clocks: dict[str, int | float] | None = None,
        result_filename: str | None = None,
        scope: Literal["candidate", "validation", "final_test"],
        metric_extension: EvaluationMetricExtension | None = None,
    ) -> dict[str, object]:
        if scope not in {"candidate", "validation", "final_test"}:
            raise ValueError(f"unknown evaluation scope: {scope}")
        identity = getattr(dataset, "evaluation_identity", None)
        is_final_test = isinstance(identity, dict) and identity.get("kind") == "final_test"
        if (scope == "final_test") != is_final_test:
            raise ValueError("final_test scope requires a locked final-test dataset, and vice versa")
        if not dataset.eval_views:
            return {}
        if metric_extension is not None and group is not None:
            raise ValueError(
                "offline metric extensions require single-device evaluation"
            )
        evaluation_identity = getattr(dataset, "evaluation_identity", None)
        if metric_extension is not None and (
            not isinstance(evaluation_identity, dict)
            or evaluation_identity.get("kind") != "final_test"
        ):
            raise ValueError(
                "offline metric extensions require a locked final-test dataset"
            )
        full_validation = scope != "candidate"
        eligible = [
            index for index, view in enumerate(dataset.eval_views)
            if (view.frame-dataset.frames[0]) % self.config.eval.frame_interval == 0
        ]
        selected = (
            eligible
            if full_validation else stratified_evaluation_indices(
                dataset.eval_views,
                eligible,
                int(getattr(self.config.eval, "candidate_max_views", 0)),
            )
        )
        if not selected:
            return {}
        if self.config.pipeline.mode == "foreground":
            ranking_color = np.asarray(
                self.config.eval.background_colors[0], np.float32,
            ).reshape(-1)
            if (
                ranking_color.shape != (3,)
                or float(np.max(np.abs(ranking_color))) > 1e-6
            ):
                raise ValueError(
                    "foreground eval requires eval.background_colors[0] to be "
                    "black; foreground PSNR and foreground SSIM share that target"
                )
        eval_begin = time.perf_counter()

        worker_fn = getattr(backend, "view_load_workers", None)
        eval_workers = int(worker_fn()) if callable(worker_fn) else 0
        eval_executor = (
            ThreadPoolExecutor(
                max_workers=eval_workers,
                thread_name_prefix="ftgspp-eval-rgb",
            )
            if eval_workers > 0 else None
        )

        def evaluate_shard(
            rank: int,
            replica: NativeVulkanBackend,
            indices: tuple[int, ...],
        ) -> dict[str, object]:
            sums: dict[str, object] = {
                "psnr_sum": 0.0, "ssim_sum": 0.0,
                "full_ssim_sum": 0.0,
                "count": 0, "ssim_count": 0,
                "full_ssim_count": 0,
                "cameras": {}, "foreground_records": [],
                "backend_stage_seconds": {}, "score_seconds": 0.0,
            }
            indices = tuple(int(index) for index in indices)
            worker_fn = getattr(replica, "view_load_workers", None)
            cache_fn = getattr(replica, "view_cache_capacity", None)
            replica_workers = int(worker_fn()) if callable(worker_fn) else 0
            cache_capacity = int(cache_fn()) if callable(cache_fn) else 1
            default_window = max(
                1, min(max(replica_workers, 1), max(cache_capacity, 1), 16),
            )
            window_raw = os.environ.get(
                "FTGSPP_EVAL_PREFETCH_LOOKAHEAD", str(default_window),
            )
            eval_window = max(1, int(window_raw))
            prefetch_eval = getattr(replica, "prefetch_eval_views", None)
            schedule_eval = getattr(replica, "schedule_eval_views", None)
            evaluate_view = getattr(replica, "evaluate_holdout_view", None)
            if not callable(evaluate_view):
                raise RuntimeError(
                    "hold-out eval requires native GPU evaluate_holdout_view"
                )
            foreground_protocol = self.config.pipeline.mode == "foreground"
            want_diagnostics = foreground_protocol and bool(full_validation)
            need_components = want_diagnostics or bool(
                metric_extension is not None
                and metric_extension.requires_components
            )
            ranking_background = np.asarray(
                self.config.eval.background_colors[0], np.float32,
            )
            ssim_workspace = _EvalSsimWorkspace()

            def score_gpu_view(
                index: int,
                gpu: dict[str, object],
                components: dict[str, np.ndarray] | None,
            ) -> None:
                psnr = _gpu_psnr(
                    float(gpu["squared_error_sum"]),
                    float(gpu["valid_pixels"]),
                )
                ssim = _gpu_ssim(
                    float(gpu["ssim_sum"]), float(gpu["valid_windows"]),
                )
                foreground_ssim = _gpu_ssim(
                    float(gpu["foreground_ssim_sum"]),
                    float(gpu["foreground_valid_windows"]),
                )
                object_psnr = _gpu_foreground_psnr(
                    float(gpu["foreground_squared_error_sum"]),
                    float(gpu["foreground_samples"]),
                )
                selection_ssim = (
                    foreground_ssim if foreground_protocol else ssim
                )
                if not np.isfinite(psnr) or (
                    ssim is not None and not np.isfinite(float(ssim))
                ) or (
                    foreground_ssim is not None
                    and not np.isfinite(float(foreground_ssim))
                ):
                    raise FloatingPointError(
                        "distributed evaluation produced non-finite metrics on "
                        f"rank {rank}, view {index}"
                    )
                sums["psnr_sum"] = float(sums["psnr_sum"]) + psnr
                sums["se_sum"] = float(sums.get("se_sum", 0.0)) + float(
                    gpu["squared_error_sum"]
                )
                sums["valid_pixels"] = float(sums.get("valid_pixels", 0.0)) + float(
                    gpu["valid_pixels"]
                )
                sums["fg_se_sum"] = float(sums.get("fg_se_sum", 0.0)) + float(
                    gpu["foreground_squared_error_sum"]
                )
                sums["fg_samples"] = float(sums.get("fg_samples", 0.0)) + float(
                    gpu["foreground_samples"]
                )
                if selection_ssim is not None:
                    sums["ssim_sum"] = float(sums["ssim_sum"]) + float(
                        selection_ssim
                    )
                    sums["ssim_count"] = int(sums["ssim_count"]) + 1
                if ssim is not None:
                    sums["full_ssim_sum"] = float(
                        sums["full_ssim_sum"]
                    ) + float(ssim)
                    sums["full_ssim_count"] = int(
                        sums["full_ssim_count"]
                    ) + 1
                sums["count"] = int(sums["count"]) + 1
                camera_key = str(dataset.eval_views[index].camera_id)
                cameras = sums["cameras"]
                if not isinstance(cameras, dict):
                    raise RuntimeError(
                        "evaluation camera accumulator is invalid")
                camera = cameras.setdefault(camera_key, {
                    "psnr_sum": 0.0, "ssim_sum": 0.0,
                    "count": 0, "ssim_count": 0,
                })
                camera["psnr_sum"] = float(camera["psnr_sum"]) + psnr
                if selection_ssim is not None:
                    camera["ssim_sum"] = float(
                        camera["ssim_sum"]
                    ) + float(selection_ssim)
                    camera["ssim_count"] = int(camera["ssim_count"]) + 1
                camera["count"] = int(camera["count"]) + 1
                if not want_diagnostics:
                    return
                if components is None:
                    raise RuntimeError("foreground diagnostic eval requires F/T")
                reference_image = components.get("reference_image")
                if reference_image is None:
                    reference_image = replica.cached_view_image(
                        is_train=False, index=index,
                    )
                reference = np.asarray(
                    reference_image[..., :3], dtype=np.float32,
                ) * np.float32(1.0 / 255.0)
                alpha = (
                    np.asarray(reference_image[..., 3], dtype=np.float32)
                    * np.float32(1.0 / 255.0)
                    if reference_image.shape[-1] == 4
                    else np.ones(reference.shape[:2], dtype=np.float32)
                )
                foreground = np.asarray(components["foreground"], np.float32)
                transmittance = np.asarray(
                    components["transmittance"], np.float32,
                )
                validity = components.get("validity")
                if validity is None:
                    validity = dataset.eval_views[index].load_validity()
                report = foreground_metrics(
                    foreground, transmittance, reference, alpha, validity,
                    # Fixed-background PSNR/SSIM are reduced by Vulkan below.
                    # Keep this helper focused on silhouette diagnostics.
                    background_colors=(),
                    opacity_threshold=self.config.eval.opacity_threshold,
                )
                report["fixed_backgrounds"] = _fixed_background_diagnostics(
                    self.config.eval.background_colors,
                    gpu,
                    foreground,
                    transmittance,
                    reference,
                    alpha,
                    validity,
                    ssim_workspace,
                )
                roi_record = report.get("foreground_roi")
                if isinstance(roi_record, dict):
                    report["foreground_roi_ssim"] = _foreground_roi_diagnostics(
                        roi_record,
                        gpu,
                        foreground,
                        transmittance,
                        reference,
                        alpha,
                        validity,
                        ssim_workspace,
                    )
                    report["foreground_roi_background"] = [0.0, 0.0, 0.0]
                else:
                    report["foreground_roi_ssim"] = None
                    report["foreground_roi_background"] = [0.0, 0.0, 0.0]
                report["foreground_psnr"] = object_psnr
                report["foreground_ssim"] = foreground_ssim
                report.update({
                    "view": int(index),
                    "camera": int(dataset.eval_views[index].camera_id),
                    "frame": int(dataset.eval_views[index].frame),
                })
                records = sums["foreground_records"]
                if not isinstance(records, list):
                    raise RuntimeError("foreground evaluation accumulator is invalid")
                records.append(report)

            begin_eval_batch = getattr(replica, "begin_evaluation_batch", None)
            end_eval_batch = getattr(replica, "end_evaluation_batch", None)
            if callable(begin_eval_batch):
                begin_eval_batch()
            try:
                for offset in range(0, len(indices), eval_window):
                    chunk = indices[offset:offset + eval_window]
                    if callable(prefetch_eval):
                        prefetch_eval(
                            chunk, executor=eval_executor,
                        )
                    next_chunk = indices[
                        offset + eval_window:offset + 2 * eval_window
                    ]
                    if (
                        next_chunk
                        and eval_executor is not None
                        and callable(schedule_eval)
                    ):
                        schedule_eval(
                            next_chunk, executor=eval_executor,
                        )
                    for index in chunk:
                        gpu, components = evaluate_view(
                            int(index),
                            background=ranking_background,
                            bind_validity=foreground_protocol,
                            read_components=need_components,
                        )
                        raw_stages = gpu.get("stage_seconds")
                        if isinstance(raw_stages, dict):
                            stage_totals = sums["backend_stage_seconds"]
                            if not isinstance(stage_totals, dict):
                                raise RuntimeError(
                                    "evaluation stage accumulator is invalid"
                                )
                            for name, seconds in raw_stages.items():
                                stage_totals[str(name)] = float(
                                    stage_totals.get(str(name), 0.0)
                                ) + float(seconds)
                        score_started = time.perf_counter()
                        score_gpu_view(int(index), gpu, components)
                        sums["score_seconds"] = float(
                            sums["score_seconds"]
                        ) + (time.perf_counter() - score_started)
                        if metric_extension is not None:
                            if components is None:
                                raise RuntimeError(
                                    "offline metric extension requires rendered components"
                                )
                            metric_extension.observe(
                                view_index=int(index),
                                view=dataset.eval_views[int(index)],
                                background=ranking_background,
                                components=components,
                            )
            finally:
                if callable(end_eval_batch):
                    end_eval_batch()
            return sums

        try:
            if group is not None:
                partials = group.parallel_shards(selected, evaluate_shard)
            else:
                partials = [evaluate_shard(0, backend, tuple(selected))]
        finally:
            if eval_executor is not None:
                eval_executor.shutdown(wait=True, cancel_futures=True)
        count = sum(int(values["count"]) for values in partials)
        if count != len(selected):
            raise RuntimeError(
                f"distributed evaluation reduced {count} views, expected {len(selected)}"
            )
        psnr_sum = sum(float(values["psnr_sum"]) for values in partials)
        ssim_sum = sum(float(values["ssim_sum"]) for values in partials)
        ssim_count = sum(int(values["ssim_count"]) for values in partials)
        full_ssim_sum = sum(
            float(values.get("full_ssim_sum", 0.0)) for values in partials
        )
        full_ssim_count = sum(
            int(values.get("full_ssim_count", 0)) for values in partials
        )
        foreground_protocol = self.config.pipeline.mode == "foreground"
        metrics: dict[str, object] = {
            "step": int(step),
            "psnr": psnr_sum / count,
            "ssim": ssim_sum / ssim_count if ssim_count else None,
            "ssim_variant": (
                "foreground_SSIM_2" if foreground_protocol else "SSIM_2"
            ),
            "ssim_data_range": 2.0,
            "views": count,
            "rank_eval_counts": [int(values["count"]) for values in partials],
            "protocol": str(self.config.pipeline.eval_protocol),
            "evaluation_scope": (
                "full_validation" if full_validation else "candidate_subset"
            ),
            "eligible_views": len(eligible),
            "eval_stage_work_seconds": {
                name: float(sum(
                    float(values.get("backend_stage_seconds", {}).get(name, 0.0))
                    for values in partials
                ))
                for name in sorted({
                    str(name)
                    for values in partials
                    for name in values.get("backend_stage_seconds", {})
                })
            },
        }
        if foreground_protocol:
            metrics["foreground_ssim"] = metrics["ssim"]
            metrics["full_ssim"] = (
                full_ssim_sum / full_ssim_count
                if full_ssim_count else None
            )
        metrics["eval_stage_work_seconds"]["cpu_score_and_diagnostics"] = sum(
            float(values.get("score_seconds", 0.0)) for values in partials
        )
        evaluation_identity = getattr(dataset, "evaluation_identity", None)
        if evaluation_identity is not None:
            metrics["evaluation_identity"] = dict(evaluation_identity)
        if self.config.pipeline.mode == "foreground":
            se = sum(float(values.get("se_sum", 0.0)) for values in partials)
            pixels = sum(float(values.get("valid_pixels", 0.0)) for values in partials)
            fg_se = sum(float(values.get("fg_se_sum", 0.0)) for values in partials)
            fg_n = sum(float(values.get("fg_samples", 0.0)) for values in partials)
            if pixels > 0.0:
                metrics["psnr"] = _gpu_psnr(se, pixels)
            metrics["foreground_psnr"] = _gpu_foreground_psnr(fg_se, fg_n)
            metrics["foreground_samples"] = int(fg_n)
            foreground_records = [
                record
                for values in partials
                for record in values.get("foreground_records", [])
            ]
            metrics["foreground_bundle"] = (
                "diagnostic" if foreground_records else "ranking"
            )
            if foreground_records:
                metrics["foreground_views"] = foreground_records
                scalar_keys = (
                    "foreground_psnr", "foreground_ssim", "foreground_roi_ssim",
                    "silhouette_iou", "soft_silhouette_iou",
                    "soft_silhouette_dice",
                    "boundary_fscore", "background_opacity_leakage",
                    "foreground_holes", "boundary_mask_mae", "mask_mae",
                )
                for key in scalar_keys:
                    samples = [
                        float(record[key]) for record in foreground_records
                        if record.get(key) is not None and np.isfinite(float(record[key]))
                    ]
                    metrics[f"{key}_frame_mean"] = (
                        float(np.mean(samples)) if samples else None
                    )
                    metrics[f"{key}_views"] = len(samples)
                def weighted_metric(
                    sum_key: str, count_key: str,
                ) -> tuple[float | None, int]:
                    sample_count = sum(
                        int(record.get(count_key, 0))
                        for record in foreground_records
                    )
                    if sample_count <= 0:
                        return None, 0
                    total = sum(
                        float(record.get(sum_key, 0.0))
                        for record in foreground_records
                    )
                    return float(total / sample_count), sample_count

                for key, sum_key, count_key in (
                    (
                        "background_opacity_leakage",
                        "background_opacity_sum", "background_samples",
                    ),
                    (
                        "foreground_holes",
                        "foreground_holes_sum", "foreground_interior_samples",
                    ),
                    (
                        "boundary_mask_mae",
                        "boundary_mask_abs_error_sum", "boundary_samples",
                    ),
                    ("mask_mae", "mask_abs_error_sum", "valid_samples"),
                ):
                    metrics[key], metrics[f"{key}_samples"] = weighted_metric(
                        sum_key, count_key,
                    )
                intersection = sum(
                    int(record.get("silhouette_intersection", 0))
                    for record in foreground_records
                )
                union = sum(
                    int(record.get("silhouette_union", 0))
                    for record in foreground_records
                )
                metrics["silhouette_iou"] = (
                    float(intersection / union) if union else None
                )
                metrics["silhouette_intersection"] = intersection
                metrics["silhouette_union"] = union
                soft_intersection = sum(
                    float(record.get("soft_silhouette_intersection", 0.0))
                    for record in foreground_records
                )
                soft_union = sum(
                    float(record.get("soft_silhouette_union", 0.0))
                    for record in foreground_records
                )
                soft_mass = sum(
                    float(record.get("soft_silhouette_mass", 0.0))
                    for record in foreground_records
                )
                metrics["soft_silhouette_iou"] = (
                    float(soft_intersection/soft_union) if soft_union > 0.0 else None
                )
                metrics["soft_silhouette_dice"] = (
                    float(2.0*soft_intersection/soft_mass)
                    if soft_mass > 0.0 else None
                )
                metrics["soft_silhouette_intersection"] = soft_intersection
                metrics["soft_silhouette_union"] = soft_union
                # ROI SSIM and boundary F-score have no additive
                # sufficient statistic; their explicitly named frame mean is the
                # aggregate contract.
                for key in (
                    "foreground_ssim", "foreground_roi_ssim", "boundary_fscore",
                ):
                    metrics[key] = metrics[f"{key}_frame_mean"]
                fixed_reports: list[dict[str, object]] = []
                for color_index, color in enumerate(self.config.eval.background_colors):
                    rows = [record["fixed_backgrounds"][color_index] for record in foreground_records]
                    valid_ssim = [
                        float(row["ssim"]) for row in rows
                        if row.get("ssim") is not None
                    ]
                    squared_error_sum = sum(
                        float(row.get("squared_error_sum", 0.0)) for row in rows
                    )
                    samples = sum(int(row.get("samples", 0)) for row in rows)
                    mse = squared_error_sum / samples if samples else None
                    fixed_reports.append({
                        "background": [float(value) for value in color],
                        "psnr": (
                            None if mse is None else (
                                float("inf") if mse == 0.0
                                else float(-10.0*np.log10(mse))
                            )
                        ),
                        "psnr_frame_mean": float(np.mean([
                            float(row["psnr"]) for row in rows
                        ])),
                        "ssim": float(np.mean(valid_ssim)) if valid_ssim else None,
                        "views": len(rows),
                        "ssim_views": len(valid_ssim),
                        "samples": samples,
                        "squared_error_sum": squared_error_sum,
                    })
                metrics["fixed_backgrounds"] = fixed_reports
                if fixed_reports:
                    metrics["psnr_frame_mean"] = psnr_sum / count
                    fixed_reports[0]["psnr"] = metrics["psnr"]
                    fixed_reports[0]["ssim"] = metrics.get(
                        "full_ssim", metrics["ssim"],
                    )
        if clocks is not None:
            metrics.update(clocks)
        if wall_equalized:
            metrics["wall_equalized"] = True
            if wall_seconds is not None:
                metrics["wall_seconds"] = float(wall_seconds)
        camera_totals: dict[str, dict[str, float | int]] = {}
        for values in partials:
            cameras = values["cameras"]
            if not isinstance(cameras, dict):
                raise RuntimeError("distributed evaluation camera payload is invalid")
            for camera_key, raw in cameras.items():
                if not isinstance(raw, dict):
                    raise RuntimeError("distributed evaluation camera record is invalid")
                target = camera_totals.setdefault(str(camera_key), {
                    "psnr_sum": 0.0, "ssim_sum": 0.0,
                    "count": 0, "ssim_count": 0,
                })
                for key in ("psnr_sum", "ssim_sum"):
                    target[key] = float(target[key]) + float(raw[key])
                target["count"] = int(target["count"]) + int(raw["count"])
                target["ssim_count"] = (
                    int(target["ssim_count"]) + int(raw["ssim_count"])
                )
        camera_metrics: dict[str, dict[str, float | int]] = {}
        for camera_key in sorted(camera_totals, key=int):
            raw = camera_totals[camera_key]
            camera_count = int(raw["count"])
            camera_ssim_count = int(raw["ssim_count"])
            if camera_count <= 0:
                raise RuntimeError("evaluation camera has no reduced views")
            record: dict[str, float | int] = {
                "psnr": float(raw["psnr_sum"]) / camera_count,
                "ssim": (
                    float(raw["ssim_sum"]) / camera_ssim_count
                    if camera_ssim_count else None
                ),
                "views": camera_count,
                "ssim_views": camera_ssim_count,
            }
            camera_metrics[camera_key] = record
        if sum(int(value["views"]) for value in camera_metrics.values()) != count:
            raise RuntimeError("per-camera evaluation counts do not cover all views")
        metrics["camera_metrics"] = camera_metrics
        metrics["camera_count"] = len(camera_metrics)
        if camera_metrics:
            worst_psnr = min(
                camera_metrics, key=lambda key: float(camera_metrics[key]["psnr"])
            )
            ssim_cameras = [
                key for key in camera_metrics
                if camera_metrics[key]["ssim"] is not None
            ]
            metrics["worst_psnr_camera"] = int(worst_psnr)
            metrics["worst_camera_psnr"] = float(camera_metrics[worst_psnr]["psnr"])
            if ssim_cameras:
                worst_ssim = min(
                    ssim_cameras,
                    key=lambda key: float(camera_metrics[key]["ssim"]),
                )
                metrics["worst_ssim_camera"] = int(worst_ssim)
                metrics["worst_camera_ssim"] = float(
                    camera_metrics[worst_ssim]["ssim"]
                )
        if metric_extension is not None:
            extension_metrics, extension_timings = metric_extension.finalize(
                vulkan_metadata=backend.device_metadata(),
            )
            collisions = set(metrics).intersection(extension_metrics)
            if collisions:
                raise RuntimeError(
                    "offline metric extension overwrote metrics: "
                    + ", ".join(sorted(collisions))
                )
            metrics.update(extension_metrics)
            metrics["eval_stage_work_seconds"].update(extension_timings)
        metrics["eval_seconds"] = float(time.perf_counter() - eval_begin)
        metrics_name = result_filename or (
            "final_test_metrics.jsonl"
            if scope == "final_test" else (
                "full_validation_metrics.jsonl" if scope == "validation" else "metrics.jsonl"
            )
        )
        with (self.output_path / metrics_name).open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n"
            )
        primary_metric = primary_metric_name(getattr(
            self.config.eval,
            "selector",
            (
                "foreground_psnr"
                if self.config.pipeline.mode == "foreground" else "psnr"
            ),
        ))
        primary_score = metrics.get(primary_metric)
        if primary_metric == "psnr":
            logger.info(
                "iteration %d: eval psnr %.3f SSIM_2 %s over %d views "
                "in %.2fs",
                step, metrics["psnr"], (
                    f"{float(metrics['ssim']):.5f}"
                    if metrics["ssim"] is not None else "null"
                ), len(selected), metrics["eval_seconds"],
            )
        else:
            logger.info(
                "iteration %d: eval %s %s (composite PSNR %.3f, "
                "foreground SSIM_2 %s) over %d views in %.2fs",
                step, primary_metric, (
                    f"{float(primary_score):.3f}"
                    if primary_score is not None else "null"
                ), metrics["psnr"], (
                    f"{float(metrics['foreground_ssim']):.5f}"
                    if metrics.get("foreground_ssim") is not None else "null"
                ), len(selected), metrics["eval_seconds"],
            )
        return metrics

