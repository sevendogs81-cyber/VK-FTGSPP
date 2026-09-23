"""Community FTGSPP pipeline with one native Vulkan training owner."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ftgspp.config import save_resolved_config
from ftgspp.config.schema import AppConfig
from ftgspp.config.validate import validate_resolved_config
from ftgspp.training.backend import (
    NativeVulkanBackend, SharedRgbViewCache, camera_extent,
)
from ftgspp.io.checkpoint import (
    CheckpointSnapshot,
    HARD_FINAL_CHECKPOINT,
    HARD_FINAL_ROLE,
    MAIN_SELECTED_CHECKPOINT,
    MAIN_SELECTED_ROLE,
    NATIVE_ABI,
    SELECTED_MODEL_ROLE,
    build_checkpoint_metadata,
    capture_checkpoint_snapshot,
    collect_runtime_provenance,
    latest_checkpoint,
    load_checkpoint_full,
    load_model,
    publish_checkpoint,
    resolve_checkpoint_path,
    save_checkpoint_snapshot,
    save_model,
    validate_training_clocks,
    validate_resume_compatibility,
)
from ftgspp.io.hashing import file_sha256
from ftgspp.data.scene import Scene
from ftgspp.training.sampler import (
    sample_rank_views,
    sampling_phase_seed,
)
from ftgspp.training.distributed import (
    DistributedPlan,
    SynchronousBackendGroup,
    _log_geometry_grads_enabled,
)
from ftgspp.training.distributed_contract import (
    shard_resumed_relocation_statistics,
    validate_checkpoint_optimizer_state,
)
from ftgspp.training.background import sample_rank_backgrounds
from ftgspp.training.foreground_metrics import foreground_metrics
from ftgspp.training.objective import composite_v1
from ftgspp.training.initialize import initialize_explicit
from ftgspp.io.model import NativeGaussians
from ftgspp.io.ply import export_4dgs

logger = logging.getLogger("ftgspp.native")


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


def _human_bytes(size: int) -> str:
    value = float(max(int(size), 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable byte unit")


@contextmanager
def _run_file_logging(path: Path, *, resume: bool):
    """Attach one run-scoped file handler without polluting global handlers."""

    package_logger = logging.getLogger("ftgspp")
    previous_level = package_logger.level
    if previous_level == logging.NOTSET or previous_level > logging.INFO:
        package_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        path, mode="a" if resume else "w", encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    ))
    package_logger.addHandler(handler)
    try:
        logger.info(
            "run session started: mode=%s output=%s",
            "resume" if resume else "train", path.parent,
        )
        yield
    except BaseException:
        logger.exception("FTGSPP native Vulkan pipeline failed: %s", path.parent)
        raise
    finally:
        package_logger.removeHandler(handler)
        handler.flush()
        handler.close()
        package_logger.setLevel(previous_level)


class _TrainingReporter:
    """TTY-only phase progress while preserving stable ordinary logs."""

    def __init__(self) -> None:
        isatty = getattr(sys.stderr, "isatty", None)
        self.enabled = bool(callable(isatty) and isatty())
        self._bar = None
        self._postfix: dict[str, str] = {}
        self._redirect = None

    def open(self) -> None:
        if self.enabled and self._redirect is None:
            self._redirect = logging_redirect_tqdm()
            self._redirect.__enter__()

    def close_phase(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def close(self) -> None:
        self.close_phase()
        if self._redirect is not None:
            self._redirect.__exit__(None, None, None)
            self._redirect = None

    def start_phase(self, name: str, *, total: int, initial: int) -> None:
        self.close_phase()
        self._postfix = {}
        if not self.enabled or initial >= total:
            return
        self._bar = tqdm(
            total=int(total), initial=max(0, int(initial)), desc=name,
            unit="step", dynamic_ncols=True, mininterval=1.0,
            smoothing=0.1, file=sys.stderr,
        )

    def advance(
        self, local_step: int, scalars: dict[str, float] | None = None,
    ) -> None:
        if self._bar is None:
            return
        if scalars is not None and "total_loss" in scalars:
            self._postfix["loss"] = f"{float(scalars['total_loss']):.5g}"
        self._bar.set_postfix(self._postfix, refresh=False)
        target = int(local_step) + 1
        if target > self._bar.n:
            self._bar.update(target - self._bar.n)

    def observe_eval(self, metrics: dict[str, object]) -> None:
        if self._bar is None or not metrics:
            return
        self._postfix["PSNR"] = f"{float(metrics['psnr']):.2f}"
        self._postfix["SSIM"] = (
            f"{float(metrics['ssim']):.4f}"
            if metrics.get("ssim") is not None else "null"
        )
        self._bar.set_postfix(self._postfix, refresh=False)

    def checkpoint_started(self, filename: str) -> None:
        if self._bar is None:
            return
        self._postfix["state"] = f"saving {filename}"
        self._bar.set_postfix(self._postfix, refresh=True)

    def checkpoint_finished(self) -> None:
        if self._bar is None:
            return
        self._postfix.pop("state", None)
        self._bar.set_postfix(self._postfix, refresh=True)


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


def _finite_rgb(array: np.ndarray) -> bool:
    return bool(np.isfinite(array.min()) and np.isfinite(array.max()))


def _rank_view_audit(
    dataset: Scene, view_indices: tuple[int, ...],
) -> list[dict[str, int | float]]:
    """Describe the one view assigned to each synchronous rank."""

    result: list[dict[str, int | float]] = []
    for slot, view_index in enumerate(view_indices):
        view = dataset.train_views[int(view_index)]
        result.append({
            "rank": int(slot),
            "view": int(view_index),
            "frame": int(view.frame),
            "camera": int(view.camera_id),
            "timestamp": float(view.timestamp),
        })
    return result


def _background_context_audit(contexts) -> dict[str, object]:
    """Persist the exact immutable background consumed by a logical step."""

    rows = [{
        "rank": int(rank),
        "color": [float(value) for value in context.color],
        "sampler_version": str(context.version),
        "phase": str(context.phase),
        "logical_step": int(context.logical_step),
        "rank_policy": str(context.rank_policy),
        "sampled_rank": (
            None if context.rank is None else int(context.rank)
        ),
    } for rank, context in enumerate(contexts)]
    return {
        "background_contexts": rows,
        "background_shared": bool(
            not rows or all(row["color"] == rows[0]["color"] for row in rows)
        ),
    }


_SYNC_TRAIN_TIMING_KEYS = (
    "data_seconds",
    "backward_seconds",
    "backward_seconds_max",
    "backward_seconds_min",
    "backward_seconds_imbalance",
    "gradient_download_seconds",
    "reduction_seconds",
    "gradient_upload_seconds",
    "gradient_host_memcpy_seconds",
    "gradient_exchange_wall_seconds",
    "gradient_peer_wait_seconds",
    "gradient_peer_wait_seconds_mean",
    "gradient_sparse_pack_seconds",
    "gradient_transfer_wall_seconds",
    "gradient_transfer_wall_seconds_mean",
    "gradient_transfer_overlap",
    "gradient_exchange_overlap",
    "gradient_async_pipeline",
    "gradient_persistent_host_mapping",
    "optimizer_seconds",
    "wave_backward_download_seconds",
)


def _sync_train_timing(sync_result: dict[str, object]) -> dict[str, float]:
    """Flatten synchronous step timing for train_metrics.jsonl.

    Keeps legacy fields and adds peer-wait / transfer / per-rank backward splits
    so full-resolution imbalance is not mistaken for PCIe overlap failure.
    """

    timing = {
        key: float(sync_result.get(key, 0.0))
        for key in _SYNC_TRAIN_TIMING_KEYS
    }
    for key, value in sync_result.items():
        if key.startswith("backward_seconds_rank") and isinstance(
            value, (int, float, np.floating)
        ):
            timing[key] = float(value)
    return timing


def _pair_exchange_log_fields(
    sync_result: dict[str, object],
) -> tuple[dict[str, float], dict[str, object]]:
    """Split pair-exchange numeric samples from per-step codec metadata."""

    exchange = sync_result.get("pair_exchange")
    if not isinstance(exchange, dict):
        return {}, {}
    timing = {
        "pair_payload_bytes": float(exchange.get("payload_bytes", 0)),
        "pair_peer_payload_bytes": float(
            exchange.get("peer_payload_bytes", 0)
        ),
        "pair_bucket_count": float(exchange.get("bucket_count", 0)),
        "pair_local_visible": float(exchange.get("local_visible_count", 0)),
        "pair_peer_visible": float(exchange.get("peer_visible_count", 0)),
    }
    metadata: dict[str, object] = {
        "pair_codec_policy": str(exchange.get("codec_policy", "unknown")),
        "pair_gaussian_codec": str(
            exchange.get("gaussian_codec", "unknown")
        ),
        "pair_sparse": bool(exchange.get("sparse", False)),
        "pair_active_sh_degree": int(exchange.get("active_sh_degree", 0)),
        "pair_active_sh_floats": int(exchange.get("active_sh_floats", 0)),
    }
    return timing, metadata


class _TrainingPerformanceProfile:
    """Low-overhead wall/stage samples used by equal-work comparisons."""

    def __init__(self, *, views_per_step: int) -> None:
        self.views_per_step = int(views_per_step)
        self.samples: dict[str, list[float]] = {"step_seconds": []}

    def record(
        self,
        *,
        step_seconds: float,
        timing: dict[str, float],
    ) -> dict[str, float]:
        self.samples["step_seconds"].append(float(step_seconds))
        for key, value in timing.items():
            if np.isfinite(value):
                self.samples.setdefault(key, []).append(float(value))
        return self.window_summary()

    def window_summary(self) -> dict[str, float]:
        steps = np.asarray(self.samples["step_seconds"], dtype=np.float64)
        if steps.size == 0:
            return {}
        total = float(np.sum(steps))
        step_rate = float(steps.size / max(total, 1e-12))
        result = {
            "step_seconds_p50": float(np.percentile(steps, 50)),
            "step_seconds_p95": float(np.percentile(steps, 95)),
            "steps_per_second": step_rate,
            "views_per_second": step_rate * self.views_per_step,
        }
        for key, values in self.samples.items():
            if key == "step_seconds" or not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            result[f"{key}_p50"] = float(np.percentile(array, 50))
            result[f"{key}_p95"] = float(np.percentile(array, 95))
        return result

    def report(
        self,
        *,
        clocks: dict[str, int | float],
        native: dict[str, object],
        quality: dict[str, object] | None,
    ) -> dict[str, object]:
        return {
            "schema": "ftgspp-explicit-performance-v1",
            "comparison_axes": {
                "optimizer_steps": int(clocks.get("optimizer_steps", 0)),
                "views_seen": int(clocks.get("views_seen", 0)),
                "wall_seconds": float(clocks.get("wall_seconds", 0.0)),
            },
            "views_per_step": self.views_per_step,
            "sample_count": len(self.samples["step_seconds"]),
            "timing": self.window_summary(),
            "native": native,
            "latest_quality_metrics": quality,
        }


def _last_jsonl_record(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    last: dict[str, object] | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    return last


def snapshot_cadence(step: int, interval: int) -> bool:
    """True on the shared eval / optional-checkpoint cadence.

    ``train.checkpoint_interval`` is the only step switch. Zero disables
    in-loop snapshots; the last trained step is still evaluated separately
    when hold-out views exist.
    """

    return interval > 0 and step > 0 and step % interval == 0


JOINT_SELECTION_RULE = "joint_psnr_ssim_db"


def primary_metric_name(selector: str) -> str:
    """Return the task PSNR that enters the joint hold-out score."""

    return "foreground_psnr" if str(selector) == "foreground_psnr" else "psnr"


def selection_ssim_name(selector: str) -> str:
    """Return the SSIM variant paired with the task PSNR selector."""

    return "foreground_ssim" if str(selector) == "foreground_psnr" else "ssim"


def ssim_distortion_db(ssim: float) -> float:
    """Map SSIM residual ``1 - ssim`` into the same ``-10 log10`` units as PSNR."""

    residual = 1.0 - min(max(float(ssim), 0.0), 1.0 - 1e-12)
    return float(-10.0 * math.log10(max(residual, 1e-12)))


def joint_quality_score(psnr: float, ssim: float) -> float:
    """Combine task PSNR and SSIM so both metrics always move the ranking.

    SSIM is converted to distortion-dB before addition. The comparison is a
    total order, which keeps online best-tracking independent of eval order.
    Perfect PSNR (``+inf``) still participates through the SSIM term.
    """

    ssim_db = ssim_distortion_db(ssim)
    psnr = float(psnr)
    if math.isinf(psnr):
        return math.copysign(1e6, psnr) + ssim_db
    if not math.isfinite(psnr):
        return -math.inf
    return psnr + ssim_db


def holdout_is_better(
    candidate: dict[str, object] | None,
    current_best: dict[str, object] | None,
    *,
    selector: str = "psnr",
    min_silhouette_iou: float | None = None,
    max_background_opacity_leakage: float | None = None,
    max_foreground_holes: float | None = None,
    min_boundary_fscore: float | None = None,
) -> bool:
    """Prefer the higher joint PSNR+SSIM score after optional quality gates."""

    score_key = primary_metric_name(selector)
    ssim_key = selection_ssim_name(selector)
    if (
        candidate is None or candidate.get(score_key) is None
        or candidate.get(ssim_key) is None
    ):
        return False
    if min_silhouette_iou is not None and "silhouette_iou" in candidate and (
        candidate.get("silhouette_iou") is None
        or float(candidate["silhouette_iou"]) < float(min_silhouette_iou)
    ):
        return False
    if max_background_opacity_leakage is not None and (
        "background_opacity_leakage" in candidate
    ) and (
        candidate.get("background_opacity_leakage") is None
        or float(candidate["background_opacity_leakage"])
        > float(max_background_opacity_leakage)
    ):
        return False
    if max_foreground_holes is not None and "foreground_holes" in candidate and (
        candidate.get("foreground_holes") is None
        or float(candidate["foreground_holes"]) > float(max_foreground_holes)
    ):
        return False
    if min_boundary_fscore is not None and "boundary_fscore" in candidate and (
        candidate.get("boundary_fscore") is None
        or float(candidate["boundary_fscore"]) < float(min_boundary_fscore)
    ):
        return False
    if (
        current_best is None
        or current_best.get(score_key) is None
        or current_best.get(ssim_key) is None
    ):
        return True
    return joint_quality_score(
        float(candidate[score_key]), float(candidate[ssim_key]),
    ) > joint_quality_score(
        float(current_best[score_key]), float(current_best[ssim_key]),
    )


def joint_selection_fields(
    psnr: float, ssim: float, *, ssim_metric: str = "ssim",
) -> dict[str, object]:
    return {
        "selection_rule": JOINT_SELECTION_RULE,
        "selection_ssim_metric": str(ssim_metric),
        "joint_score": joint_quality_score(float(psnr), float(ssim)),
    }


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


def _load_json_object(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _quality_for_performance_summary(
    output_path: Path,
) -> dict[str, object] | None:
    """Prefer the exported snapshot's metrics over the last jsonl eval line."""

    selection = _load_json_object(output_path / "export_selection.json")
    latest = _last_jsonl_record(output_path / "metrics.jsonl")
    if selection is None or "psnr" not in selection:
        return latest
    quality: dict[str, object] = dict(latest) if latest is not None else {}
    quality.update({
        "export_source": selection.get("source", "best_eval"),
        "export_step": int(selection["step"]),
        "step": int(selection["step"]),
        "psnr": float(selection["psnr"]),
        "ssim_variant": selection.get("ssim_variant", "SSIM_2"),
        "selection_ssim_metric": selection.get(
            "selection_ssim_metric", "ssim",
        ),
        "ssim_data_range": float(selection.get("ssim_data_range", 2.0)),
    })
    selected_ssim = _optional_metric(selection, "ssim")
    if selected_ssim is not None:
        quality["ssim"] = selected_ssim
    for key in ("foreground_psnr", "foreground_ssim", "full_ssim"):
        if key in selection and selection[key] is not None:
            quality[key] = float(selection[key])
    if "views" in selection:
        quality["views"] = int(selection["views"])
    return quality


def _validate_resume_phase_cursor(
    metadata: dict[str, object],
    *,
    completed: int,
    main_iterations: int,
    hard_separation: bool,
    hard_steps: int = 3000,
) -> str:
    """Validate the global iteration/phase-local cursor before native upload.

    ``runtime__iteration`` is the authoritative completed optimizer step.  V4
    also records a phase-local cursor so a forged/stale phase label cannot make
    the runner skip the main phase or destructively split hard separation a
    second time.  Converted checkpoints may deliberately use ``phase_step=-1``;
    in that case the cursor is reconstructed from the global iteration after
    the declared phase has passed the boundary checks below.
    """

    phase = str(metadata.get("phase", "unknown"))
    allowed = {
        "unknown", "main", "main_complete", "main_all_dead",
        "hard_separation", "hard_separation_complete",
    }
    if phase not in allowed:
        raise ValueError(f"unsupported checkpoint phase {phase!r}")
    completed_step = int(completed)
    main_steps = int(main_iterations)
    if phase == "unknown":
        if completed_step >= main_steps:
            raise ValueError(
                "checkpoint phase is unknown at/after the hard-separation "
                "boundary; reconvert it with --phase hard_separation (or "
                "hard_separation_complete) before resume"
            )
        return phase

    hard_phase = phase.startswith("hard_separation")
    if hard_phase and not hard_separation:
        raise ValueError(
            "hard-separation checkpoint cannot resume with "
            "train.hard_separation=false"
        )
    if hard_phase:
        if completed_step < main_steps:
            raise ValueError(
                "hard-separation checkpoint iteration precedes the main "
                "training boundary"
            )
        expected_phase_step = completed_step - main_steps
        if expected_phase_step >= int(hard_steps):
            raise ValueError(
                "hard-separation checkpoint iteration exceeds the "
                f"{int(hard_steps)}-step phase namespace"
            )
    else:
        if completed_step >= main_steps:
            raise ValueError(
                "main-phase checkpoint iteration crosses into hard separation"
            )
        expected_phase_step = completed_step
        if phase == "main_complete" and completed_step != main_steps - 1:
            raise ValueError(
                "main_complete checkpoint does not end at the configured main "
                "training boundary"
            )

    raw_phase_step = metadata.get("phase_step", -1)
    if isinstance(raw_phase_step, bool) or not isinstance(raw_phase_step, int):
        raise ValueError("checkpoint phase_step is not an integer")
    recorded_phase_step = int(raw_phase_step)
    if recorded_phase_step not in {-1, expected_phase_step}:
        raise ValueError(
            "checkpoint phase_step does not match runtime__iteration: "
            f"recorded={recorded_phase_step}, expected={expected_phase_step}"
        )
    return phase


@dataclass
class NativeRunner:
    config: AppConfig
    @property
    def output_path(self) -> Path:
        return self.config.run.output_path

    def _preflight(self) -> Scene:
        if self.config.train.color_correction:
            logger.info("native Vulkan color correction enabled (per-camera affine)")
        dataset = Scene.from_config(self.config)
        resume_requested = bool(
            self.config.train.resume
            or self.config.train.resume_checkpoint is not None
        )
        if resume_requested and not dataset.calibration_available:
            raise RuntimeError(
                "cannot resume an uncalibrated dataset before DA3 has produced "
                "a camera_manifest.npz; run one fresh init.backend=da3/auto run "
                "first so the estimated camera rig is cached"
            )
        self.config.data.train_cameras = dataset.train_cameras
        self.config.data.eval_cameras = dataset.eval_cameras
        self.config.data.camera_split_source = dataset.camera_split_source
        validate_resolved_config(self.config, dataset.cameras)
        first = dataset.train_views[0]
        image = first.load_rgb()
        logger.info(
            "native dataset: %d frames, %d cameras, %d train views, "
            "%d eval views, first=%s shape=%s",
            len(dataset.frames), len(dataset.cameras), len(dataset.train_views), len(dataset.eval_views), first.path, image.shape,
        )
        return dataset

    def _evaluate(
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
        full_validation: bool = False,
        metric_extension: EvaluationMetricExtension | None = None,
    ) -> dict[str, object]:
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
        full_validation = bool(
            full_validation
            or getattr(dataset, "evaluation_identity", None) is not None
        )
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
            if evaluation_identity is not None else "metrics.jsonl"
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

    def _optimizer_update(self, backend, group, views, step, should_log,
                          next_views, *, phase="main"):
        """One view per rank, one exchange (pair only), one Adam update."""
        world_size = len(views)
        background_contexts = sample_rank_backgrounds(
            self.config.train.background,
            phase=phase,
            logical_step=int(step),
            world_size=world_size,
        )
        backgrounds = tuple(context.color for context in background_contexts)
        background_audit = _background_context_audit(background_contexts)
        if group is not None:
            result = group.train_step(
                views, step, collect_loss=should_log,
                next_view_indices=next_views, phase=phase,
                backgrounds=backgrounds,
            )
            timing = _sync_train_timing(result)
            pair_timing, fields = _pair_exchange_log_fields(result)
            timing.update(pair_timing)
            fields.update(background_audit)
            scalars = None
            if should_log:
                if not isinstance(result["loss"], dict):
                    raise RuntimeError("synchronous log step did not return loss scalars")
                scalars = dict(result["loss"])
            return scalars, timing, fields
        collect_grads = should_log and _log_geometry_grads_enabled()
        if collect_grads:
            backend.set_collect_geometry_grad_stats(True)
        try:
            backend.train_view(
                views[0], step, collect_loss=should_log,
                background=backgrounds[0],
            )
            scalars = backend.last_loss() if should_log else None
            if collect_grads:
                scalars.update(backend.geometry_gradient_stats())
            return scalars, {}, background_audit
        finally:
            if collect_grads:
                backend.set_collect_geometry_grad_stats(False)

    def _train(self, dataset: Scene, model: NativeGaussians | None = None) -> NativeGaussians | None:
        checkpoint_dir = self.output_path / "ckpt"
        distributed = DistributedPlan.from_config(self.config)
        provenance = collect_runtime_provenance(
            device_ids=list(distributed.device_ids)
        )
        loaded_metadata: dict[str, object] | None = None
        loaded_clocks: dict[str, int | float] | None = None
        clock_resume_warning: str | None = None
        selected_main_handoff = False
        resume_path = self.config.train.resume_checkpoint
        if self.config.train.resume and resume_path is None:
            resume_path = latest_checkpoint(checkpoint_dir)
            if resume_path is None:
                raise FileNotFoundError("train.resume=true but no native checkpoint exists")
        if resume_path is not None:
            resume_path = resolve_checkpoint_path(Path(resume_path))
            loaded = load_checkpoint_full(resume_path)
            model = loaded.model
            optimizer = dict(loaded.optimizer)
            completed = loaded.iteration
            loaded_metadata = dict(loaded.metadata)
            selected_main_handoff = (
                loaded_metadata.get("checkpoint_role") == MAIN_SELECTED_ROLE
            )
            if (
                resume_path.name == MAIN_SELECTED_CHECKPOINT
                and self.config.train.hard_separation
                and not selected_main_handoff
            ):
                raise ValueError(
                    "retained main selection is not eligible for hard "
                    "separation handoff"
                )
            if selected_main_handoff and not self.config.train.hard_separation:
                raise ValueError(
                    "main-selected checkpoint is only valid for starting "
                    "train.hard_separation=true"
                )
            loaded_clocks, clock_resume_warning = (
                validate_training_clocks(
                    loaded_metadata, iteration=completed,
                    fallback_world_size=distributed.world_size,
                )
            )
            # The model/optimizer variables below retain the owning arrays;
            # discard the aggregate checkpoint object so it cannot keep a
            # second long-lived mapping reference through the training loop.
            del loaded
            resume_phase = _validate_resume_phase_cursor(
                loaded_metadata,
                completed=completed,
                main_iterations=self.config.train.iterations,
                hard_separation=self.config.train.hard_separation,
                hard_steps=self.config.train.hard_separation_steps,
            )
            start = completed+1
        else:
            if model is None:
                model = initialize_explicit(self.config, dataset)
            optimizer, completed, start = {}, -1, 0
            resume_phase = "none"
        if resume_path is not None:
            color_keys = {"color_params", "color_m", "color_v"}
            color_state_present = bool(color_keys & set(optimizer))
            # The training pass activates correction only after the configured
            # start step.  Evaluation may initialize it earlier, while every
            # hard-separation step uses the frozen affine transform.  Accept
            # either legitimate path, but require the whole atomic state group
            # once any member is present or correction must already be active.
            color_state_required = bool(
                self.config.train.color_correction
                and (
                    color_state_present
                    or completed > self.config.train.color_correction_start
                    or completed >= self.config.train.iterations
                )
            )
            if selected_main_handoff:
                unexpected = set(optimizer) - {"color_params"}
                if unexpected:
                    raise ValueError(
                        "main-selected handoff contains unexpected optimizer "
                        f"arrays: {sorted(unexpected)}"
                    )
                if color_state_required:
                    params = np.asarray(
                        optimizer.get("color_params", []), dtype=np.float32,
                    )
                    expected = 12 * len(set(dataset.cameras))
                    if params.size != expected or not np.all(np.isfinite(params)):
                        raise ValueError(
                            "main-selected handoff has invalid color_params: "
                            f"expected {expected} finite floats, got {params.size}"
                        )
            else:
                validate_checkpoint_optimizer_state(
                    optimizer,
                    num_gaussians=len(model),
                    color_num_cameras=(
                        len(set(dataset.cameras)) if color_state_required else 0
                    ),
                )
        if not self.config.model.marginal_gating:
            # Align convert / train-only / resume with initialize_explicit:
            # inactive gates stay at -100 and receive no Adam updates.
            model.gate_raw = np.full_like(model.gate_raw, -100.0, dtype=np.float32)
        backends: list[NativeVulkanBackend] = []
        try:
            for device_id in distributed.device_ids:
                backends.append(NativeVulkanBackend(
                    device_id=device_id,
                    honor_device_env=not distributed.enabled,
                ))
        except BaseException:
            for replica in reversed(backends):
                replica.close()
            raise
        backend = backends[0]
        group: SynchronousBackendGroup | None = None
        training_clocks: dict[str, int | float] = {}
        performance_profile = _TrainingPerformanceProfile(
            views_per_step=distributed.world_size,
        )
        reporter = _TrainingReporter()
        rgb_prefetch_executor: ThreadPoolExecutor | None = None
        rgb_prefetch_lookahead = 0
        if (
            distributed.enabled
            and distributed.collective_bucket_bytes == 0
            and loaded_metadata is not None
        ):
            # An automatic bucket is a performance choice, not optimizer math.
            # Resume reuses the measured checkpoint value so calibration noise
            # cannot manufacture an otherwise meaningless ABI mismatch.
            saved_training = loaded_metadata.get("training", {})
            saved_collective = (
                saved_training.get("collective", {})
                if isinstance(saved_training, dict) else {}
            )
            saved_bucket = (
                int(saved_collective.get("bucket_bytes", 0))
                if isinstance(saved_collective, dict) else 0
            )
            if saved_bucket > 0:
                distributed = replace(
                    distributed, collective_bucket_bytes=saved_bucket,
                )
        if distributed.enabled:
            try:
                group = SynchronousBackendGroup(
                    backends,
                    device_ids=distributed.device_ids,
                    reduction=distributed.reduction,
                    transport=distributed.backend,
                    collective_algorithm=distributed.collective_algorithm,
                    collective_bucket_bytes=distributed.collective_bucket_bytes,
                    collective_timeout_seconds=distributed.collective_timeout_seconds,
                    collective_sparse=distributed.collective_sparse,
                    collective_codec_policy=(
                        distributed.collective_codec_policy),
                )
                # Persist the measured value, never the configuration sentinel
                # (bucket_mb=0), in resume/checkpoint compatibility metadata.
                distributed = replace(
                    distributed,
                    collective_bucket_bytes=group.collective_bucket_bytes,
                )
            except BaseException:
                for replica in reversed(backends):
                    replica.close()
                raise
        try:
            reporter.open()
            device_metadata = [replica.device_metadata() for replica in backends]
            for rank, metadata in enumerate(device_metadata):
                logger.info(
                    "Vulkan device rank=%d metadata=%s",
                    rank, json.dumps(metadata, sort_keys=True, default=str),
                )
            reported_abis = {
                replica.reported_native_abi() for replica in backends
            }
            if len(reported_abis) != 1:
                raise RuntimeError(
                    "Vulkan replicas report different native ABIs: "
                    f"{sorted(reported_abis)}"
                )
            reported_abi = next(iter(reported_abis))
            if reported_abi != NATIVE_ABI:
                raise RuntimeError(
                    "stale/incompatible native extension ABI: Python expects "
                    f"{NATIVE_ABI!r}, but Vulkan reports {reported_abi!r}; "
                    "rebuild the native module and every SPIR-V shader"
                )
            extension_metadata = [
                dict(value.get("native_extension", {}))
                for value in device_metadata
            ]
            if any(
                value != extension_metadata[0]
                for value in extension_metadata[1:]
            ):
                raise RuntimeError(
                    "Vulkan replicas loaded different native extension binaries"
                )
            provenance["devices"] = device_metadata
            provenance["native_reported_abi"] = reported_abi
            provenance["native_extension"] = extension_metadata[0]
            if loaded_metadata is not None:
                current_resume_metadata = build_checkpoint_metadata(
                    self.config, dataset, distributed,
                    phase="resume", phase_step=-1, provenance=provenance,
                    transport=(
                        group.transport if group is not None
                        else distributed.backend
                    ),
                    effective_sparse=(
                        group.collective_sparse
                        if group is not None else distributed.collective_sparse
                    ),
                )
                for warning in validate_resume_compatibility(
                    loaded_metadata,
                    current_resume_metadata,
                    allow_world_size_change=bool(
                        self.config.distributed.allow_resume_world_size_change
                    ),
                    allow_hard_separation_enable=selected_main_handoff,
                ):
                    logger.warning(
                        "checkpoint resume compatibility: %s", warning
                    )
                if clock_resume_warning is not None:
                    logger.warning(
                        "checkpoint resume compatibility: %s",
                        clock_resume_warning,
                    )
            extent = camera_extent(dataset)
            for replica in backends:
                replica.configure(self.config, extent)
                replica.upload_model(model)
            from ftgspp.training.backend import resolve_view_load_options
            view_opts = resolve_view_load_options(self.config)
            shared_rgb_cache = (
                SharedRgbViewCache(max(
                    int(view_opts["rgb_cache"]), distributed.world_size,
                ))
                if distributed.enabled else None
            )
            logger.info(
                "registering views on %d native replica(s) devices=%s "
                "(lazy, rgb_cache=%s, workers=%s)",
                distributed.world_size,
                list(distributed.device_ids),
                view_opts["rgb_cache"],
                view_opts["workers"],
            )
            for replica in backends:
                replica.upload_dataset(
                    dataset,
                    rgb_cache=int(view_opts["rgb_cache"]),
                    workers=int(view_opts["workers"]),
                    shared_rgb_cache=shared_rgb_cache,
                )
            rgb_workers = int(view_opts["workers"])
            if rgb_workers > 0:
                default_lookahead = max(1, min(rgb_workers, 8))
                lookahead_raw = os.environ.get("FTGSPP_VIEW_PREFETCH_LOOKAHEAD")
                rgb_prefetch_lookahead = (
                    int(lookahead_raw)
                    if lookahead_raw is not None else default_lookahead
                )
                if group is not None:
                    group.configure_view_lookahead(workers=rgb_workers)
                    logger.info(
                        "sync lazy RGB lookahead enabled: "
                        "workers=%d lookahead=%d",
                        rgb_workers, max(rgb_prefetch_lookahead, 0),
                    )
                elif rgb_prefetch_lookahead > 0:
                    rgb_prefetch_executor = ThreadPoolExecutor(
                        max_workers=rgb_workers,
                        thread_name_prefix="ftgspp-rgb-lookahead",
                    )
                    logger.info(
                        "single-device lazy RGB lookahead enabled: "
                        "workers=%d lookahead=%d",
                        rgb_workers, rgb_prefetch_lookahead,
                    )
            if optimizer:
                if selected_main_handoff:
                    color_params = optimizer.get("color_params")
                    if color_params is not None:
                        for replica in backends:
                            replica.set_color_params(color_params)
                elif group is None:
                    backend.restore_optimizer(optimizer)
                else:
                    global_stats = optimizer.get("relocation_stats")
                    if global_stats is None:
                        raise ValueError(
                            "synchronous resume checkpoint lacks relocation_stats"
                        )
                    global_stats = np.asarray(global_stats, dtype=np.float32)
                    if global_stats.ndim == 1:
                        if global_stats.size != 2 * len(model):
                            raise ValueError(
                                "flat checkpoint relocation_stats has "
                                f"{global_stats.size} values, expected {2 * len(model)}"
                            )
                        global_stats = global_stats.reshape(len(model), 2)
                    if global_stats.shape != (len(model), 2):
                        raise ValueError(
                            "checkpoint relocation_stats must have shape "
                            f"({len(model)}, 2), got {global_stats.shape}"
                        )
                    relocation_shards = shard_resumed_relocation_statistics(
                        global_stats, distributed.world_size, owner_rank=0
                    )
                    for rank, replica in enumerate(backends):
                        # restore_optimizer only reads arrays; a shallow mapping
                        # avoids another multi-gigabyte host copy per replica.
                        rank_optimizer = dict(optimizer)
                        rank_optimizer["relocation_stats"] = relocation_shards[rank]
                        replica.restore_optimizer(rank_optimizer)

                    # Loop locals persist in Python. Drop the last shallow rank
                    # mapping and statistics views before clearing the source.
                    rank_optimizer.clear()
                    del rank_optimizer, relocation_shards, global_stats

            # Upload/restore calls synchronously consume these owning arrays.
            # Keeping them in local variables would pin a complete host model
            # and (on resume) multi-GB Adam state throughout native training.
            if loaded_metadata is not None and not selected_main_handoff:
                rasterizers = loaded_metadata.get("rasterizers", [])
                replicas = group.backends if group is not None else [backend]
                if len(rasterizers) != len(replicas):
                    raise ValueError("checkpoint is missing per-rank rasterizer state")
                for replica, state in zip(replicas, rasterizers):
                    replica.restore_rasterizer_state(state)
            optimizer.clear()
            del optimizer, model

            last_published_iteration: int | None = None
            final_model_snapshot: NativeGaussians | None = None
            training_clocks: dict[str, int | float] = (
                dict(loaded_clocks) if loaded_clocks is not None else {
                    "optimizer_steps": 0,
                    "views_seen": 0,
                    "wall_seconds": 0.0,
                }
            )
            wall_clock_base = float(training_clocks["wall_seconds"])

            def publish_runtime_checkpoint(
                iteration: int,
                *,
                phase: str,
                phase_step: int,
                filename: str | None = None,
            ) -> CheckpointSnapshot:
                """Capture model/optimizer once, write once, advance pointer."""

                nonlocal last_published_iteration
                checkpoint_name = filename or f"{int(iteration):06d}.npz"
                relative_path = Path("ckpt") / checkpoint_name
                checkpoint_begin = time.perf_counter()
                logger.info(
                    "checkpoint saving: %s (phase=%s, step=%d)",
                    relative_path, phase, iteration,
                )
                reporter.checkpoint_started(checkpoint_name)
                try:
                    checkpoint_optimizer = (
                        group.checkpoint_optimizer_state()
                        if group is not None else None
                    )
                    metadata = build_checkpoint_metadata(
                        self.config, dataset, distributed,
                        phase=phase, phase_step=phase_step,
                        provenance=provenance,
                        transport=(
                            group.transport if group is not None
                            else distributed.backend
                        ),
                        effective_sparse=(
                            group.collective_sparse
                            if group is not None else distributed.collective_sparse
                        ),
                        clocks={
                            **training_clocks,
                            "wall_seconds": (
                                wall_clock_base
                                + max(0.0, time.perf_counter() - begin)
                            ),
                        },
                    )
                    metadata["rasterizers"] = [
                        replica.rasterizer_state()
                        for replica in (
                            group.backends if group is not None else [backend]
                        )
                    ]
                    snapshot = capture_checkpoint_snapshot(
                        backend, iteration,
                        optimizer_state=checkpoint_optimizer,
                        metadata=metadata,
                    )
                    checkpoint_path = publish_checkpoint(
                        checkpoint_dir, snapshot,
                        filename=filename, update_latest=True,
                    )
                finally:
                    reporter.checkpoint_finished()
                logger.info(
                    "checkpoint saved: %s (%s) in %.2fs; latest.json updated",
                    relative_path, _human_bytes(checkpoint_path.stat().st_size),
                    time.perf_counter() - checkpoint_begin,
                )
                last_published_iteration = int(iteration)
                return snapshot

            def publish_selected_checkpoint(
                selected_model: NativeGaussians,
                selected_color_params: np.ndarray | None,
                *,
                filename: str,
                role: str,
                boundary_iteration: int,
                phase: str,
                phase_step: int,
                source_step: int,
            ) -> Path:
                """Persist the selected model needed at a phase boundary.

                HS constructs a fresh Adam optimizer, so this handoff stores
                only the selected model and its inference-time color affine.
                Numbered checkpoints remain the exact mid-optimizer resume
                artifacts and continue to own ``latest.json``.
                """

                metadata = build_checkpoint_metadata(
                    self.config, dataset, distributed,
                    phase=phase, phase_step=phase_step,
                    provenance=provenance,
                    transport=(
                        group.transport if group is not None
                        else distributed.backend
                    ),
                    effective_sparse=(
                        group.collective_sparse
                        if group is not None else distributed.collective_sparse
                    ),
                    clocks={
                        **training_clocks,
                        "wall_seconds": (
                            wall_clock_base
                            + max(0.0, time.perf_counter() - begin)
                        ),
                    },
                )
                metadata.update({
                    "checkpoint_role": role,
                    "selected_source_step": int(source_step),
                    "selected_pipeline_mode": str(self.config.pipeline.mode),
                    "selected_selector": str(self.config.eval.selector),
                })
                selected_optimizer = {}
                if selected_color_params is not None:
                    selected_optimizer["color_params"] = np.ascontiguousarray(
                        selected_color_params, dtype=np.float32,
                    )
                snapshot = capture_checkpoint_snapshot(
                    backend, boundary_iteration,
                    model=selected_model,
                    optimizer_state=selected_optimizer,
                    metadata=metadata,
                )
                target = checkpoint_dir / filename
                save_checkpoint_snapshot(target, snapshot, overwrite=True)
                logger.info(
                    "selected checkpoint saved: %s (role=%s source_step=%d)",
                    target.relative_to(self.output_path), role, source_step,
                )
                return target

            snapshot_interval = int(self.config.train.checkpoint_interval)
            save_checkpoints = bool(self.config.train.save_checkpoints)
            export_best = bool(self.config.train.export_best)
            logger.info(
                "training mode=%s world_size=%d views_per_step=%d "
                "reduction=%s transport=%s sampling=%s "
                "snapshot_interval=%d save_checkpoints=%s export_best=%s",
                distributed.mode,
                distributed.world_size,
                distributed.world_size,
                distributed.reduction,
                group.transport if group is not None else distributed.backend,
                distributed.sampling_policy,
                snapshot_interval,
                save_checkpoints,
                export_best,
            )
            main_phase_seed = sampling_phase_seed(
                seed=self.config.train.seed,
                phase="main",
                train_iterations=self.config.train.iterations,
            )
            hard_phase_seed = sampling_phase_seed(
                seed=self.config.train.seed,
                phase="hard_separation",
                train_iterations=self.config.train.iterations,
            )
            log_interval = int(self.config.train.log_interval)
            train_log = self.output_path / "train_metrics.jsonl"
            last = start-1
            main_terminated_all_dead = resume_phase == "main_all_dead"
            main_steps_executed = False
            begin = time.perf_counter()
            wall_eval_interval = float(
                getattr(distributed, "wall_eval_interval_seconds", 0.0) or 0.0
            )
            last_wall_eval = begin
            best_model: NativeGaussians | None = None
            best_color_params: np.ndarray | None = None
            best_metrics: dict[str, object] | None = None
            last_eval_step: int | None = None
            last_eval_metrics: dict[str, object] | None = None

            def consider_export_candidate(
                step: int,
                metrics: dict[str, object],
                *,
                checkpoint_snapshot: CheckpointSnapshot | None = None,
                track_best: bool = True,
            ) -> None:
                nonlocal best_model, best_color_params, best_metrics
                nonlocal last_eval_step, last_eval_metrics
                if metrics:
                    last_eval_step = int(step)
                    reporter.observe_eval(metrics)
                    last_eval_metrics = {
                        key: value for key, value in metrics.items()
                        if key != "foreground_views"
                    }
                    last_eval_metrics.update({
                        "step": int(step),
                        "selector": str(self.config.eval.selector),
                        "selector_constraints": {
                            "min_silhouette_iou": (
                                self.config.eval.selector_min_silhouette_iou
                            ),
                            "max_background_opacity_leakage": (
                                self.config.eval.selector_max_background_opacity_leakage
                            ),
                            "max_foreground_holes": (
                                self.config.eval.selector_max_foreground_holes
                            ),
                            "min_boundary_fscore": (
                                self.config.eval.selector_min_boundary_fscore
                            ),
                        },
                    })
                if not track_best or not export_best or not metrics:
                    return
                if not holdout_is_better(
                    metrics, best_metrics, selector=self.config.eval.selector,
                    min_silhouette_iou=(
                        self.config.eval.selector_min_silhouette_iou
                    ),
                    max_background_opacity_leakage=(
                        self.config.eval.selector_max_background_opacity_leakage
                    ),
                    max_foreground_holes=(
                        self.config.eval.selector_max_foreground_holes
                    ),
                    min_boundary_fscore=(
                        self.config.eval.selector_min_boundary_fscore
                    ),
                ):
                    return
                best_model = (
                    checkpoint_snapshot.model
                    if checkpoint_snapshot is not None else backend.model()
                )
                if self.config.train.color_correction:
                    snapshot_color = (
                        checkpoint_snapshot.optimizer.get("color_params")
                        if checkpoint_snapshot is not None else None
                    )
                    best_color_params = (
                        np.ascontiguousarray(snapshot_color, dtype=np.float32)
                        if snapshot_color is not None
                        else backend.color_params()
                    )
                else:
                    best_color_params = None
                best_metrics = dict(last_eval_metrics or {})
                primary_metric = primary_metric_name(self.config.eval.selector)
                selection_ssim_metric = selection_ssim_name(
                    self.config.eval.selector
                )
                primary_score = float(best_metrics[primary_metric])
                joint_score = joint_quality_score(
                    primary_score, float(best_metrics[selection_ssim_metric]),
                )
                if primary_metric == "psnr":
                    logger.info(
                        "iteration %d: new best hold-out joint %.3f "
                        "(psnr %.3f, SSIM %.5f; export candidate)",
                        step, joint_score, primary_score,
                        best_metrics[selection_ssim_metric],
                    )
                else:
                    logger.info(
                        "iteration %d: new best hold-out joint %.3f "
                        "(%s %.3f, composite PSNR %.3f, foreground SSIM %.5f; "
                        "export candidate)",
                        step, joint_score, primary_metric, primary_score,
                        best_metrics["psnr"], best_metrics[selection_ssim_metric],
                    )

            def publish_if_enabled(
                iteration: int, *, phase: str, phase_step: int,
            ) -> CheckpointSnapshot | None:
                if not save_checkpoints:
                    return None
                return publish_runtime_checkpoint(
                    iteration, phase=phase, phase_step=phase_step,
                )
            train_view_frames = tuple(
                int(view.frame) for view in dataset.train_views
            )
            train_view_cameras = tuple(
                int(view.camera_id) for view in dataset.train_views
            )

            def _sample_batch(local_step: int) -> tuple[int, ...]:
                return sample_rank_views(
                    seed=main_phase_seed,
                    step=local_step,
                    world_size=distributed.world_size,
                    num_views=len(dataset.train_views),
                    sampling_policy=distributed.sampling_policy,
                    view_frames=train_view_frames,
                    view_cameras=train_view_cameras,
                    sample_count=main_stop,
                )

            def _future_train_views(
                sample_fn,
                local_step: int,
                stop: int,
            ) -> tuple[int, ...] | None:
                if rgb_prefetch_lookahead <= 0:
                    return None
                future_views: list[int] = []
                lookahead_stop = min(
                    int(stop), int(local_step) + 1 + rgb_prefetch_lookahead,
                )
                for future_step in range(int(local_step) + 1, lookahead_stop):
                    future_views.extend(int(value) for value in sample_fn(future_step))
                return tuple(future_views) if future_views else None

            def _schedule_train_lookahead(
                sample_fn,
                local_step: int,
                stop: int,
            ) -> None:
                if rgb_prefetch_executor is None:
                    return
                future_views = _future_train_views(sample_fn, local_step, stop)
                if future_views:
                    backend.schedule_train_views(
                        future_views, executor=rgb_prefetch_executor,
                    )

            main_stop = (
                start if main_terminated_all_dead
                else self.config.train.iterations
            )
            if main_stop > start:
                reporter.start_phase(
                    "main", total=self.config.train.iterations, initial=start,
                )
            for step in range(start, main_stop):
                step_begin = time.perf_counter()
                main_steps_executed = True
                last = step
                rank_views = _sample_batch(step)
                view_index = rank_views[0]
                next_rank_views = _future_train_views(
                    _sample_batch, step, main_stop,
                )
                if group is None:
                    backend.prefetch_train_views(
                        rank_views, executor=rgb_prefetch_executor,
                    )
                    _schedule_train_lookahead(_sample_batch, step, main_stop)
                should_log = step == start or step % log_interval == 0
                scalars, sync_timing, pair_log_fields = self._optimizer_update(
                    backend, group, rank_views, step, should_log, next_rank_views,
                )
                training_clocks["optimizer_steps"] = (
                    int(training_clocks["optimizer_steps"]) + 1
                )
                training_clocks["views_seen"] = (
                    int(training_clocks["views_seen"])
                    + distributed.world_size
                )
                training_clocks["wall_seconds"] = (
                    wall_clock_base + max(0.0, time.perf_counter() - begin)
                )
                profile_summary = performance_profile.record(
                    step_seconds=time.perf_counter() - step_begin,
                    timing=sync_timing,
                )
                if should_log:
                    if scalars is None:
                        raise RuntimeError("log step did not collect loss scalars")
                    scalars.update({
                        "step": int(step),
                        "view": int(view_index),
                        "views": [int(value) for value in rank_views],
                        "rank_view_audit": _rank_view_audit(
                            dataset, rank_views,
                        ),
                        "world_size": distributed.world_size,
                        "views_per_step": distributed.world_size,
                        "sampling_policy": distributed.sampling_policy,
                        "world_size_lr_scaling": str(
                            distributed.world_size_lr_scaling
                        ),
                        "transport": (
                            str(group.transport) if group is not None else "off"
                        ),
                        **training_clocks,
                        **sync_timing,
                        **pair_log_fields,
                        **profile_summary,
                    })
                    with train_log.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(
                                scalars, sort_keys=True, allow_nan=False,
                            ) + "\n"
                        )
                    image_loss = scalars["image_loss"]
                    grad_suffix = ""
                    if "grad_xyz_rms" in scalars:
                        grad_suffix = (
                            f", grad_rms xyz {scalars['grad_xyz_rms']:.3e} "
                            f"scale {scalars['grad_scaling_rms']:.3e} "
                            f"oc {scalars['grad_oc_rms']:.3e} "
                            f"rot {scalars['grad_rotation_rms']:.3e}"
                        )
                        if "grad_sh_dc_rms" in scalars:
                            grad_suffix += (
                                f" sh_dc {scalars['grad_sh_dc_rms']:.3e} "
                                f"sh_rest {scalars['grad_sh_rest_rms']:.3e}"
                            )
                    logger.info(
                        "iteration %d/%d: total_loss %.7f (image %.7f, l1 %.7f, "
                        "ssim %.7f, reg_opacity %.7f, reg_scale %.7f, gate %.7f, "
                        "color_cc %.7f%s)",
                        step, self.config.train.iterations, scalars["total_loss"],
                        image_loss, scalars["l1"], scalars["ssim"],
                        scalars["reg_opacity"], scalars["reg_scale"], scalars["gate"],
                        scalars["color_cc"], grad_suffix,
                    )
                relocation = self.config.train.relocation
                if relocation.start < step < relocation.stop and step % relocation.every == 0:
                    include_gate = (
                        relocation.score_mode == "gate_included"
                        and step >= relocation.score_mode_start
                    )
                    result = (
                        group.relocate(
                            step, relocation.mode, include_gate,
                            self.config.train.seed,
                        )
                        if group is not None
                        else backend.relocate_native(
                            step, relocation.mode, include_gate,
                            self.config.train.seed,
                        )
                    )
                    if result["all_dead"]:
                        main_terminated_all_dead = True
                        logger.warning("stopping at %d because every Gaussian is dead", step)
                        reporter.advance(step, scalars)
                        break
                    if result["relocated"]:
                        logger.info("iteration %d: relocated %d Gaussians in Vulkan", step, result["relocated"])
                is_final_main_step = step == self.config.train.iterations - 1
                checkpoint_snapshot = (
                    publish_if_enabled(
                        step,
                        phase=(
                            "main_complete" if is_final_main_step else "main"
                        ),
                        phase_step=step,
                    )
                    if snapshot_cadence(step, snapshot_interval) else None
                )
                if snapshot_cadence(step, snapshot_interval):
                    consider_export_candidate(
                        step,
                        self._evaluate(
                            backend, dataset, step, group=group,
                            clocks=dict(training_clocks),
                        ),
                        checkpoint_snapshot=checkpoint_snapshot,
                    )
                    elapsed = time.perf_counter()-begin
                    step_rate = (step-start+1)/max(elapsed, 1e-9)
                    logger.info(
                        "iteration %d/%d, %.2f steps/s, %.2f views/s",
                        step, self.config.train.iterations, step_rate,
                        step_rate * distributed.world_size,
                    )
                if (
                    wall_eval_interval > 0
                    and (time.perf_counter() - last_wall_eval) >= wall_eval_interval
                ):
                    wall_now = time.perf_counter()
                    consider_export_candidate(
                        step,
                        self._evaluate(
                            backend, dataset, step, group=group,
                            wall_equalized=True,
                            wall_seconds=float(training_clocks["wall_seconds"]),
                            clocks=dict(training_clocks),
                        ),
                    )
                    last_wall_eval = wall_now
                    logger.info(
                        "wall-equalized eval at step %d after %.1fs",
                        step, wall_now - begin,
                    )
                if (
                    checkpoint_snapshot is not None
                    and (
                        not self.config.train.hard_separation
                        or main_terminated_all_dead
                    )
                    and is_final_main_step
                ):
                    final_model_snapshot = checkpoint_snapshot.model
                # Drop multi-GB optimizer arrays immediately. Only a model
                # retained as best/final may survive for imminent export.
                del checkpoint_snapshot
                verify_interval = distributed.verify_replica_interval
                if (
                    group is not None and verify_interval > 0 and step > 0
                    and step % verify_interval == 0
                ):
                    group.assert_replicas(include_optimizer=True)
                    logger.info(
                        "iteration %d: all model and Adam replicas are identical",
                        step,
                    )
                reporter.advance(step, scalars)
            if (
                save_checkpoints
                and main_steps_executed
                and last_published_iteration != last
            ):
                checkpoint_snapshot = publish_runtime_checkpoint(
                    last,
                    phase=(
                        "main_all_dead"
                        if main_terminated_all_dead else "main_complete"
                    ),
                    phase_step=last,
                )
                if (
                    not self.config.train.hard_separation
                    or main_terminated_all_dead
                ):
                    final_model_snapshot = checkpoint_snapshot.model
                del checkpoint_snapshot

            if self.config.train.hard_separation and not main_terminated_all_dead:
                separate_steps = int(self.config.train.hard_separation_steps)
                hard_relocation_steps = frozenset(
                    int(step)
                    for step in self.config.train.hard_relocation_steps
                )
                logger.info(
                    "hard separation schedule: steps=%d relocation_steps=%s "
                    "relocation_score=opacity_gradient",
                    separate_steps, sorted(hard_relocation_steps),
                )
                resuming_hard_phase = resume_phase.startswith("hard_separation")
                if not selected_main_handoff and not resuming_hard_phase:
                    # Complete ordinary model selection before crossing the
                    # phase boundary. The last main step is a valid candidate
                    # even when it is off the periodic checkpoint cadence.
                    if dataset.eval_views and last_eval_step != last:
                        consider_export_candidate(
                            last,
                            self._evaluate(
                                backend, dataset, last, group=group,
                                clocks=dict(training_clocks),
                            ),
                        )
                    if (
                        export_best
                        and best_model is not None
                        and best_metrics is not None
                    ):
                        main_selected_model = best_model
                        main_selected_color = best_color_params
                        main_selected_step = int(best_metrics["step"])
                    else:
                        main_selected_model = backend.model()
                        main_selected_color = (
                            backend.color_params()
                            if self.config.train.color_correction else None
                        )
                        main_selected_step = int(last)
                    publish_selected_checkpoint(
                        main_selected_model,
                        main_selected_color,
                        filename=MAIN_SELECTED_CHECKPOINT,
                        role=MAIN_SELECTED_ROLE,
                        boundary_iteration=self.config.train.iterations - 1,
                        phase="main_complete",
                        phase_step=self.config.train.iterations - 1,
                        source_step=main_selected_step,
                    )
                    if main_selected_step != int(last):
                        if group is not None:
                            group.upload_model(main_selected_model)
                        else:
                            backend.upload_model(main_selected_model)
                    if main_selected_color is not None:
                        for replica in backends:
                            replica.set_color_params(main_selected_color)
                    logger.info(
                        "hard separation starts from selected main snapshot "
                        "step %d (main last step %d)",
                        main_selected_step, last,
                    )
                    del main_selected_model, main_selected_color
                elif selected_main_handoff:
                    main_selected_step = int(
                        (loaded_metadata or {}).get(
                            "selected_source_step",
                            self.config.train.iterations - 1,
                        )
                    )
                    logger.info(
                        "hard separation starts from retained main-selected "
                        "checkpoint source step %d",
                        main_selected_step,
                    )
                else:
                    logger.info(
                        "hard separation resumes directly from phase checkpoint "
                        "step %d; retained main selection is left unchanged",
                        int(completed),
                    )
                # HS is a terminal recovery pass. Its output contract is the
                # state after all hard-phase updates, while main_selected.npz
                # remains the retained soft-gate comparison/fallback model.
                best_model = None
                best_color_params = None
                best_metrics = None
                last_eval_step = None
                last_eval_metrics = None
                # Checkpoints use the globally unique iteration namespace:
                # main phase 0..iterations-1, separation phase
                # iterations..iterations+hard_separation_steps-1.
                separation_start = max(0, start - self.config.train.iterations)
                separation_already_complete = (
                    resume_phase == "hard_separation_complete"
                    or separation_start >= separate_steps
                )
                if separation_already_complete:
                    # The checkpoint already contains the split topology, but
                    # a freshly constructed native backend still has normal-
                    # phase runtime flags.  Restore the config-only separation
                    # state; never dispatch the split shader again.
                    separation = (
                        group.resume_hard_separation(separate_steps)
                        if group is not None
                        else backend.resume_hard_separation(separate_steps)
                    )
                    logger.info(
                        "hard persistence already completed in resumed checkpoint "
                        "(completed local steps=%d, persistent=%d transient=%d); "
                        "skipping optimizer steps",
                        separation_start,
                        separation["persistent"], separation["transient"],
                    )
                elif separation_start == 0:
                    separation = (
                        group.begin_hard_separation(separate_steps)
                        if group is not None
                        else backend.begin_hard_separation(separate_steps)
                    )
                    logger.info(
                        "starting native hard persistence: persistent=%d transient=%d",
                        separation["persistent"], separation["transient"],
                    )
                else:
                    # Topology was already split before this checkpoint. The
                    # native resume entry restores phase-specific LR/regularizer
                    # config without applying the destructive split twice.
                    separation = (
                        group.resume_hard_separation(separate_steps)
                        if group is not None
                        else backend.resume_hard_separation(separate_steps)
                    )
                    logger.info(
                        "resuming native hard persistence at local step %d/%d "
                        "(persistent=%d transient=%d)",
                        separation_start, separate_steps,
                        separation["persistent"], separation["transient"],
                    )
                separation_loop_start = (
                    separate_steps
                    if separation_already_complete else separation_start
                )
                if separation_loop_start < separate_steps:
                    reporter.start_phase(
                        "hard-separation", total=separate_steps,
                        initial=separation_loop_start,
                    )

                def _sample_hard_batch(local_step: int) -> tuple[int, ...]:
                    return sample_rank_views(
                        seed=hard_phase_seed,
                        # Keep the step phase-local for addressable resume. The
                        # separate phase seed above matches FTGSPP's
                        # seed+start_iteration hard-phase reseed; each policy
                        # then consumes its phase-local stream by local_step.
                        step=local_step,
                        world_size=distributed.world_size,
                        num_views=len(dataset.train_views),
                        sampling_policy=distributed.sampling_policy,
                        view_frames=train_view_frames,
                        view_cameras=train_view_cameras,
                        sample_count=separate_steps,
                    )

                for local_step in range(separation_loop_start, separate_steps):
                    step_begin = time.perf_counter()
                    global_step = self.config.train.iterations + local_step
                    last = global_step
                    rank_views = _sample_hard_batch(local_step)
                    next_rank_views = _future_train_views(
                        _sample_hard_batch, local_step, separate_steps,
                    )
                    view_index = rank_views[0]
                    if group is None:
                        backend.prefetch_train_views(
                            rank_views, executor=rgb_prefetch_executor,
                        )
                        _schedule_train_lookahead(
                            _sample_hard_batch, local_step, separate_steps,
                        )
                    should_log = local_step == 0 or local_step % log_interval == 0
                    scalars: dict[str, float] | None = None
                    scalars, sync_timing, pair_log_fields = self._optimizer_update(
                        backend, group, rank_views, local_step, should_log,
                        next_rank_views, phase="hard_separation",
                    )
                    training_clocks["optimizer_steps"] = (
                        int(training_clocks["optimizer_steps"]) + 1
                    )
                    training_clocks["views_seen"] = (
                        int(training_clocks["views_seen"])
                        + distributed.world_size
                    )
                    training_clocks["wall_seconds"] = (
                        wall_clock_base
                        + max(0.0, time.perf_counter() - begin)
                    )
                    profile_summary = performance_profile.record(
                        step_seconds=time.perf_counter() - step_begin,
                        timing=sync_timing,
                    )
                    if should_log:
                        if scalars is None:
                            raise RuntimeError(
                                "hard-separation log step did not collect loss scalars"
                            )
                        scalars.update({
                            "step": int(global_step), "separation_step": int(local_step),
                            "view": int(view_index),
                            "views": [int(value) for value in rank_views],
                            "rank_view_audit": _rank_view_audit(
                                dataset, rank_views,
                            ),
                            "world_size": distributed.world_size,
                            "views_per_step": distributed.world_size,
                            "sampling_policy": distributed.sampling_policy,
                            "phase": "hard_separation",
                            **training_clocks,
                            **sync_timing,
                            **pair_log_fields,
                            **profile_summary,
                        })
                        with train_log.open("a", encoding="utf-8") as stream:
                            stream.write(
                                json.dumps(
                                    scalars, sort_keys=True, allow_nan=False,
                                ) + "\n"
                            )
                        logger.info(
                            "hard separation %d/%d (global step %d): "
                            "total_loss %.7f (image %.7f, l1 %.7f, "
                            "ssim %.7f, reg_opacity %.7f, reg_scale %.7f, "
                            "gate %.7f, color_cc %.7f)",
                            local_step, separate_steps, global_step,
                            scalars["total_loss"], scalars["image_loss"],
                            scalars["l1"], scalars["ssim"],
                            scalars["reg_opacity"], scalars["reg_scale"],
                            scalars["gate"],
                            scalars["color_cc"],
                        )
                    if local_step in hard_relocation_steps:
                        relocation = self.config.train.relocation
                        result = (
                            group.relocate(
                                local_step, relocation.mode,
                                False,
                                self.config.train.seed,
                            )
                            if group is not None
                            else backend.relocate_native(
                                local_step, relocation.mode,
                                False,
                                self.config.train.seed,
                            )
                        )
                        logger.info(
                            "hard separation local step %d: relocated %d "
                            "Gaussians within persistence groups "
                            "(alive=%d, donor_score=opacity_gradient)",
                            local_step, int(result["relocated"]),
                            int(result["alive"]),
                        )
                        if result["all_dead"]:
                            logger.warning("hard separation stopped: every Gaussian is dead")
                            reporter.advance(local_step, scalars)
                            break
                    hard_checkpoint_snapshot = (
                        publish_if_enabled(
                            global_step,
                            phase="hard_separation",
                            phase_step=local_step,
                        )
                        if (
                            snapshot_cadence(local_step, snapshot_interval)
                            and local_step != separate_steps - 1
                        ) else None
                    )
                    if snapshot_cadence(local_step, snapshot_interval):
                        consider_export_candidate(
                            global_step,
                            self._evaluate(
                                backend, dataset, global_step, group=group,
                                clocks=dict(training_clocks),
                            ),
                            track_best=False,
                        )
                    del hard_checkpoint_snapshot
                    verify_interval = distributed.verify_replica_interval
                    if (
                        group is not None and verify_interval > 0
                        and local_step > 0
                        and local_step % verify_interval == 0
                    ):
                        group.assert_replicas(include_optimizer=True)
                        logger.info(
                            "hard separation step %d: all model and Adam "
                            "replicas are identical",
                            local_step,
                        )
                    reporter.advance(local_step, scalars)
                separated: NativeGaussians | None = None
                if separation_already_complete:
                    # The loaded immutable checkpoint already is the final
                    # resumable snapshot. Do not rewrite/collide with it.
                    # The upload-time checkpoint model was released before the
                    # training loop. Download the already-separated native
                    # state once, only when it is immediately needed for export.
                    separated = backend.model()
                else:
                    # Serialize the post-separation optimizer state before
                    # writing standalone artifacts.
                    if save_checkpoints:
                        separated_snapshot = publish_runtime_checkpoint(
                            last,
                            phase="hard_separation_complete",
                            phase_step=max(
                                0, last-self.config.train.iterations,
                            ),
                        )
                        separated = separated_snapshot.model
                        # Release multi-GB optimizer arrays immediately, retaining
                        # only the model host copy for imminent exports.
                        del separated_snapshot
                    else:
                        separated = backend.model()
                if separated is None:
                    separated = backend.model()
                final_model_snapshot = separated

            if self.config.train.hard_separation and main_terminated_all_dead:
                logger.warning(
                    "skipping hard separation because main training terminated "
                    "with an all-dead Gaussian topology"
                )

            hard_phase_output = bool(
                self.config.train.hard_separation
                and not main_terminated_all_dead
            )
            if dataset.eval_views and last_eval_step != last:
                consider_export_candidate(
                    last,
                    self._evaluate(
                        backend, dataset, last, group=group,
                        clocks=dict(training_clocks),
                    ),
                    track_best=not hard_phase_output,
                )

            if (
                export_best
                and not hard_phase_output
                and best_model is not None
                and best_metrics is not None
            ):
                final = best_model
                final_color_params = best_color_params
                primary_metric = primary_metric_name(
                    str(best_metrics["selector"])
                )
                selection_ssim_metric = selection_ssim_name(
                    str(best_metrics["selector"])
                )
                selection: dict[str, object] = {
                    "source": "best_eval",
                    "step": int(best_metrics["step"]),
                    "last_step": int(last),
                    "psnr": float(best_metrics["psnr"]),
                    "ssim": float(best_metrics["ssim"]),
                    "ssim_variant": best_metrics.get(
                        "ssim_variant", "SSIM_2",
                    ),
                    "ssim_data_range": 2.0,
                    "selection_ssim_metric": selection_ssim_metric,
                    "views": int(best_metrics.get("views", 0)),
                    "selector": str(best_metrics["selector"]),
                    "primary_metric": primary_metric,
                    "primary_score": float(best_metrics[primary_metric]),
                    **joint_selection_fields(
                        float(best_metrics[primary_metric]),
                        float(best_metrics[selection_ssim_metric]),
                        ssim_metric=selection_ssim_metric,
                    ),
                    "candidate_evaluation_scope": str(
                        best_metrics.get("evaluation_scope", "candidate_subset")
                    ),
                    "candidate_views": int(best_metrics.get("views", 0)),
                    "selector_constraints": best_metrics[
                        "selector_constraints"
                    ],
                    "selection_metrics": {
                        key: value for key, value in best_metrics.items()
                        if key not in {
                            "selector", "selector_constraints",
                            "evaluation_identity",
                        }
                    },
                }
                if best_metrics.get("foreground_psnr") is not None:
                    selection["foreground_psnr"] = float(
                        best_metrics["foreground_psnr"]
                    )
                for key in ("foreground_ssim", "full_ssim"):
                    if best_metrics.get(key) is not None:
                        selection[key] = float(best_metrics[key])
                if int(best_metrics["step"]) != int(last):
                    if primary_metric == "psnr":
                        logger.info(
                            "exporting best hold-out snapshot from step %d "
                            "(joint %.3f, psnr %.3f, SSIM %.5f) instead of "
                            "last step %d",
                            selection["step"], selection["joint_score"],
                            selection["primary_score"],
                            selection["ssim"], last,
                        )
                    else:
                        logger.info(
                            "exporting best hold-out snapshot from step %d "
                            "(joint %.3f, %s %.3f, composite PSNR %.3f, "
                            "foreground SSIM %.5f) instead of last step %d",
                            selection["step"], selection["joint_score"],
                            primary_metric,
                            selection["primary_score"], selection["psnr"],
                            selection.get("foreground_ssim", selection["ssim"]),
                            last,
                        )
            else:
                final = final_model_snapshot
                if final is None:
                    final = backend.model()
                final_color_params = (
                    backend.color_params()
                    if self.config.train.color_correction else None
                )
                selection = {
                    "source": "last",
                    "step": int(last),
                    "last_step": int(last),
                    "ssim_variant": "SSIM_2",
                    "ssim_data_range": 2.0,
                }
                selected_metrics = (
                    last_eval_metrics
                    if last_eval_metrics is not None
                    and int(last_eval_metrics["step"]) == int(last)
                    else None
                )
                if selected_metrics is not None:
                    selection["psnr"] = float(selected_metrics["psnr"])
                    selected_ssim = _optional_metric(selected_metrics, "ssim")
                    if selected_ssim is not None:
                        selection["ssim"] = selected_ssim
                    selection["views"] = int(selected_metrics.get("views", 0))
                    primary_metric = primary_metric_name(
                        str(selected_metrics["selector"])
                    )
                    selection_ssim_metric = selection_ssim_name(
                        str(selected_metrics["selector"])
                    )
                    selection["ssim_variant"] = selected_metrics.get(
                        "ssim_variant", "SSIM_2",
                    )
                    selection["selection_ssim_metric"] = selection_ssim_metric
                    selection["selector"] = str(selected_metrics["selector"])
                    selection["primary_metric"] = primary_metric
                    primary_score = _optional_metric(
                        selected_metrics, primary_metric,
                    )
                    selection_ssim = _optional_metric(
                        selected_metrics, selection_ssim_metric,
                    )
                    if primary_score is not None:
                        selection["primary_score"] = primary_score
                    if (
                        primary_score is not None
                        and selection_ssim is not None
                    ):
                        selection.update(joint_selection_fields(
                            primary_score,
                            selection_ssim,
                            ssim_metric=selection_ssim_metric,
                        ))
                    for key in ("foreground_psnr", "foreground_ssim", "full_ssim"):
                        value = _optional_metric(selected_metrics, key)
                        if value is not None:
                            selection[key] = value
                    selection["candidate_evaluation_scope"] = str(
                        selected_metrics.get(
                            "evaluation_scope", "candidate_subset",
                        )
                    )
                    selection["candidate_views"] = int(
                        selected_metrics.get("views", 0)
                    )
                if (
                    not hard_phase_output
                    and export_best
                    and dataset.eval_views
                    and best_metrics is None
                ):
                    selection["fallback_reason"] = "no_candidate_passed_selector_gates"
                    selection["selector"] = str(self.config.eval.selector)
                    selection["selector_constraints"] = {
                        "min_silhouette_iou": (
                            self.config.eval.selector_min_silhouette_iou
                        ),
                        "max_background_opacity_leakage": (
                            self.config.eval.selector_max_background_opacity_leakage
                        ),
                        "max_foreground_holes": (
                            self.config.eval.selector_max_foreground_holes
                        ),
                        "min_boundary_fscore": (
                            self.config.eval.selector_min_boundary_fscore
                        ),
                    }

            if dataset.eval_views:
                selected_is_current = int(selection["step"]) == int(last)
                if not selected_is_current:
                    if group is not None:
                        group.upload_model(final)
                    else:
                        backend.upload_model(final)
                if final_color_params is not None:
                    for replica in backends:
                        replica.set_color_params(final_color_params)
                full_metrics = self._evaluate(
                    backend,
                    dataset,
                    int(selection["step"]),
                    group=group,
                    clocks=dict(training_clocks),
                    result_filename="full_validation_metrics.jsonl",
                    full_validation=True,
                )
                selection["full_validation_metrics"] = {
                    key: value for key, value in full_metrics.items()
                    if key not in {"foreground_views", "evaluation_identity"}
                }
                full_primary_metric = primary_metric_name(
                    str(self.config.eval.selector)
                )
                selection["full_validation_primary_metric"] = (
                    full_primary_metric
                )
                selection["full_validation_primary_score"] = (
                    None
                    if full_metrics.get(full_primary_metric) is None
                    else float(full_metrics[full_primary_metric])
                )
                selection["full_validation_evaluation_scope"] = str(
                    full_metrics.get("evaluation_scope", "full_validation")
                )
                selection["full_validation_views"] = int(
                    full_metrics.get("views", 0)
                )
                selection["full_validation_passed_selector_gates"] = (
                    holdout_is_better(
                        full_metrics,
                        None,
                        selector=self.config.eval.selector,
                        min_silhouette_iou=(
                            self.config.eval.selector_min_silhouette_iou
                        ),
                        max_background_opacity_leakage=(
                            self.config.eval.selector_max_background_opacity_leakage
                        ),
                        max_foreground_holes=(
                            self.config.eval.selector_max_foreground_holes
                        ),
                        min_boundary_fscore=(
                            self.config.eval.selector_min_boundary_fscore
                        ),
                    )
                )

            if self.config.train.hard_separation and not main_terminated_all_dead:
                selected_checkpoint = publish_selected_checkpoint(
                    final,
                    final_color_params,
                    filename=HARD_FINAL_CHECKPOINT,
                    role=HARD_FINAL_ROLE,
                    boundary_iteration=int(last),
                    phase="hard_separation_complete",
                    phase_step=max(
                        0, int(last) - self.config.train.iterations,
                    ),
                    source_step=int(selection["step"]),
                )
                selection["phase"] = "hard_separation"
                selection["hard_separation_schedule"] = {
                    "steps": int(separate_steps),
                    "relocation_steps": sorted(hard_relocation_steps),
                    "relocation_score_mode": "default",
                    "relocation_groups": "binary_persistence",
                }
                main_selected_path = checkpoint_dir / MAIN_SELECTED_CHECKPOINT
                if main_selected_path.is_file():
                    selection["parent_checkpoint_artifact"] = str(
                        main_selected_path.relative_to(self.output_path)
                    )
            else:
                selected_checkpoint = publish_selected_checkpoint(
                    final,
                    final_color_params,
                    filename=MAIN_SELECTED_CHECKPOINT,
                    role=(
                        SELECTED_MODEL_ROLE
                        if main_terminated_all_dead else MAIN_SELECTED_ROLE
                    ),
                    boundary_iteration=(
                        int(last)
                        if main_terminated_all_dead
                        else self.config.train.iterations - 1
                    ),
                    phase=(
                        "main_all_dead"
                        if main_terminated_all_dead else "main_complete"
                    ),
                    phase_step=(
                        int(last)
                        if main_terminated_all_dead
                        else self.config.train.iterations - 1
                    ),
                    source_step=int(selection["step"]),
                )
                selection["phase"] = "main"
            selection["checkpoint_artifact"] = str(
                selected_checkpoint.relative_to(self.output_path)
            )

            from ftgspp.training.temporal_metrics import (
                temporal_support_diagnostics,
            )
            configured_cutoff = float(self.config.model.marginal_threshold)
            diagnostic_cutoffs = sorted({
                *(float(value) for value in self.config.eval.temporal_cutoffs),
                configured_cutoff,
            })
            selection["temporal_support"] = temporal_support_diagnostics(
                final,
                [
                    float(view.timestamp)
                    for view in (*dataset.train_views, *dataset.eval_views)
                ],
                diagnostic_cutoffs,
            )
            selection["temporal_support"]["configured_cutoff"] = configured_cutoff
            selection["training_dataset_fingerprint"] = str(
                dataset.checkpoint_metadata()["fingerprint"]
            )

            if final_color_params is not None:
                color_path = self.output_path / "color_correction.npz"
                color_tmp = color_path.with_name(color_path.name + ".tmp.npz")
                np.savez(
                    color_tmp,
                    params=np.asarray(final_color_params, np.float32),
                    camera_ids=np.asarray(sorted(dataset.cameras), np.int64),
                    snapshot_step=np.asarray(selection["step"], np.int64),
                )
                color_tmp.replace(color_path)
                selection["color_correction"] = {
                    "artifact": color_path.name,
                    "snapshot_step": int(selection["step"]),
                    "camera_ids": [int(value) for value in sorted(dataset.cameras)],
                }

            model_path = self.output_path / "gaussians.npz"
            save_model(model_path, final)
            selection["model_artifact"] = model_path.name
            selection["model_sha256"] = file_sha256(model_path)
            from ftgspp.io.viewer_manifest import write_training_manifest
            write_training_manifest(
                model_path, final, dataset, self.config, selection["model_sha256"],
            )
            selection_path = self.output_path / "export_selection.json"
            selection_tmp = selection_path.with_suffix(".json.tmp")
            selection_tmp.write_text(
                json.dumps(
                    selection, indent=2, sort_keys=True, allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )
            selection_tmp.replace(selection_path)
            return final
        finally:
            reporter.close()
            if rgb_prefetch_executor is not None:
                rgb_prefetch_executor.shutdown(
                    wait=True, cancel_futures=True)
            try:
                native_performance = (
                    group.performance_snapshot()
                    if group is not None
                    else {
                        "world_size": 1,
                        "transport": "off",
                        "replicas": [backend.performance_snapshot()],
                    }
                )
                performance_report = performance_profile.report(
                    clocks=training_clocks,
                    native=native_performance,
                    quality=_quality_for_performance_summary(self.output_path),
                )
                performance_path = self.output_path / "performance_summary.json"
                temporary_path = performance_path.with_suffix(".json.tmp")
                temporary_path.write_text(
                    json.dumps(
                        performance_report,
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    ) + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(performance_path)
            except Exception:
                logger.exception("failed to publish performance_summary.json")
            if group is not None:
                group.close()
            else:
                backend.close()

    def run(self) -> None:
        """One public training pipeline: initialize/resume, train, export."""
        cv2.setNumThreads(1)
        completed = self.output_path / "gaussians.npz"
        if completed.exists() and not self.config.run.overwrite_completed:
            raise FileExistsError(
                f"refusing to overwrite {completed}; set run.overwrite_completed=true"
            )
        self.output_path.mkdir(parents=True, exist_ok=True)
        resume = bool(
            self.config.train.resume
            or self.config.train.resume_checkpoint is not None
        )
        with _run_file_logging(self.output_path / "run.log", resume=resume):
            logger.info("persistent log: %s", self.output_path / "run.log")
            logger.info(
                "pipeline mode=%s init=%s objective=%s eval=%s selector=%s",
                self.config.pipeline.mode,
                self.config.init.backend,
                self.config.pipeline.image_objective,
                self.config.pipeline.eval_protocol,
                self.config.eval.selector,
            )
            dataset = self._preflight()
            save_resolved_config(
                self.config, self.output_path / "config.resolved.yaml",
            )
            initial = []
            if resume:
                initial.append(None)
            else:
                initial.append(initialize_explicit(self.config, dataset))
            # pop transfers ownership so the caller does not retain a full host
            # copy after the native upload inside _train.
            model = self._train(dataset, initial.pop())
            export_4dgs(
                self.output_path / self.config.run.export_filename, model,
            )
            logger.info(
                "FTGSPP native Vulkan pipeline completed: %s", self.output_path,
            )
