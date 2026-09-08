"""Hydra-compatible FTGSPP pipeline with one native Vulkan training owner."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig

from ftgspp.config import from_hydra, save_composed_config, save_resolved_config
from ftgspp.config.schema import AppConfig
from ftgspp.config.validate import validate_resolved_config
from ftgspp.native_runtime.backend import (
    NativeVulkanBackend, SharedRgbViewCache, camera_extent,
)
from ftgspp.native_runtime.checkpoint import (
    CheckpointSnapshot,
    NATIVE_ABI,
    build_checkpoint_metadata,
    capture_checkpoint_snapshot,
    collect_runtime_provenance,
    latest_checkpoint,
    load_checkpoint_full,
    load_model,
    publish_checkpoint,
    resolve_checkpoint_path,
    save_model,
    validate_or_infer_training_clocks,
    validate_resume_compatibility,
)
from ftgspp.native_runtime.dataset import NativeDataset
from ftgspp.native_runtime.distributed import (
    DistributedPlan,
    SynchronousBackendGroup,
    sample_rank_views,
    sampling_phase_seed,
    _log_geometry_grads_enabled,
)
from ftgspp.native_runtime.distributed_contract import (
    shard_resumed_relocation_statistics,
    validate_checkpoint_optimizer_state,
)
from ftgspp.native_runtime.initialize import initialize_explicit
from ftgspp.native_runtime.model import NativeGaussians
from ftgspp.native_runtime.ply import export_4dgs

logger = logging.getLogger("ftgspp.native")


def _psnr(prediction: np.ndarray, reference: np.ndarray) -> float:
    mse = float(np.mean((prediction-reference)**2))
    return -10.0*np.log10(max(mse, 1e-12))


def _ssim(prediction: np.ndarray, reference: np.ndarray) -> float:
    return _EvalSsimWorkspace().ssim(prediction, reference)


class _EvalSsimWorkspace:
    """Reuse OpenCV/NumPy scratch so 90-view eval does not allocate per blur."""

    def __init__(self) -> None:
        self._bufs: dict[str, np.ndarray] = {}

    def _buf(self, key: str, shape: tuple[int, ...]) -> np.ndarray:
        array = self._bufs.get(key)
        if array is None or array.shape != shape:
            array = np.empty(shape, dtype=np.float32)
            self._bufs[key] = array
        return array

    def ssim(self, prediction: np.ndarray, reference: np.ndarray) -> float:
        c1, c2 = 0.01**2, 0.03**2
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
        return float(np.mean(value))


def _finite_rgb(array: np.ndarray) -> bool:
    return bool(np.isfinite(array.min()) and np.isfinite(array.max()))


def _rank_view_audit(
    dataset: NativeDataset, view_indices: tuple[int, ...], microsteps: int = 1,
) -> list[dict[str, int | float]]:
    """Describe every rank-major ``(rank, microstep)`` global slot."""

    result: list[dict[str, int | float]] = []
    for slot, view_index in enumerate(view_indices):
        view = dataset.train_views[int(view_index)]
        result.append({
            "rank": int(slot // microsteps),
            "microstep": int(slot % microsteps),
            "view": int(view_index),
            "frame": int(view.frame),
            "camera": int(view.camera_id),
            "timestamp": float(view.timestamp),
        })
    return result


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

    def __init__(self, *, global_batch_size: int) -> None:
        self.global_batch_size = int(global_batch_size)
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
            "views_per_second": step_rate * self.global_batch_size,
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
            "global_batch_size": self.global_batch_size,
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


def _validate_resume_phase_cursor(
    metadata: dict[str, object],
    *,
    completed: int,
    main_iterations: int,
    hard_separation: bool,
    hard_steps: int = 5000,
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
    composed: DictConfig

    @classmethod
    def from_hydra(cls, config: DictConfig) -> "NativeRunner":
        return cls(from_hydra(config), config)

    @property
    def output_path(self) -> Path:
        return self.config.run.output_path

    def _preflight(self) -> NativeDataset:
        if self.config.train.lpips_loss:
            # LPIPS stays a Torch host plugin — never a Vulkan shader.
            from ftgspp.native_runtime.lpips_plugin import LpipsPlugin
            weight = float(getattr(self.config.train, "lpips_weight", 0.01))
            try:
                self._lpips = LpipsPlugin(weight=weight)
                self._lpips._ensure()
            except RuntimeError as exc:
                raise RuntimeError(
                    "train.lpips_loss=true requires optional Torch LPIPS deps "
                    "(pip install -e '.[lpips]'); LPIPS is not implemented as a Vulkan shader"
                ) from exc
            logger.info(
                "LPIPS enabled via Torch host plugin (weight=%.4f)%s",
                weight,
                " after native color correction" if self.config.train.color_correction else " on raw RGB",
            )
        else:
            self._lpips = None
        if self.config.train.color_correction:
            logger.info("native Vulkan color correction enabled (per-camera affine)")
        dataset = NativeDataset.from_config(self.config)
        self.config.data.train_cameras = dataset.train_cameras
        self.config.data.eval_cameras = dataset.eval_cameras
        self.config.data.camera_split_source = dataset.camera_split_source
        validate_resolved_config(self.config, dataset.cameras)
        first = dataset.train_views[0]
        image = first.load_rgb()
        logger.info(
            "SelfCap native dataset: %d frames, %d cameras, %d train views, %d eval views, first=%s shape=%s",
            len(dataset.frames), len(dataset.cameras), len(dataset.train_views), len(dataset.eval_views), first.path, image.shape,
        )
        return dataset

    def _evaluate(
        self,
        backend: NativeVulkanBackend,
        dataset: NativeDataset,
        step: int,
        *,
        group: SynchronousBackendGroup | None = None,
        wall_equalized: bool = False,
        wall_seconds: float | None = None,
        clocks: dict[str, int | float] | None = None,
    ) -> dict[str, object]:
        if not dataset.eval_views:
            return {}
        selected = [
            index for index, view in enumerate(dataset.eval_views)
            if (view.frame-dataset.frames[0]) % self.config.eval.frame_interval == 0
        ]
        if not selected:
            return {}
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
                "count": 0, "cameras": {},
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
            begin_eval = getattr(replica.native, "begin_eval_val_view", None)
            submit_eval = getattr(replica.native, "submit_eval_rgb_readback", None)
            complete_eval = getattr(replica.native, "complete_eval_rgb", None)
            drain_eval = getattr(replica.native, "drain_eval", None)
            ssim_workspace = _EvalSsimWorkspace()
            pipeline_eval = all(
                callable(fn) for fn in (
                    begin_eval, submit_eval, complete_eval, drain_eval,
                )
            )

            def score_view(index: int, prediction: np.ndarray) -> None:
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
                if (
                    prediction.ndim != 3
                    or prediction.shape[-1] != 3
                    or prediction.shape != reference.shape
                ):
                    raise RuntimeError(
                        "distributed evaluation RGB layout mismatch on rank "
                        f"{rank}, view {index}: prediction={prediction.shape}, "
                        f"reference={reference.shape}"
                    )
                if not _finite_rgb(prediction) or not _finite_rgb(reference):
                    raise FloatingPointError(
                        "distributed evaluation received non-finite RGB on "
                        f"rank {rank}, view {index}"
                    )
                prediction_metric = (
                    np.clip(prediction, 0.0, 1.0) * alpha[..., None]
                )
                reference_metric = reference * alpha[..., None]
                psnr = _psnr(prediction_metric, reference_metric)
                ssim = ssim_workspace.ssim(prediction_metric, reference_metric)
                if not np.isfinite(psnr) or not np.isfinite(ssim):
                    raise FloatingPointError(
                        "distributed evaluation produced non-finite metrics on "
                        f"rank {rank}, view {index}"
                    )
                sums["psnr_sum"] = float(sums["psnr_sum"]) + psnr
                sums["ssim_sum"] = float(sums["ssim_sum"]) + ssim
                sums["count"] = int(sums["count"]) + 1
                camera_key = str(dataset.eval_views[index].camera_id)
                cameras = sums["cameras"]
                if not isinstance(cameras, dict):
                    raise RuntimeError(
                        "evaluation camera accumulator is invalid")
                camera = cameras.setdefault(camera_key, {
                    "psnr_sum": 0.0, "ssim_sum": 0.0, "count": 0,
                })
                camera["psnr_sum"] = float(camera["psnr_sum"]) + psnr
                camera["ssim_sum"] = float(camera["ssim_sum"]) + ssim
                camera["count"] = int(camera["count"]) + 1

            if pipeline_eval:
                drain_eval()
            for offset in range(0, len(indices), eval_window):
                chunk = indices[offset:offset + eval_window]
                if callable(prefetch_eval):
                    prefetch_eval(chunk, executor=eval_executor)
                next_chunk = indices[
                    offset + eval_window:offset + 2 * eval_window
                ]
                if (
                    next_chunk
                    and eval_executor is not None
                    and callable(schedule_eval)
                ):
                    schedule_eval(next_chunk, executor=eval_executor)
                if pipeline_eval:
                    begin_eval(int(chunk[0]))
                    for cursor, index in enumerate(chunk):
                        submit_eval()
                        prediction = np.asarray(
                            complete_eval(), dtype=np.float32,
                        )
                        if cursor + 1 < len(chunk):
                            begin_eval(int(chunk[cursor + 1]))
                        score_view(int(index), prediction)
                    continue
                for index in chunk:
                    prediction = np.asarray(
                        replica.render_eval(index), dtype=np.float32,
                    )
                    score_view(int(index), prediction)
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
        metrics: dict[str, object] = {
            "step": int(step),
            "psnr": psnr_sum / count,
            "ssim": ssim_sum / count,
            "views": count,
            "rank_eval_counts": [int(values["count"]) for values in partials],
        }
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
                    "psnr_sum": 0.0, "ssim_sum": 0.0, "count": 0,
                })
                for key in ("psnr_sum", "ssim_sum"):
                    target[key] = float(target[key]) + float(raw[key])
                target["count"] = int(target["count"]) + int(raw["count"])
        camera_metrics: dict[str, dict[str, float | int]] = {}
        for camera_key in sorted(camera_totals, key=int):
            raw = camera_totals[camera_key]
            camera_count = int(raw["count"])
            if camera_count <= 0:
                raise RuntimeError("evaluation camera has no reduced views")
            record: dict[str, float | int] = {
                "psnr": float(raw["psnr_sum"]) / camera_count,
                "ssim": float(raw["ssim_sum"]) / camera_count,
                "views": camera_count,
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
            worst_ssim = min(
                camera_metrics, key=lambda key: float(camera_metrics[key]["ssim"])
            )
            metrics["worst_psnr_camera"] = int(worst_psnr)
            metrics["worst_camera_psnr"] = float(camera_metrics[worst_psnr]["psnr"])
            metrics["worst_ssim_camera"] = int(worst_ssim)
            metrics["worst_camera_ssim"] = float(camera_metrics[worst_ssim]["ssim"])
        metrics["eval_seconds"] = float(time.perf_counter() - eval_begin)
        with (self.output_path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n"
            )
        logger.info(
            "iteration %d: eval PSNR %.3f SSIM %.5f over %d views in %.2fs",
            step, metrics["psnr"], metrics["ssim"], len(selected),
            metrics["eval_seconds"],
        )
        return metrics

    def _train(self, dataset: NativeDataset) -> NativeGaussians | None:
        checkpoint_dir = self.output_path / "ckpt"
        distributed = DistributedPlan.from_config(self.config)
        provenance = collect_runtime_provenance(
            device_ids=list(distributed.device_ids)
        )
        loaded_metadata: dict[str, object] | None = None
        loaded_clocks: dict[str, int | float] | None = None
        clock_resume_warning: str | None = None
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
            rng_state = dict(loaded.rng_state)
            loaded_metadata = dict(loaded.metadata)
            loaded_clocks, clock_resume_warning = (
                validate_or_infer_training_clocks(
                    loaded_metadata, iteration=completed,
                    fallback_world_size=distributed.world_size,
                    fallback_microsteps=distributed.microsteps,
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
            )
            start = completed+1
            relocation_rng = np.random.default_rng()
            relocation_rng.bit_generator.state = rng_state
        else:
            model = load_model(self.output_path / "init.npz")
            optimizer, completed, start = {}, -1, 0
            resume_phase = "none"
            relocation_rng = np.random.default_rng(self.config.train.seed)
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
            global_batch_size=distributed.global_batch_size,
        )
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
                    collective_sparse=distributed.collective_sparse,
                    collective_codec_policy=(
                        distributed.collective_codec_policy),
                    local_microsteps=distributed.microsteps,
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
            device_metadata = [replica.device_metadata() for replica in backends]
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
            from ftgspp.native_runtime.backend import resolve_view_load_options
            view_opts = resolve_view_load_options(self.config)
            shared_rgb_cache = None
            if distributed.enabled and view_opts["mode"] == "lazy":
                shared_rgb_cache = SharedRgbViewCache(max(
                    int(view_opts["rgb_cache"]), distributed.world_size,
                ))
            elif distributed.enabled and view_opts["mode"] == "resident":
                view_count = len(dataset.train_views) + len(dataset.eval_views)
                shared_rgb_cache = SharedRgbViewCache(
                    max(int(view_count), distributed.world_size),
                    pin=True,
                )
                if view_opts["cache_on_device"]:
                    logger.warning(
                        "sync shared resident ignores image_cache_device=gpu; "
                        "using one pinned host RGB pool bound per step"
                    )
                logger.info(
                    "sync shared resident: one host RGB pool for %d views "
                    "across world_size=%d (not duplicated per rank)",
                    view_count, distributed.world_size,
                )
            logger.info(
                "registering views on %d native replica(s) devices=%s "
                "(mode=%s, image_cache=%s, rgb_cache=%s, workers=%s)",
                distributed.world_size,
                list(distributed.device_ids),
                view_opts["mode"],
                view_opts["image_cache_device"],
                view_opts["rgb_cache"],
                view_opts["workers"],
            )
            for replica in backends:
                replica.upload_dataset(
                    dataset,
                    view_load_mode=str(view_opts["mode"]),
                    cache_on_device=bool(view_opts["cache_on_device"]),
                    rgb_cache=int(view_opts["rgb_cache"]),
                    workers=int(view_opts["workers"]),
                    shared_rgb_cache=shared_rgb_cache,
                )
            rgb_workers = int(view_opts["workers"])
            if str(view_opts["mode"]) == "lazy" and rgb_workers > 0:
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
                if group is None:
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
            optimizer.clear()
            del optimizer, model

            last_published_iteration: int | None = None
            final_model_snapshot: NativeGaussians | None = None
            need_final_model = True
            training_clocks: dict[str, int | float] = (
                dict(loaded_clocks) if loaded_clocks is not None else {
                    "optimizer_steps": 0,
                    "microsteps": 0,
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
                snapshot = capture_checkpoint_snapshot(
                    backend, iteration, relocation_rng,
                    optimizer_state=checkpoint_optimizer,
                    metadata=metadata,
                )
                publish_checkpoint(
                    checkpoint_dir, snapshot,
                    filename=filename, update_latest=True,
                )
                last_published_iteration = int(iteration)
                return snapshot
            logger.info(
                "training mode=%s world_size=%d per_device_batch=1 global_batch=%d "
                "reduction=%s transport=%s sampling=%s",
                distributed.mode,
                distributed.world_size,
                distributed.global_batch_size,
                distributed.reduction,
                group.transport if group is not None else distributed.backend,
                distributed.sampling_policy,
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
            eval_interval = int(os.environ.get("FTGSPP_TRAIN_EVAL_INTERVAL", "2000"))
            log_interval = max(int(os.environ.get("FTGSPP_TRAIN_LOG_INTERVAL", "2000")), 1)
            train_log = self.output_path / "train_metrics.jsonl"
            last = start-1
            main_terminated_all_dead = resume_phase == "main_all_dead"
            main_steps_executed = False
            begin = time.perf_counter()
            wall_eval_interval = float(
                getattr(distributed, "wall_eval_interval_seconds", 0.0) or 0.0
            )
            last_wall_eval = begin
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
                    microsteps=distributed.microsteps,
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
                lpips_extra: dict[str, float] | None = None
                scalars: dict[str, float] | None = None
                sync_timing: dict[str, float] = {}
                pair_log_fields: dict[str, object] = {}
                if group is not None:
                    sync_result = group.train_step(
                        rank_views, step, collect_loss=should_log,
                        next_view_indices=next_rank_views,
                    )
                    sync_timing = _sync_train_timing(sync_result)
                    pair_timing, pair_log_fields = (
                        _pair_exchange_log_fields(sync_result)
                    )
                    sync_timing.update(pair_timing)
                    if should_log:
                        loss = sync_result["loss"]
                        if not isinstance(loss, dict):
                            raise RuntimeError(
                                "synchronous log step did not return loss scalars"
                            )
                        scalars = dict(loss)
                elif self._lpips is not None:
                    ref = (
                        backend.cached_view_image(is_train=True, index=view_index)
                        .astype(np.float32) / 255.0
                    )
                    collect_grads = should_log and _log_geometry_grads_enabled()
                    if collect_grads:
                        backend.set_collect_geometry_grad_stats(True)
                    try:
                        lpips_extra = backend.train_view(
                            view_index, step, collect_loss=should_log,
                            lpips=self._lpips, reference_rgb=ref)
                        if should_log:
                            scalars = backend.last_loss()
                            if collect_grads:
                                scalars.update(backend.geometry_gradient_stats())
                    finally:
                        if collect_grads:
                            backend.set_collect_geometry_grad_stats(False)
                else:
                    collect_grads = should_log and _log_geometry_grads_enabled()
                    if collect_grads:
                        backend.set_collect_geometry_grad_stats(True)
                    try:
                        backend.train_view(
                            view_index, step, collect_loss=should_log)
                        if should_log:
                            scalars = backend.last_loss()
                            if collect_grads:
                                scalars.update(backend.geometry_gradient_stats())
                    finally:
                        if collect_grads:
                            backend.set_collect_geometry_grad_stats(False)
                training_clocks["optimizer_steps"] = (
                    int(training_clocks["optimizer_steps"]) + 1
                )
                training_clocks["microsteps"] = (
                    int(training_clocks["microsteps"]) + distributed.microsteps
                )
                training_clocks["views_seen"] = (
                    int(training_clocks["views_seen"])
                    + distributed.microsteps * distributed.world_size
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
                    if lpips_extra:
                        scalars.update(lpips_extra)
                        # LpipsPlugin.loss is already weighted by
                        # train.lpips_weight, exactly matching the injected
                        # pixel gradient. Keep the logged objective identical
                        # to the gradient actually optimized.
                        scalars["total_loss"] += float(lpips_extra["lpips"])
                    scalars.update({
                        "step": int(step),
                        "view": int(view_index),
                        "views": [int(value) for value in rank_views],
                        "rank_view_audit": _rank_view_audit(
                            dataset, rank_views, distributed.microsteps,
                        ),
                        "world_size": distributed.world_size,
                        "global_batch_size": distributed.global_batch_size,
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
                        "color_cc %.7f%s%s)",
                        step, self.config.train.iterations, scalars["total_loss"],
                        image_loss, scalars["l1"], scalars["ssim"],
                        scalars["reg_opacity"], scalars["reg_scale"], scalars["gate"],
                        scalars["color_cc"],
                        f", lpips {scalars['lpips']:.7f}" if "lpips" in scalars else "",
                        grad_suffix,
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
                        break
                    if result["relocated"]:
                        logger.info("iteration %d: relocated %d Gaussians in Vulkan", step, result["relocated"])
                if eval_interval > 0 and step > 0 and step % eval_interval == 0:
                    self._evaluate(
                        backend, dataset, step, group=group,
                        clocks=dict(training_clocks),
                    )
                    elapsed = time.perf_counter()-begin
                    step_rate = (step-start+1)/max(elapsed, 1e-9)
                    logger.info(
                        "iteration %d/%d, %.2f steps/s, %.2f views/s",
                        step, self.config.train.iterations, step_rate,
                        step_rate * distributed.global_batch_size,
                    )
                if (
                    wall_eval_interval > 0
                    and (time.perf_counter() - last_wall_eval) >= wall_eval_interval
                ):
                    wall_now = time.perf_counter()
                    self._evaluate(
                        backend, dataset, step, group=group,
                        wall_equalized=True,
                        wall_seconds=float(training_clocks["wall_seconds"]),
                        clocks=dict(training_clocks),
                    )
                    last_wall_eval = wall_now
                    logger.info(
                        "wall-equalized eval at step %d after %.1fs",
                        step, wall_now - begin,
                    )
                interval = self.config.train.checkpoint_interval
                is_final_main_step = step == self.config.train.iterations - 1
                if (
                    interval > 0 and step > 0 and step % interval == 0
                ):
                    checkpoint_snapshot = publish_runtime_checkpoint(
                        step,
                        phase=(
                            "main_complete"
                            if is_final_main_step
                            else "main"
                        ),
                        phase_step=step,
                    )
                    if (
                        (
                            not self.config.train.hard_separation
                            or main_terminated_all_dead
                        )
                        and is_final_main_step
                        and need_final_model
                    ):
                        final_model_snapshot = checkpoint_snapshot.model
                    # Drop multi-GB optimizer arrays immediately. Only the
                    # final model host copy may survive for imminent export.
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
            if (
                main_steps_executed
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
                separate_steps = 5000
                # Checkpoints use the globally unique iteration namespace:
                # main phase 0..iterations-1, separation phase
                # iterations..iterations+4999.
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
                        group.resume_hard_separation()
                        if group is not None else backend.resume_hard_separation()
                    )
                    logger.info(
                        "hard separation already completed in resumed checkpoint "
                        "(completed local steps=%d, marginalized=%d kept=%d); "
                        "skipping optimizer steps",
                        separation_start,
                        separation["marginalized"], separation["kept"],
                    )
                elif separation_start == 0:
                    separation = (
                        group.begin_hard_separation()
                        if group is not None else backend.begin_hard_separation()
                    )
                    logger.info(
                        "starting native hard separation: marginalized=%d kept=%d",
                        separation["marginalized"], separation["kept"],
                    )
                else:
                    # Topology was already split before this checkpoint. The
                    # native resume entry restores phase-specific LR/regularizer
                    # config without applying the destructive split twice.
                    separation = (
                        group.resume_hard_separation()
                        if group is not None else backend.resume_hard_separation()
                    )
                    logger.info(
                        "resuming native hard separation at local step %d/%d "
                        "(marginalized=%d kept=%d)",
                        separation_start, separate_steps,
                        separation["marginalized"], separation["kept"],
                    )
                separation_loop_start = (
                    separate_steps
                    if separation_already_complete else separation_start
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
                        microsteps=distributed.microsteps,
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
                    lpips_extra: dict[str, float] | None = None
                    sync_timing: dict[str, float] = {}
                    pair_log_fields: dict[str, object] = {}
                    if group is not None:
                        sync_result = group.train_step(
                            rank_views, global_step, phase="hard_separation",
                            collect_loss=should_log,
                            next_view_indices=next_rank_views,
                        )
                        sync_timing = _sync_train_timing(sync_result)
                        pair_timing, pair_log_fields = (
                            _pair_exchange_log_fields(sync_result)
                        )
                        sync_timing.update(pair_timing)
                        if should_log:
                            loss = sync_result["loss"]
                            if not isinstance(loss, dict):
                                raise RuntimeError(
                                    "synchronous hard-separation log step did "
                                    "not return loss scalars"
                                )
                            scalars = dict(loss)
                    elif self._lpips is not None:
                        reference = (
                            backend.cached_view_image(
                                is_train=True, index=view_index,
                            ).astype(np.float32) / 255.0
                        )
                        collect_grads = (
                            should_log and _log_geometry_grads_enabled()
                        )
                        if collect_grads:
                            backend.set_collect_geometry_grad_stats(True)
                        try:
                            lpips_extra = backend.train_view(
                                view_index, local_step,
                                collect_loss=should_log, lpips=self._lpips,
                                reference_rgb=reference,
                            )
                            if should_log:
                                scalars = backend.last_loss()
                                if collect_grads:
                                    scalars.update(
                                        backend.geometry_gradient_stats())
                        finally:
                            if collect_grads:
                                backend.set_collect_geometry_grad_stats(False)
                    else:
                        collect_grads = (
                            should_log and _log_geometry_grads_enabled()
                        )
                        if collect_grads:
                            backend.set_collect_geometry_grad_stats(True)
                        try:
                            backend.train_view(
                                view_index, global_step,
                                collect_loss=should_log)
                            if should_log:
                                scalars = backend.last_loss()
                                if collect_grads:
                                    scalars.update(
                                        backend.geometry_gradient_stats())
                        finally:
                            if collect_grads:
                                backend.set_collect_geometry_grad_stats(False)
                    training_clocks["optimizer_steps"] = (
                        int(training_clocks["optimizer_steps"]) + 1
                    )
                    training_clocks["microsteps"] = (
                        int(training_clocks["microsteps"])
                        + distributed.microsteps
                    )
                    training_clocks["views_seen"] = (
                        int(training_clocks["views_seen"])
                        + distributed.microsteps * distributed.world_size
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
                        if lpips_extra:
                            scalars.update(lpips_extra)
                            scalars["total_loss"] += float(
                                lpips_extra["lpips"]
                            )
                        scalars.update({
                            "step": int(global_step), "separation_step": int(local_step),
                            "view": int(view_index),
                            "views": [int(value) for value in rank_views],
                            "rank_view_audit": _rank_view_audit(
                                dataset, rank_views, distributed.microsteps,
                            ),
                            "world_size": distributed.world_size,
                            "global_batch_size": distributed.global_batch_size,
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
                    if 500 < local_step < 3000 and local_step % 100 == 0:
                        relocation = self.config.train.relocation
                        result = (
                            group.relocate(
                                local_step, relocation.mode,
                                relocation.score_mode == "gate_included",
                                self.config.train.seed,
                            )
                            if group is not None
                            else backend.relocate_native(
                                local_step, relocation.mode,
                                relocation.score_mode == "gate_included",
                                self.config.train.seed,
                            )
                        )
                        if result["all_dead"]:
                            logger.warning("hard separation stopped: every Gaussian is dead")
                            break
                    if eval_interval > 0 and local_step > 0 and local_step % eval_interval == 0:
                        self._evaluate(
                            backend, dataset, global_step, group=group,
                            clocks=dict(training_clocks),
                        )
                    interval = self.config.train.checkpoint_interval
                    if (
                        interval > 0 and local_step > 0
                        and local_step % interval == 0
                        and local_step != separate_steps - 1
                    ):
                        publish_runtime_checkpoint(
                            global_step,
                            phase="hard_separation",
                            phase_step=local_step,
                        )
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
                separated: NativeGaussians | None = None
                if separation_already_complete:
                    # The loaded immutable checkpoint already is the final
                    # resumable snapshot. Do not rewrite/collide with it.
                    # The upload-time checkpoint model was released before the
                    # training loop. Download the already-separated native
                    # state once, only when it is immediately needed for export.
                    if need_final_model:
                        separated = backend.model()
                else:
                    # Serialize the post-separation optimizer state before
                    # writing standalone artifacts.
                    separated_snapshot = publish_runtime_checkpoint(
                        last,
                        phase="hard_separation_complete",
                        phase_step=max(
                            0, last-self.config.train.iterations,
                        ),
                        filename="separated.npz",
                    )
                    separated = separated_snapshot.model
                    # Release multi-GB optimizer arrays immediately, retaining
                    # only the model host copy for imminent exports.
                    del separated_snapshot
                if separated is None:
                    separated = backend.model()
                save_model(self.output_path / "separated.npz", separated)
                if need_final_model:
                    final_model_snapshot = separated

            if self.config.train.hard_separation and main_terminated_all_dead:
                logger.warning(
                    "skipping hard separation because main training terminated "
                    "with an all-dead Gaussian topology"
                )

            final = final_model_snapshot
            if need_final_model and final is None:
                final = backend.model()
            if final is None:
                final = backend.model()
            save_model(self.output_path / "gaussians.npz", final)
            if dataset.eval_views:
                self._evaluate(
                    backend, dataset, last, group=group,
                    clocks=dict(training_clocks),
                )
            return final
        finally:
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
                    quality=_last_jsonl_record(
                        self.output_path / "metrics.jsonl"),
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
        # Lazy multi-camera loading already owns explicit decode workers.
        # Prevent each OpenCV call from spawning another internal pool and
        # oversubscribing the host while two Vulkan ranks are active.
        cv2.setNumThreads(1)
        self.output_path.mkdir(parents=True, exist_ok=True)
        save_composed_config(self.composed, self.output_path / "config.composed.yaml")
        dataset = self._preflight()
        save_resolved_config(self.config, self.output_path / "config.resolved.yaml")
        if self.config.pipeline.check_only:
            logger.info("configuration and native SelfCap dataset check passed")
            return
        completed = self.output_path / "gaussians.npz"
        if (
            completed.exists()
            and not self.config.run.overwrite_completed
            and "train" in self.config.pipeline.stages
        ):
            raise FileExistsError(f"refusing to overwrite {completed}; set run.overwrite_completed=true")
        model: NativeGaussians | None = None
        for stage in self.config.pipeline.stages:
            logger.info("running native pipeline stage: %s", stage)
            if stage == "init":
                init_path = self.output_path / "init.npz"
                if not init_path.exists():
                    save_model(init_path, initialize_explicit(self.config, dataset))
                model = load_model(init_path)
            elif stage == "train":
                # _train() loads init/checkpoint state itself.  Release the
                # stage-level init model before evaluating the call; otherwise
                # Python keeps a second 65-float/Gaussian host copy alive for
                # the entire native training run until this assignment occurs.
                model = None
                model = self._train(dataset)
            elif stage == "export":
                if model is None:
                    model_path = self.output_path / "gaussians.npz"
                    if not model_path.exists():
                        raise FileNotFoundError(
                            "export stage requires an in-memory train result or "
                            f"{model_path}; keep train and export in one run, or "
                            "run the train stage first"
                        )
                    model = load_model(model_path)
                export_4dgs(self.output_path / self.config.run.export_filename, model)
            else:
                raise ValueError(f"unsupported native pipeline stage: {stage}")
        save_resolved_config(self.config, self.output_path / "config.resolved.yaml")
        logger.info("FTGSPP native Vulkan pipeline completed: %s", self.output_path)
