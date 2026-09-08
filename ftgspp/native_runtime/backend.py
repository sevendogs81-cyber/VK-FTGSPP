"""Thin NumPy binding for the Vulkan-owned FTGSPP training engine."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Executor, Future, ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from ftgspp.native_runtime.dataset import NativeDataset, NativeView
from ftgspp.native_runtime.model import NativeGaussians

logger = logging.getLogger("ftgspp.native")


class NativeBackendUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _native_extension_identity(path_value: str) -> dict[str, object]:
    """Hash a loaded extension path once per process for V4 provenance."""

    if not path_value:
        return {"path": "unknown", "sha256": "unknown", "bytes": "unknown"}
    path = Path(path_value).expanduser()
    digest = hashlib.sha256()
    try:
        path = path.resolve()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "bytes": int(path.stat().st_size),
        }
    except OSError as exc:
        return {
            "path": str(path),
            "sha256": "unknown",
            "bytes": "unknown",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _shader_dir() -> Path:
    override = os.environ.get("FTGSPP_VK_SHADER_DIR")
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[2] / "native" / "shader"


def resolve_view_load_options(config) -> dict[str, object]:
    """Resolve Hydra run.* with optional env overrides for view residency."""
    mode = str(os.environ.get("FTGSPP_VIEW_LOAD_MODE", config.run.view_load_mode)).strip().lower()
    if mode not in {"lazy", "resident"}:
        raise ValueError(f"view_load_mode must be 'lazy' or 'resident', got {mode!r}")
    cache_raw = os.environ.get("FTGSPP_VIEW_RGB_CACHE")
    rgb_cache = int(cache_raw) if cache_raw is not None else int(config.run.view_rgb_cache)
    device_raw = os.environ.get("FTGSPP_VK_IMAGE_CACHE")
    image_device = (
        str(device_raw).strip().lower()
        if device_raw is not None
        else str(config.run.image_cache_device).strip().lower()
    )
    if image_device not in {"cpu", "gpu"}:
        raise ValueError(f"image_cache_device must be 'cpu' or 'gpu', got {image_device!r}")
    workers_raw = os.environ.get("FTGSPP_VIEW_LOAD_WORKERS")
    workers = int(workers_raw) if workers_raw is not None else int(config.run.view_load_workers)
    if rgb_cache <= 0:
        raise ValueError("view_rgb_cache must be > 0")
    if workers < 0:
        raise ValueError("view_load_workers must be >= 0")
    return {
        "mode": mode,
        "rgb_cache": rgb_cache,
        "cache_on_device": image_device == "gpu",
        "image_cache_device": image_device,
        "workers": workers,
    }


def _canonicalize_reference_rgba(rgb: np.ndarray) -> np.ndarray:
    """Normalize decoded reference images to contiguous HxWx4 uint8."""

    value = np.ascontiguousarray(rgb, dtype=np.uint8)
    if value.ndim != 3 or value.shape[-1] not in (3, 4):
        raise RuntimeError(
            f"decoded reference must be HxWx3 or HxWx4, got {value.shape}"
        )
    if value.shape[-1] == 3:
        rgba = np.empty(value.shape[:2] + (4,), dtype=np.uint8)
        rgba[..., :3] = value
        rgba[..., 3] = 255
        return np.ascontiguousarray(rgba)
    return value


class SharedRgbViewCache:
    """Thread-safe decoded-RGB(A) LRU shared by Vulkan replicas.

    A per-key future prevents two ranks selecting the same view from decoding
    it twice. :meth:`prefetch` submits requests in manifest order and inserts
    completed images in that same order, so cache residency is independent of
    JPEG worker completion timing.

    When ``pin=True`` (sync shared-resident), entries are never evicted and
    :meth:`prefill_arrays` may load the full dataset once for all ranks.
    """

    def __init__(self, capacity: int, *, pin: bool = False) -> None:
        self.capacity = max(1, int(capacity))
        self.pin = bool(pin)
        self._items: OrderedDict[tuple[bool, int], np.ndarray] = OrderedDict()
        self._pending: dict[tuple[bool, int], Future[np.ndarray]] = {}
        # Lookahead futures decode pixels off the critical path, but do not
        # enter/touch the deterministic LRU until their batch is consumed.
        self._scheduled: dict[tuple[bool, int], Future[np.ndarray]] = {}
        self._prefetched_reads: dict[tuple[bool, int], int] = {}
        self._prefilled = False
        self._lock = threading.RLock()

    @property
    def is_prefilled(self) -> bool:
        return self._prefilled

    def prefill_arrays(
        self,
        items: list[tuple[bool, int, np.ndarray]],
    ) -> bool:
        """Insert a full resident set once. Later calls are no-ops.

        Returns True if this call performed the insert.
        """

        with self._lock:
            if self._prefilled:
                return False
            for is_train, index, rgb in items:
                key = (bool(is_train), int(index))
                self._items[key] = _canonicalize_reference_rgba(rgb)
            if self.pin:
                self.capacity = max(self.capacity, len(self._items))
            else:
                while len(self._items) > self.capacity:
                    evicted, _value = self._items.popitem(last=False)
                    self._prefetched_reads.pop(evicted, None)
            self._prefilled = True
            return True

    def _store(self, key: tuple[bool, int], rgb: np.ndarray) -> np.ndarray:
        value = _canonicalize_reference_rgba(rgb)
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            if not self.pin:
                while len(self._items) > self.capacity:
                    evicted, _value = self._items.popitem(last=False)
                    self._prefetched_reads.pop(evicted, None)
            else:
                self.capacity = max(self.capacity, len(self._items))
        return value

    def get_or_load(self, *, is_train: bool, index: int, view: NativeView) -> np.ndarray:
        key = (bool(is_train), int(index))
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                remaining = self._prefetched_reads.get(key, 0)
                if remaining > 1:
                    self._prefetched_reads[key] = remaining - 1
                elif remaining == 1:
                    self._prefetched_reads.pop(key, None)
                else:
                    self._items.move_to_end(key)
                return cached
            pending = self._pending.get(key)
            producer = pending is None
            if pending is None:
                pending = Future()
                self._pending[key] = pending
        if not producer:
            return pending.result()
        try:
            rgb = self._store(key, view.load_rgb())
        except BaseException as exc:
            pending.set_exception(exc)
            with self._lock:
                self._pending.pop(key, None)
            raise
        pending.set_result(rgb)
        with self._lock:
            self._pending.pop(key, None)
        return rgb

    def prefetch(
        self,
        requests: list[tuple[bool, int, NativeView]],
        *,
        workers: int,
        executor: Executor | None = None,
    ) -> None:
        """Decode a deterministic ordered request set before rank dispatch."""

        ordered: list[tuple[tuple[bool, int], NativeView]] = []
        counts: dict[tuple[bool, int], int] = {}
        with self._lock:
            # Resolve LRU touches for the complete global batch before Vulkan
            # rank workers run. Their completion order must not change which
            # RGB is evicted on the following step.
            self._prefetched_reads.clear()
            for is_train, index, view in requests:
                key = (bool(is_train), int(index))
                counts[key] = counts.get(key, 0) + 1
                if counts[key] == 1:
                    ordered.append((key, view))
            for key, _view in ordered:
                if key in self._items:
                    self._items.move_to_end(key)
            missing = [
                (key, view) for key, view in ordered
                if key not in self._items
            ]

        scheduled: dict[tuple[bool, int], Future[np.ndarray]] = {}
        with self._lock:
            for key, _view in missing:
                future = self._scheduled.pop(key, None)
                if future is not None:
                    scheduled[key] = future

        decoded: list[np.ndarray] = []
        unscheduled = [
            (key, view) for key, view in missing if key not in scheduled
        ]
        submitted: dict[tuple[bool, int], Future[np.ndarray]] = {}
        owned_pool: ThreadPoolExecutor | None = None
        if workers > 1 and len(unscheduled) > 1:
            worker_limit = min(int(workers), len(missing))
            if executor is None:
                owned_pool = ThreadPoolExecutor(
                    max_workers=worker_limit,
                    thread_name_prefix="ftgspp-rgb-prefetch",
                )
                executor = owned_pool
            # Only submit unscheduled work. Results below are consumed in
            # manifest order regardless of completion order.
            submitted = {
                key: executor.submit(view.load_rgb)
                for key, view in unscheduled
            }
        try:
            for key, view in missing:
                future = scheduled.get(key) or submitted.get(key)
                decoded.append(future.result() if future is not None else view.load_rgb())
        finally:
            if owned_pool is not None:
                owned_pool.shutdown(wait=True, cancel_futures=True)
        for (key, _view), rgb in zip(missing, decoded):
            self._store(key, rgb)
        with self._lock:
            # Missing insertions may have shifted old entries. Reapply one
            # deterministic batch-level touch order, then suppress the worker
            # cache hits that immediately consume these prefetched views.
            for key, _view in ordered:
                if key in self._items:
                    self._items.move_to_end(key)
            self._prefetched_reads.update({
                key: count for key, count in counts.items()
                if key in self._items
            })

    def schedule(
        self,
        requests: list[tuple[bool, int, NativeView]],
        *,
        executor: Executor,
    ) -> None:
        """Start lookahead decodes without changing deterministic LRU state."""

        with self._lock:
            for is_train, index, view in requests:
                key = (bool(is_train), int(index))
                if (
                    key in self._items
                    or key in self._pending
                    or key in self._scheduled
                ):
                    continue
                self._scheduled[key] = executor.submit(view.load_rgb)


class NativeVulkanBackend:
    def __init__(self, device_id: int = 0, *, honor_device_env: bool = True):
        # A process-wide override is convenient for the single-device CLI but
        # would silently put every synchronous replica on the same GPU.  The
        # multi-device runner therefore passes honor_device_env=False.
        if honor_device_env:
            device_id = int(os.environ.get("FTGSPP_VK_DEVICE", device_id))
        try:
            import _ftgspp_vk as native_module
        except ImportError as exc:
            raise NativeBackendUnavailable(
                "_ftgspp_vk is not built; build the standalone native target on the Vulkan server"
            ) from exc
        FTGSPPVulkan = native_module.FTGSPPVulkan
        self._native_extension_metadata = _native_extension_identity(
            str(getattr(native_module, "__file__", ""))
        )
        shader_dir = _shader_dir()
        required = (
            "ftgspp_materialize.spv", "ftgspp_dynamic_projection.spv",
            "ftgspp_fused_backward_optimizer.spv",
            "ftgspp_fused_dynamic_backward_adam.spv",
            "ftgspp_tile_work_prepare.spv",
            "ftgspp_explicit_adam.spv",
            "ftgspp_peer_mean.spv",
            "ftgspp_sparse_flag.spv", "ftgspp_sparse_pack.spv",
            "ftgspp_sparse_prepare.spv",
            "ftgspp_sparse_scatter.spv", "ftgspp_regularizer_mean.spv",
            "ftgspp_relocation_prepare.spv", "ftgspp_relocation_copy.spv",
            "ftgspp_relocation_score.spv", "ftgspp_relocation_cdf.spv",
            "ftgspp_relocation_cdf_normalize.spv",
            "ftgspp_relocation_sample.spv",
            "ftgspp_flow_features.spv", "ftgspp_hard_separation.spv",
            "ftgspp_loss_reduce.spv",
            "ftgspp_color_correct_forward.spv", "ftgspp_color_correct_backward.spv",
            "ftgspp_color_adam.spv",
            "ssim_forward.spv", "ssim_backward.spv", "projection_forward.spv",
            "rasterize_forward.spv",
        )
        missing = [name for name in required if not (shader_dir / "generated" / name).is_file()]
        if missing:
            raise NativeBackendUnavailable(f"compiled shaders are missing below {shader_dir}: {missing}")
        self.native = FTGSPPVulkan()
        self.native.initialize(str(shader_dir) + os.sep, int(device_id))
        self.device_id = int(device_id)
        self.max_duration = 0.0
        self._dataset: NativeDataset | None = None
        self._view_load_mode = "lazy"
        self._shared_host_resident = False
        self._rgb_cache = SharedRgbViewCache(64)
        self._device_image_cache_capacity = 0
        self._bound_train_images: OrderedDict[int, None] = OrderedDict()
        self._view_load_workers = 0
        self._bound_train_index: int | None = None
        # Reused by test/reference packed-state diagnostics. Production
        # synchronization no longer registers a packed-host transport.
        self._packed_gradient_host: np.ndarray | None = None
        self._num_gaussians = 0

    def configure(self, config, scene_extent: float) -> None:
        lrs, schedules = config.train.lrs, config.train.lr_schedules
        self.max_duration = float(config.model.max_duration)
        self.native.configure_ftgspp({
            "max_steps": int(config.train.iterations),
            "max_sh_degree": int(config.init.sh_degree),
            "optimizer_batch_size": (
                int(config.distributed.global_batch_size)
                if config.distributed.world_size_lr_scaling
                == "effective_batch_adam"
                else 1
            ),
            "scene_scale": float(scene_extent),
            "ssim_lambda": 0.2, "means_lr": float(lrs["means"]),
            "means_lr_final": float(lrs["means"]) * float(schedules.get("means", 1.0)),
            "quats_lr": float(lrs["quats"]), "scales_lr": float(lrs["scales"]),
            "opacities_lr": float(lrs["opacities"]), "features_dc_lr": float(lrs["sh_0"]),
            "features_rest_lr": float(lrs["sh_n"]), "times_lr": float(lrs["times"]),
            "durations_lr": float(lrs["durations"]),
            "velocities_lr": float(lrs["velocity_model"]),
            "velocities_lr_final": float(lrs["velocity_model"]) * float(schedules.get("velocity_model", 1.0)),
            "gates_lr": (
                float(lrs.get("marginal_gates", 0.0))
                if config.model.marginal_gating else 0.0
            ),
            "scale_reg": 0.01, "opacity_reg": 0.01,
            "gate_reg": 0.001 if config.model.marginal_gating else 0.0,
            "max_duration": -1.0 if np.isinf(self.max_duration) else self.max_duration,
            "marginal_threshold": 0.05,
            "min_opacity": float(config.train.relocation.opacity_threshold),
            "seed": int(config.train.seed),
            "color_correction": bool(config.train.color_correction),
            "color_lr": float(config.train.lrs.get("color_correctors", 1e-3)),
            "color_reg": float(config.train.color_corrector_weight) if config.train.color_correction else 0.0,
            "color_num_cameras": 0,
            "color_correction_start": int(config.train.color_correction_start),
        })
        self._pending_color_cameras = bool(config.train.color_correction)
        self._color_lr = float(config.train.lrs.get("color_correctors", 1e-3))
        self._color_reg = (
            float(config.train.color_corrector_weight) if config.train.color_correction else 0.0
        )
        self._color_correction_start = int(config.train.color_correction_start)

    def upload_model(self, model: NativeGaussians) -> None:
        self.max_duration = model.max_duration
        self._num_gaussians = int(model.means.shape[0])
        self.native.set_dynamic_gaussians(
            model.means, model.log_scales, model.quaternions, model.opacity_logits,
            model.sh_coefficients, model.times, model.duration_raw, model.velocities,
            model.gate_raw,
        )
        release_upload = getattr(
            self.native, "release_model_upload_host_copies", None,
        )
        if callable(release_upload):
            release_upload()

    def _camera_index_map(self, dataset: NativeDataset) -> tuple[dict[int, int], list[int]]:
        camera_ids = sorted({int(c) for c in dataset.cameras})
        return {cid: i for i, cid in enumerate(camera_ids)}, camera_ids

    def _add_view_meta(
        self,
        view: NativeView,
        is_train: bool,
        *,
        camera_index: int,
    ) -> None:
        height, width = view.size
        self.native.add_dynamic_view_meta(
            int(height), int(width), view.w2c, view.intrinsic, float(view.timestamp),
            int(camera_index), int(view.frame), bool(is_train),
        )

    def _add_view_resident(
        self,
        view: NativeView,
        is_train: bool,
        *,
        camera_index: int,
        rgb: np.ndarray,
        cache_on_device: bool,
    ) -> None:
        self.native.add_dynamic_view(
            np.ascontiguousarray(rgb, dtype=np.uint8),
            view.w2c,
            view.intrinsic,
            float(view.timestamp),
            int(camera_index),
            int(view.frame),
            bool(is_train),
            bool(cache_on_device),
        )

    def _finish_color_cameras(self, camera_ids: list[int]) -> None:
        if getattr(self, "_pending_color_cameras", False):
            self.native.configure_ftgspp({
                "color_correction": True,
                "color_num_cameras": int(len(camera_ids)),
                "color_lr": float(getattr(self, "_color_lr", 1e-3)),
                "color_reg": float(getattr(self, "_color_reg", 1e-3)),
                "color_correction_start": int(getattr(self, "_color_correction_start", 0)),
            })
            self._pending_color_cameras = False
        self._color_camera_ids = camera_ids
        self._color_camera_to_index = {cid: i for i, cid in enumerate(camera_ids)}

    def _upload_dataset_lazy(
        self,
        dataset: NativeDataset,
        *,
        rgb_cache: int,
        camera_to_index: dict[int, int],
        camera_ids: list[int],
        shared_rgb_cache: SharedRgbViewCache | None,
    ) -> None:
        self._rgb_cache = (
            shared_rgb_cache
            if shared_rgb_cache is not None
            else SharedRgbViewCache(rgb_cache)
        )
        for view in dataset.train_views:
            self._add_view_meta(
                view, True, camera_index=camera_to_index[int(view.camera_id)])
        for view in dataset.eval_views:
            self._add_view_meta(
                view, False, camera_index=camera_to_index[int(view.camera_id)])
        logger.info(
            "registered view metadata: %d train + %d eval (lazy RGB(A), cache=%d)",
            len(dataset.train_views), len(dataset.eval_views), self._rgb_cache.capacity,
        )
        self._finish_color_cameras(camera_ids)

    def _decode_views(
        self,
        views: list[NativeView],
        *,
        workers: int,
        label: str,
    ) -> list[np.ndarray]:
        total = len(views)
        if total == 0:
            return []
        begin = time.perf_counter()
        if workers <= 0:
            out: list[np.ndarray] = []
            for index, view in enumerate(views):
                out.append(np.ascontiguousarray(view.load_rgb(), dtype=np.uint8))
                if index == 0 or (index + 1) % 200 == 0 or index + 1 == total:
                    logger.info(
                        "decoding %s views %d/%d (%.1fs)",
                        label, index + 1, total, time.perf_counter() - begin,
                    )
            return out

        def _load(view: NativeView) -> np.ndarray:
            return np.ascontiguousarray(view.load_rgb(), dtype=np.uint8)

        out: list[np.ndarray | None] = [None] * total
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_load, view): index for index, view in enumerate(views)}
            for future in as_completed(futures):
                index = futures[future]
                out[index] = future.result()
                done += 1
                if done == 1 or done % 200 == 0 or done == total:
                    logger.info(
                        "decoding %s views %d/%d (workers=%d, %.1fs)",
                        label, done, total, workers, time.perf_counter() - begin,
                    )
        if any(rgb is None for rgb in out):
            raise RuntimeError(f"failed to decode all {label} views")
        return out  # type: ignore[return-value]

    def _upload_dataset_resident(
        self,
        dataset: NativeDataset,
        *,
        cache_on_device: bool,
        workers: int,
        rgb_cache: int,
        camera_to_index: dict[int, int],
        camera_ids: list[int],
    ) -> None:
        # Metrics/LPIPS still need host references in resident mode.
        self._rgb_cache = SharedRgbViewCache(rgb_cache)
        train_rgbs = self._decode_views(
            dataset.train_views, workers=workers, label="train")
        eval_rgbs = self._decode_views(
            dataset.eval_views, workers=workers, label="eval")
        begin = time.perf_counter()
        for index, (view, rgb) in enumerate(zip(dataset.train_views, train_rgbs)):
            self._add_view_resident(
                view, True,
                camera_index=camera_to_index[int(view.camera_id)],
                rgb=rgb,
                cache_on_device=cache_on_device,
            )
            if index == 0 or (index + 1) % 500 == 0 or index + 1 == len(train_rgbs):
                logger.info(
                    "registered resident train views %d/%d (cache=%s, %.1fs)",
                    index + 1, len(train_rgbs),
                    "gpu" if cache_on_device else "cpu",
                    time.perf_counter() - begin,
                )
        for view, rgb in zip(dataset.eval_views, eval_rgbs):
            self._add_view_resident(
                view, False,
                camera_index=camera_to_index[int(view.camera_id)],
                rgb=rgb,
                cache_on_device=cache_on_device,
            )
        logger.info(
            "registered resident views: %d train + %d eval (image_cache=%s)",
            len(dataset.train_views), len(dataset.eval_views),
            "gpu" if cache_on_device else "cpu",
        )
        self._finish_color_cameras(camera_ids)

    def _upload_dataset_shared_resident(
        self,
        dataset: NativeDataset,
        *,
        workers: int,
        camera_to_index: dict[int, int],
        camera_ids: list[int],
        shared_rgb_cache: SharedRgbViewCache,
    ) -> None:
        """Sync multi-GPU resident: one host RGB pool, meta-only per rank."""

        self._shared_host_resident = True
        self._rgb_cache = shared_rgb_cache
        if not shared_rgb_cache.is_prefilled:
            train_rgbs = self._decode_views(
                dataset.train_views, workers=workers, label="train")
            eval_rgbs = self._decode_views(
                dataset.eval_views, workers=workers, label="eval")
            items = [
                (True, index, rgb)
                for index, rgb in enumerate(train_rgbs)
            ]
            items.extend(
                (False, index, rgb)
                for index, rgb in enumerate(eval_rgbs)
            )
            did_prefill = shared_rgb_cache.prefill_arrays(items)
            if did_prefill and train_rgbs:
                sample = train_rgbs[0]
                height, width = int(sample.shape[0]), int(sample.shape[1])
                channels = int(sample.shape[2]) if sample.ndim == 3 else 3
                total = len(train_rgbs) + len(eval_rgbs)
                host_gib = (
                    float(total) * float(height) * float(width) * float(channels)
                ) / (1024.0 ** 3)
                logger.info(
                    "shared_resident prefilled %d views once "
                    "(host_rgb≈%.2f GiB, shape=%dx%dx%d)",
                    total, host_gib, height, width, channels,
                )
            del train_rgbs, eval_rgbs, items
        for view in dataset.train_views:
            self._add_view_meta(
                view, True, camera_index=camera_to_index[int(view.camera_id)])
        for view in dataset.eval_views:
            self._add_view_meta(
                view, False, camera_index=camera_to_index[int(view.camera_id)])
        logger.info(
            "registered shared-resident view metadata: %d train + %d eval "
            "(pinned shared RGB cache=%d)",
            len(dataset.train_views), len(dataset.eval_views),
            self._rgb_cache.capacity,
        )
        self._finish_color_cameras(camera_ids)

    def upload_dataset(
        self,
        dataset: NativeDataset,
        *,
        view_load_mode: str = "resident",
        cache_on_device: bool = False,
        rgb_cache: int = 64,
        workers: int = 8,
        shared_rgb_cache: SharedRgbViewCache | None = None,
    ) -> None:
        mode = str(view_load_mode).strip().lower()
        if mode not in {"lazy", "resident"}:
            raise ValueError(f"view_load_mode must be 'lazy' or 'resident', got {mode!r}")
        self.native.clear_dynamic_views()
        self._dataset = dataset
        self._bound_train_index = None
        self._bound_train_images.clear()
        self._view_load_mode = mode
        self._shared_host_resident = False
        self._device_image_cache_capacity = (
            max(1, int(rgb_cache))
            if mode == "lazy" and bool(cache_on_device) else 0
        )
        if (
            self._device_image_cache_capacity > 0
            and not hasattr(self.native, "set_dynamic_view_cache_on_device")
        ):
            raise NativeBackendUnavailable(
                "lazy image_cache_device=gpu requires a rebuilt native module "
                "with set_dynamic_view_cache_on_device"
            )
        self._view_load_workers = int(workers)
        camera_to_index, camera_ids = self._camera_index_map(dataset)
        if mode == "lazy":
            self._upload_dataset_lazy(
                dataset,
                rgb_cache=rgb_cache,
                camera_to_index=camera_to_index,
                camera_ids=camera_ids,
                shared_rgb_cache=shared_rgb_cache,
            )
        elif shared_rgb_cache is not None:
            if cache_on_device:
                logger.warning(
                    "image_cache_device=gpu is ignored under sync shared "
                    "resident (one host RGB pool is bound per step)"
                )
            self._upload_dataset_shared_resident(
                dataset,
                workers=workers,
                camera_to_index=camera_to_index,
                camera_ids=camera_ids,
                shared_rgb_cache=shared_rgb_cache,
            )
        else:
            self._upload_dataset_resident(
                dataset,
                cache_on_device=cache_on_device,
                workers=workers,
                rgb_cache=rgb_cache,
                camera_to_index=camera_to_index,
                camera_ids=camera_ids,
            )
        if self.native.num_train != len(dataset.train_views) or self.native.num_val != len(dataset.eval_views):
            raise RuntimeError("native dataset registration count mismatch")

    def _binds_rgb_per_step(self) -> bool:
        return self._view_load_mode == "lazy" or self._shared_host_resident

    def _release_bound_train_image(self, view_index: int) -> None:
        self.native.set_dynamic_view_cache_on_device(
            int(view_index), True, False)
        self.native.release_dynamic_view_image(int(view_index), True)
        self._bound_train_images.pop(int(view_index), None)
        if self._bound_train_index == int(view_index):
            self._bound_train_index = None

    def view_load_workers(self) -> int:
        return int(self._view_load_workers)

    def view_cache_capacity(self) -> int:
        return int(self._rgb_cache.capacity)

    def _view_requests(
        self,
        *,
        is_train: bool,
        view_indices: list[int] | tuple[int, ...],
    ) -> list[tuple[bool, int, NativeView]]:
        if self._dataset is None:
            raise RuntimeError("upload_dataset must be called before prefetch")
        views = self._dataset.train_views if is_train else self._dataset.eval_views
        return [
            (bool(is_train), int(index), views[int(index)])
            for index in view_indices
        ]

    def _prefetch_views(
        self,
        *,
        is_train: bool,
        view_indices: list[int] | tuple[int, ...],
        executor: Executor | None = None,
    ) -> None:
        if self._shared_host_resident and self._rgb_cache.is_prefilled:
            return
        self._rgb_cache.prefetch(
            self._view_requests(is_train=is_train, view_indices=view_indices),
            workers=max(1, int(self._view_load_workers)),
            executor=executor,
        )

    def _schedule_views(
        self,
        *,
        is_train: bool,
        view_indices: list[int] | tuple[int, ...],
        executor: Executor,
    ) -> None:
        if self._view_load_workers <= 0:
            return
        if self._shared_host_resident and self._rgb_cache.is_prefilled:
            return
        self._rgb_cache.schedule(
            self._view_requests(is_train=is_train, view_indices=view_indices),
            executor=executor,
        )

    def prefetch_train_views(
        self,
        view_indices: list[int] | tuple[int, ...],
        *,
        executor: Executor | None = None,
    ) -> None:
        """Populate the shared lazy RGB(A) cache before Vulkan dispatch."""

        if not self._binds_rgb_per_step():
            return
        self._prefetch_views(
            is_train=True, view_indices=view_indices, executor=executor,
        )

    def schedule_train_views(
        self,
        view_indices: list[int] | tuple[int, ...],
        *,
        executor: Executor,
    ) -> None:
        """Decode future train RGB(A) views off the critical path."""

        if not self._binds_rgb_per_step():
            return
        self._schedule_views(
            is_train=True, view_indices=view_indices, executor=executor,
        )

    def prefetch_eval_views(
        self,
        view_indices: list[int] | tuple[int, ...],
        *,
        executor: Executor | None = None,
    ) -> None:
        """Populate decoded eval references before metric reads."""

        self._prefetch_views(
            is_train=False, view_indices=view_indices, executor=executor,
        )

    def schedule_eval_views(
        self,
        view_indices: list[int] | tuple[int, ...],
        *,
        executor: Executor,
    ) -> None:
        """Decode future eval references without touching LRU order."""

        self._schedule_views(
            is_train=False, view_indices=view_indices, executor=executor,
        )

    def cached_view_rgb(self, *, is_train: bool, index: int) -> np.ndarray:
        """Return contiguous RGB only for three-channel metrics/LPIPS.

        The underlying cache deliberately retains RGB(A), because native image
        upload and SSIM consume the dataset alpha channel when it exists.
        """

        if self._dataset is None:
            raise RuntimeError("upload_dataset must be called before loading RGB")
        views = self._dataset.train_views if is_train else self._dataset.eval_views
        rgb_or_rgba = self._rgb_cache.get_or_load(
            is_train=bool(is_train), index=int(index), view=views[int(index)],
        )
        if (
            rgb_or_rgba.ndim != 3
            or rgb_or_rgba.shape[-1] not in {3, 4}
        ):
            raise RuntimeError(
                "decoded reference must be HxWx3 RGB or HxWx4 RGBA, got "
                f"{rgb_or_rgba.shape}"
            )
        return np.ascontiguousarray(rgb_or_rgba[..., :3], dtype=np.uint8)

    def cached_view_image(self, *, is_train: bool, index: int) -> np.ndarray:
        """Return the cached RGB(A) image without discarding its alpha mask."""

        if self._dataset is None:
            raise RuntimeError("upload_dataset must be called before loading RGB(A)")
        views = self._dataset.train_views if is_train else self._dataset.eval_views
        image = self._rgb_cache.get_or_load(
            is_train=bool(is_train), index=int(index), view=views[int(index)],
        )
        if image.ndim != 3 or image.shape[-1] not in {3, 4}:
            raise RuntimeError(
                "decoded reference must be HxWx3 RGB or HxWx4 RGBA, got "
                f"{image.shape}"
            )
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _bind_train_rgb(self, view_index: int) -> np.ndarray | None:
        """Bind and return the full RGB(A) training image for native loss."""

        if not self._binds_rgb_per_step():
            return None
        if self._dataset is None:
            raise RuntimeError("upload_dataset must be called before train_view")
        view = self._dataset.train_views[int(view_index)]
        rgb = self._rgb_cache.get_or_load(is_train=True, index=int(view_index), view=view)
        if int(view_index) in self._bound_train_images:
            self._bound_train_images.move_to_end(int(view_index))
            self._bound_train_index = int(view_index)
            return rgb
        if (
            self._device_image_cache_capacity <= 0
            and self._bound_train_index is not None
            and self._bound_train_index != int(view_index)
        ):
            self._release_bound_train_image(int(self._bound_train_index))
        self.native.set_dynamic_view_cache_on_device(
            int(view_index), True, self._device_image_cache_capacity > 0)
        self.native.bind_dynamic_view_image(int(view_index), True, rgb)
        self._bound_train_images[int(view_index)] = None
        self._bound_train_images.move_to_end(int(view_index))
        self._bound_train_index = int(view_index)
        while (
            self._device_image_cache_capacity > 0
            and len(self._bound_train_images) > self._device_image_cache_capacity
        ):
            evicted = next(iter(self._bound_train_images))
            self._release_bound_train_image(int(evicted))
        return rgb

    def _maybe_log_pipeline_telemetry(self, step: int) -> None:
        path = os.environ.get("FTGSPP_PIPELINE_TELEMETRY")
        if not path:
            return
        try:
            row = self.native.get_pipeline_telemetry()
        except Exception:
            return
        row["step"] = float(step)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def train_view(
        self,
        view_index: int,
        step: int,
        *,
        collect_loss: bool = True,
        lpips=None,
        reference_rgb: np.ndarray | None = None,
    ) -> dict[str, float] | None:
        """Run one native step. Optional LPIPS adds grads after L1/SSIM on corrected RGB."""
        rgb = self._bind_train_rgb(int(view_index))
        if lpips is None:
            self.native.train_dynamic_view(
                int(view_index), int(step), bool(collect_loss))
            self._maybe_log_pipeline_telemetry(int(step))
            return None
        reference_alpha: np.ndarray | None = None
        if reference_rgb is None:
            if rgb is None:
                reference_image = self.cached_view_image(
                    is_train=True, index=int(view_index),
                )
            else:
                if rgb.ndim != 3 or rgb.shape[-1] not in {3, 4}:
                    raise RuntimeError(
                        "bound LPIPS reference must be RGB(A), got "
                        f"{rgb.shape}"
                    )
                reference_image = rgb
            reference_rgb = reference_image[..., :3].astype(np.float32) / 255.0
            if reference_image.shape[-1] == 4:
                reference_alpha = reference_image[..., 3].astype(np.float32) / 255.0
        else:
            supplied = np.asarray(reference_rgb)
            if supplied.ndim != 3 or supplied.shape[-1] not in {3, 4}:
                raise RuntimeError(
                    "LPIPS reference must be HxWx3 RGB or HxWx4 RGBA, got "
                    f"{supplied.shape}"
                )
            if supplied.shape[-1] == 4:
                reference_alpha = supplied[..., 3].astype(np.float32)
                reference_rgb = supplied[..., :3]
        self.native.train_dynamic_view_to_loss(
            int(view_index), int(step), bool(collect_loss))
        pred = np.array(self.native.pixel_state, dtype=np.float32, copy=True)[..., :3]
        result = lpips.compute(pred, reference_rgb, mask=reference_alpha)
        lpips_loss = float(result.loss)
        if result.pixel_grad is None:
            raise RuntimeError(
                "LPIPS training requested a scalar-only result without a "
                "pixel gradient"
            )
        pixel_grad = np.asarray(result.pixel_grad, dtype=np.float32)
        expected_gradient_shape = (*pred.shape[:2], 4)
        if not np.isfinite(lpips_loss):
            raise FloatingPointError("LPIPS returned a non-finite weighted loss")
        if (
            pixel_grad.shape != expected_gradient_shape
            or not np.isfinite(pixel_grad).all()
        ):
            raise FloatingPointError(
                "LPIPS returned an invalid finite HxWx4 pixel gradient: "
                f"{pixel_grad.shape}, expected {expected_gradient_shape}"
            )
        self.native.accumulate_pixel_state_grad(
            np.ascontiguousarray(pixel_grad, dtype=np.float32))
        self.native.train_dynamic_view_from_loss(int(view_index), int(step))
        return {"lpips": lpips_loss}

    def backward_view(
        self, view_index: int, step: int, *, collect_loss: bool = True,
    ) -> None:
        """Compute one view's raw FP32 gradients without updating Adam.

        This split entry point is the synchronization boundary used by
        replicated multi-device training. Native code preserves every raw
        gradient in a device-local FP32 accumulator until
        :meth:`finalize_local_gradient` applies exactly ``1/K``.
        """
        self._bind_train_rgb(int(view_index))
        method = getattr(self.native, "backward_dynamic_view", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks backward_dynamic_view; rebuild the "
                "multi-device Vulkan target"
            )
        method(int(view_index), int(step), bool(collect_loss))

    def finalize_local_gradient(self, microsteps: int) -> None:
        """Finalize the rank-local raw sum as a mean before collective."""

        method = getattr(self.native, "finalize_local_gradient", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks finalize_local_gradient; rebuild the "
                "K-view Vulkan target"
            )
        method(int(microsteps))

    def apply_sync_tile_capacity(self, exact_count: int) -> None:
        method = getattr(self.native, "apply_sync_tile_capacity", None)
        if callable(method):
            method(int(exact_count))

    def clear_sync_collectives_for_replay(self) -> None:
        method = getattr(self.native, "clear_sync_collectives_for_replay", None)
        if callable(method):
            method()

    def drain_for_structure_change(self) -> None:
        method = getattr(self.native, "drain_for_structure_change", None)
        if callable(method):
            method()

    def inject_sync_tile_overflow(self, exact_count: int) -> None:
        method = getattr(self.native, "inject_sync_tile_overflow", None)
        if not callable(method):
            raise NativeBackendUnavailable(
                "the native module lacks inject_sync_tile_overflow; rebuild"
            )
        method(int(exact_count))

    def packed_gradient_state(self) -> np.ndarray:
        """Download one native-layout FP32 gradient arena into reusable host RAM."""

        size_method = getattr(self.native, "packed_gradient_size", None)
        download = getattr(self.native, "gradient_state_packed_into", None)
        if size_method is None or download is None:
            raise NativeBackendUnavailable(
                "the native module lacks packed gradient readback; rebuild the "
                "native target"
            )
        size = int(size_method())
        if size <= 0:
            raise RuntimeError("native packed gradient payload is empty")
        if self._packed_gradient_host is None or self._packed_gradient_host.size != size:
            self._packed_gradient_host = np.empty(size, dtype=np.float32)
        download(self._packed_gradient_host)
        timing = getattr(self.native, "last_packed_download_timing", None)
        if callable(timing):
            self._last_packed_download_timing = dict(timing())
        return self._packed_gradient_host

    def last_packed_download_timing(self) -> dict[str, object]:
        return dict(getattr(self, "_last_packed_download_timing", {}))

    def last_packed_upload_timing(self) -> dict[str, object]:
        return dict(getattr(self, "_last_packed_upload_timing", {}))

    def supports_packed_gradient_transport(self) -> bool:
        """Test/reference helper only; not a production training backend."""

        return all(
            callable(getattr(self.native, name, None))
            for name in (
                "packed_gradient_size",
                "packed_gradient_layout",
                "gradient_state_packed_into",
                "set_gradient_state_packed",
                "mean_packed_gradients_into",
            )
        )

    def supports_pair_exchange_gpu_merge(self) -> bool:
        """Probe the temporary native methods backing hierarchical P=2."""

        return all(
            callable(getattr(self.native, name, None))
            for name in (
                "finalize_local_gradient",
                "create_pair_exchange_gpu_merge_owner",
                "configure_pair_exchange_gpu_merge",
                "exchange_pair_gradient",
                "exchange_pair_relocation",
                "optimizer_step_pair_exchange_gpu_merge",
                "reset_pair_exchange_gpu_merge_attachments",
                "release_pair_exchange_gpu_merge_owner",
            )
        )

    def create_pair_exchange_gpu_merge_owner(
        self, *, world_size: int, bucket_bytes: int, sparse: bool,
    ) -> dict[str, object]:
        method = getattr(type(self.native), "create_pair_exchange_gpu_merge_owner", None)
        if method is None:
            method = getattr(self.native, "create_pair_exchange_gpu_merge_owner", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks the pair-exchange owner capability")
        return dict(method(int(world_size), int(bucket_bytes), bool(sparse)))

    def configure_pair_exchange_gpu_merge(
        self, *, sparse: bool, codec_policy: str = "auto_exact",
    ) -> None:
        method = getattr(self.native, "configure_pair_exchange_gpu_merge", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks pair-exchange mode configuration")
        method(str(codec_policy) if sparse else "dense")

    def exchange_pair_gradient(
        self, *, rank: int, generation: int,
    ) -> dict[str, object]:
        method = getattr(self.native, "exchange_pair_gradient", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks pair gradient exchange")
        return dict(method(int(rank), int(generation)))

    def optimizer_step_pair_exchange_gpu_merge(
        self, step: int, timestamps: Sequence[float],
    ) -> dict[str, float]:
        method = getattr(self.native, "optimizer_step_pair_exchange_gpu_merge", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks GPU peer-mean optimizer step")
        return {
            str(key): float(value)
            for key, value in method(
                int(step), [float(value) for value in timestamps],
            ).items()
        }

    def exchange_pair_relocation(
        self, *, rank: int, generation: int,
    ) -> dict[str, object]:
        method = getattr(self.native, "exchange_pair_relocation", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks pair relocation exchange")
        return dict(method(int(rank), int(generation)))

    def train_view_timestamp(self, view_index: int) -> float:
        if self._dataset is None:
            raise RuntimeError("upload_dataset must precede distributed training")
        return float(self._dataset.train_views[int(view_index)].timestamp)

    def release_pair_exchange_gpu_merge_owner(self) -> None:
        method = getattr(type(self.native), "release_pair_exchange_gpu_merge_owner", None)
        if method is None:
            method = getattr(self.native, "release_pair_exchange_gpu_merge_owner", None)
        if callable(method):
            method()

    def reset_pair_exchange_gpu_merge_attachments(self) -> None:
        method = getattr(self.native, "reset_pair_exchange_gpu_merge_attachments", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks mailbox attachment reset")
        method()

    def packed_gradient_layout(self) -> dict[str, object]:
        method = getattr(self.native, "packed_gradient_layout", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks the packed gradient ABI layout"
            )
        return {str(key): value for key, value in method().items()}

    @staticmethod
    def _unpack_reordered_sh(
        values: np.ndarray, num_gaussians: int, reorder_size: int,
    ) -> np.ndarray:
        packed = np.asarray(values, dtype=np.float32).reshape(-1, 4)
        canonical = np.empty((num_gaussians, 48), dtype=np.float32)
        for index in range(num_gaussians):
            lane = index % reorder_size
            block = index - lane
            for coefficient4 in range(12):
                packed_index = (
                    12 * block + coefficient4 * reorder_size + lane
                )
                canonical[index, 4 * coefficient4:4 * coefficient4 + 4] = (
                    packed[packed_index]
                )
        return canonical.reshape(num_gaussians, 16, 3)

    @staticmethod
    def _pack_reordered_sh(
        values: np.ndarray, padded_gaussians: int, reorder_size: int,
    ) -> np.ndarray:
        canonical = np.asarray(values, dtype=np.float32)
        if canonical.ndim != 3 or canonical.shape[1:] != (16, 3):
            raise ValueError(
                "canonical SH gradient must have shape (N, 16, 3), got "
                f"{canonical.shape}"
            )
        packed = np.zeros((12 * padded_gaussians, 4), dtype=np.float32)
        flat = canonical.reshape(canonical.shape[0], 48)
        for index in range(canonical.shape[0]):
            lane = index % reorder_size
            block = index - lane
            for coefficient4 in range(12):
                packed_index = (
                    12 * block + coefficient4 * reorder_size + lane
                )
                packed[packed_index] = flat[
                    index, 4 * coefficient4:4 * coefficient4 + 4
                ]
        return packed.reshape(-1)

    @staticmethod
    def _packed_region(
        arena: np.ndarray, layout: Mapping[str, object], name: str,
    ) -> np.ndarray:
        region = layout.get(name)
        if not isinstance(region, (tuple, list)) or len(region) != 2:
            raise RuntimeError(f"packed gradient layout is missing {name}")
        offset, count = (int(value) for value in region)
        return np.asarray(arena[offset:offset + count], dtype=np.float32)

    def canonical_gradient_state(self) -> dict[str, np.ndarray]:
        """Return raw native gradients in framework-neutral parameter shapes."""

        layout = self.packed_gradient_layout()
        arena = np.array(self.packed_gradient_state(), copy=True)
        xyz = self._packed_region(
            arena, layout, "grad_base_xyz_ws",
        )
        if xyz.size % 3:
            raise RuntimeError("packed mean gradient size is not divisible by 3")
        num_gaussians = xyz.size // 3
        scale_opacity = self._packed_region(
            arena, layout, "grad_base_log_scales_opacity",
        ).reshape(num_gaussians, 4)
        time_duration = self._packed_region(
            arena, layout, "grad_times_durations",
        ).reshape(num_gaussians, 2)
        velocity_gate_region = layout.get("grad_velocities_gates")
        if isinstance(velocity_gate_region, (tuple, list)):
            velocity_gate = self._packed_region(
                arena, layout, "grad_velocities_gates",
            ).reshape(num_gaussians, 4)
        else:
            velocity_gate = np.zeros((num_gaussians, 4), dtype=np.float32)
        sh_native = self._packed_region(
            arena, layout, "grad_sh_coeffs_reordered",
        )
        if sh_native.size % 48:
            raise RuntimeError("packed SH gradient size is not divisible by 48")
        padded_gaussians = sh_native.size // 48
        reorder_size = int(layout.get("sh_reorder_size", 16))
        rotations = self._packed_region(
            arena, layout, "grad_rotations",
        ).reshape(num_gaussians, 4)
        return {
            "means": xyz.reshape(num_gaussians, 3).copy(),
            "log_scales": scale_opacity[:, :3].copy(),
            "opacity_logits": scale_opacity[:, 3:4].copy(),
            "times": time_duration[:, :1].copy(),
            "duration_raw": time_duration[:, 1:2].copy(),
            "velocities": velocity_gate[:, :3].copy(),
            "gate_raw": velocity_gate[:, 3:4].copy(),
            "sh_coefficients": self._unpack_reordered_sh(
                sh_native, num_gaussians, reorder_size,
            ),
            "quaternions": rotations.copy(),
            "_packed_padded_gaussians": np.asarray(
                [padded_gaussians], dtype=np.int64,
            ),
        }

    def canonical_gradient_to_packed(
        self,
        values: Mapping[str, np.ndarray],
        *,
        base: np.ndarray | None = None,
    ) -> np.ndarray:
        """Pack canonical Gaussian gradients for a controlled Adam reference."""

        layout = self.packed_gradient_layout()
        total = int(layout["total_floats"])
        arena = (
            np.zeros(total, dtype=np.float32)
            if base is None
            else np.array(base, dtype=np.float32, copy=True)
        )
        if arena.shape != (total,):
            raise ValueError(
                f"packed gradient base must have shape {(total,)}, got {arena.shape}"
            )
        means = np.asarray(values["means"], dtype=np.float32)
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError(
                f"canonical mean gradient must have shape (N, 3), got {means.shape}"
            )
        num_gaussians = means.shape[0]

        def write(name: str, value: np.ndarray) -> None:
            region = layout.get(name)
            if not isinstance(region, (tuple, list)) or len(region) != 2:
                raise RuntimeError(f"packed gradient layout is missing {name}")
            offset, count = (int(item) for item in region)
            flat = np.asarray(value, dtype=np.float32).reshape(-1)
            if flat.size != count:
                raise ValueError(
                    f"{name} expects {count} floats, got {flat.size}"
                )
            arena[offset:offset + count] = flat

        write("grad_base_xyz_ws", means)
        write("grad_base_log_scales_opacity", np.concatenate((
            np.asarray(values["log_scales"], dtype=np.float32),
            np.asarray(values["opacity_logits"], dtype=np.float32),
        ), axis=1))
        write("grad_times_durations", np.concatenate((
            np.asarray(values["times"], dtype=np.float32),
            np.asarray(values["duration_raw"], dtype=np.float32),
        ), axis=1))
        if "grad_velocities_gates" in layout:
            write("grad_velocities_gates", np.concatenate((
                np.asarray(values["velocities"], dtype=np.float32),
                np.asarray(values["gate_raw"], dtype=np.float32),
            ), axis=1))
        sh_region = layout["grad_sh_coeffs_reordered"]
        sh_count = int(sh_region[1])
        padded_gaussians = sh_count // 48
        write("grad_sh_coeffs_reordered", self._pack_reordered_sh(
            np.asarray(values["sh_coefficients"], dtype=np.float32),
            padded_gaussians,
            int(layout.get("sh_reorder_size", 16)),
        ))
        write(
            "grad_rotations",
            np.asarray(values["quaternions"], dtype=np.float32),
        )
        return arena

    def set_canonical_gradient_state(
        self, values: Mapping[str, np.ndarray],
    ) -> None:
        """Upload one canonical gradient set for a controlled Adam step."""

        self.set_packed_gradient_state(
            self.canonical_gradient_to_packed(values),
        )

    def canonical_optimizer_state(self) -> dict[str, np.ndarray]:
        """Return Gaussian Adam moments in canonical parameter shapes."""

        raw = self.optimizer_state()
        scales_opacity = np.asarray(
            raw["g_scales_opacs"], dtype=np.float32,
        )
        if scales_opacity.size % 8:
            raise RuntimeError("scale/opacity Adam state size is not divisible by 8")
        num_gaussians = (
            scales_opacity.size // 8
            if scales_opacity.size
            else self._num_gaussians
        )

        def paired(name: str, width: int) -> tuple[np.ndarray, np.ndarray]:
            flat = np.asarray(raw[name], dtype=np.float32)
            if flat.size == 0:
                zeros = np.zeros((num_gaussians, width), dtype=np.float32)
                return zeros.copy(), zeros
            values = flat.reshape(num_gaussians, 2, width)
            return values[:, 0].copy(), values[:, 1].copy()

        means_m, means_v = paired("g_xyz_ws", 3)
        scales_opacity_m, scales_opacity_v = paired("g_scales_opacs", 4)
        time_duration_m, time_duration_v = paired("g_times_durations", 2)
        velocity_gate_m, velocity_gate_v = paired(
            "g_velocities_gates", 4,
        )
        quaternion_m, quaternion_v = paired("g_rotations", 4)
        sh_m_native = np.asarray(raw["g_sh_coeffs_1"], dtype=np.float32)
        sh_v_native = np.asarray(raw["g_sh_coeffs_2"], dtype=np.float32)
        reorder_size = 16
        layout_method = getattr(self.native, "packed_gradient_layout", None)
        if callable(layout_method):
            try:
                reorder_size = int(layout_method().get(
                    "sh_reorder_size", reorder_size,
                ))
            except Exception:
                pass
        if sh_m_native.size == 0 and sh_v_native.size == 0:
            sh_m = np.zeros((num_gaussians, 16, 3), dtype=np.float32)
            sh_v = np.zeros_like(sh_m)
        else:
            sh_m = self._unpack_reordered_sh(
                sh_m_native, num_gaussians, reorder_size,
            )
            sh_v = self._unpack_reordered_sh(
                sh_v_native, num_gaussians, reorder_size,
            )
        return {
            "means_m": means_m,
            "means_v": means_v,
            "log_scales_m": scales_opacity_m[:, :3].copy(),
            "log_scales_v": scales_opacity_v[:, :3].copy(),
            "opacity_logits_m": scales_opacity_m[:, 3:4].copy(),
            "opacity_logits_v": scales_opacity_v[:, 3:4].copy(),
            "times_m": time_duration_m[:, :1].copy(),
            "times_v": time_duration_v[:, :1].copy(),
            "duration_raw_m": time_duration_m[:, 1:2].copy(),
            "duration_raw_v": time_duration_v[:, 1:2].copy(),
            "velocities_m": velocity_gate_m[:, :3].copy(),
            "velocities_v": velocity_gate_v[:, :3].copy(),
            "gate_raw_m": velocity_gate_m[:, 3:4].copy(),
            "gate_raw_v": velocity_gate_v[:, 3:4].copy(),
            "sh_coefficients_m": sh_m,
            "sh_coefficients_v": sh_v,
            "quaternions_m": quaternion_m,
            "quaternions_v": quaternion_v,
        }

    def restore_canonical_optimizer_state(
        self, values: Mapping[str, np.ndarray],
    ) -> None:
        """Upload Gaussian Adam moments from canonical parameter shapes."""

        raw = self.optimizer_state()
        num_gaussians = self._num_gaussians

        def paired(
            name: str,
            first: np.ndarray,
            second: np.ndarray,
            width: int,
        ) -> None:
            left = np.asarray(first, dtype=np.float32).reshape(
                num_gaussians, width,
            )
            right = np.asarray(second, dtype=np.float32).reshape(
                num_gaussians, width,
            )
            raw[name] = np.stack((left, right), axis=1).reshape(-1)

        paired("g_xyz_ws", values["means_m"], values["means_v"], 3)
        paired(
            "g_scales_opacs",
            np.concatenate(
                (values["log_scales_m"], values["opacity_logits_m"]), axis=1,
            ),
            np.concatenate(
                (values["log_scales_v"], values["opacity_logits_v"]), axis=1,
            ),
            4,
        )
        paired(
            "g_times_durations",
            np.concatenate(
                (values["times_m"], values["duration_raw_m"]), axis=1,
            ),
            np.concatenate(
                (values["times_v"], values["duration_raw_v"]), axis=1,
            ),
            2,
        )
        paired(
            "g_velocities_gates",
            np.concatenate(
                (values["velocities_m"], values["gate_raw_m"]), axis=1,
            ),
            np.concatenate(
                (values["velocities_v"], values["gate_raw_v"]), axis=1,
            ),
            4,
        )
        paired(
            "g_rotations",
            values["quaternions_m"],
            values["quaternions_v"],
            4,
        )
        reorder_size = int(
            self.packed_gradient_layout().get("sh_reorder_size", 16)
        )
        for suffix in ("m", "v"):
            raw_name = "g_sh_coeffs_1" if suffix == "m" else "g_sh_coeffs_2"
            padded_gaussians = np.asarray(raw[raw_name]).size // 48
            raw[raw_name] = self._pack_reordered_sh(
                values[f"sh_coefficients_{suffix}"],
                padded_gaussians,
                reorder_size,
            )
        self.restore_optimizer(raw)

    def reduce_packed_gradients_into(
        self, per_rank: list[np.ndarray], output: np.ndarray,
    ) -> None:
        """Run the native fixed-rank-order FP32 mean and finite check.

        ``output`` may be exactly ``per_rank[0]``.  The native implementation
        consumes all rank values for an element before replacing that element,
        which lets the coordinator reuse rank 0's download arena in place.
        """

        method = getattr(self.native, "mean_packed_gradients_into", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks the packed C++ FP32 reducer"
            )
        method(per_rank, output)

    def set_packed_gradient_state(self, values: np.ndarray) -> None:
        """Upload one reduced native-layout arena with one transfer batch."""

        method = getattr(self.native, "set_gradient_state_packed", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks packed gradient upload"
            )
        # The native reducer already performed the required finite scan.
        method(np.ascontiguousarray(values, dtype=np.float32), False)
        timing = getattr(self.native, "last_packed_upload_timing", None)
        if callable(timing):
            self._last_packed_upload_timing = dict(timing())

    def device_metadata(self) -> dict[str, object]:
        """Return JSON-safe native Vulkan identity/capability provenance."""

        info_method = getattr(self.native, "device_info", None)
        info = (
            {str(key): value for key, value in info_method().items()}
            if callable(info_method) else {}
        )
        unknown = "unknown"
        uuid = info.get("device_uuid", unknown)
        return {
            "device_id": int(self.device_id),
            "native_extension": dict(self._native_extension_metadata),
            "name": info.get("name", unknown),
            "vendor_id": info.get("vendor_id", info.get("vendor", unknown)),
            "device_id_pci": info.get("device_id_pci", unknown),
            "driver_version": info.get("driver_version", unknown),
            "api_version": info.get("api_version", unknown),
            "api_version_raw": info.get("api_version_raw", unknown),
            "uuid": uuid,
            "device_uuid": uuid,
            "pci_bdf": info.get("pci_bdf", unknown),
            "subgroup_size": info.get("subgroup_size", unknown),
            "features": {
                "int16": info.get("has_int16", unknown),
                "int64": info.get("has_int64", unknown),
                "float32_atomic_add": info.get(
                    "has_float32_atomic_add", unknown,
                ),
            },
        }

    def reported_native_abi(self) -> str:
        method = getattr(self.native, "native_abi", None)
        return str(method()) if callable(method) else "unknown"

    def optimizer_step(self, step: int) -> None:
        """Apply Adam exactly once from the previously uploaded gradients."""
        method = getattr(self.native, "optimizer_step", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks optimizer_step; rebuild the "
                "multi-device Vulkan target"
            )
        method(int(step))

    def set_collect_geometry_grad_stats(self, enabled: bool) -> None:
        """Gate optional xyz/scale/opacity/rotation/SH gradient scalars in Adam."""
        method = getattr(self.native, "set_collect_geometry_grad_stats", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks set_collect_geometry_grad_stats; "
                "rebuild the Vulkan target"
            )
        method(bool(enabled))

    def geometry_gradient_stats(self) -> dict[str, float]:
        """Return RMS/L2 scalars collected by the previous log-step Adam."""
        method = getattr(self.native, "get_geometry_gradient_stats", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks get_geometry_gradient_stats; "
                "rebuild the Vulkan target"
            )
        result = {
            str(key): float(value) for key, value in method().items()
        }
        required = {
            "grad_xyz_rms", "grad_scaling_rms", "grad_oc_rms",
            "grad_rotation_rms", "grad_sh_dc_rms", "grad_sh_rest_rms",
        }
        missing = sorted(required - result.keys())
        if missing:
            raise NativeBackendUnavailable(
                "geometry gradient stats are incomplete: "
                f"{missing}; rebuild the Vulkan target"
            )
        non_finite = sorted(
            key for key in required if not np.isfinite(result[key])
        )
        if non_finite:
            raise FloatingPointError(
                f"geometry gradient stats are non-finite: {non_finite}"
            )
        return result

    def last_loss(self) -> dict[str, float]:
        result = {
            str(key): float(value)
            for key, value in self.native.get_last_loss().items()
        }
        required = {
            "l1", "ssim", "reg_opacity", "reg_scale", "gate", "color_cc",
            "image_loss", "total_loss",
        }
        missing = sorted(required - result.keys())
        if missing:
            raise NativeBackendUnavailable(
                "the native module lacks complete loss scalar fields: "
                f"{missing}; rebuild the Vulkan target"
            )
        non_finite = sorted(
            key for key in required if not np.isfinite(result[key])
        )
        if non_finite:
            raise FloatingPointError(
                f"native loss scalar fields are non-finite: {non_finite}"
            )
        # Keep single-device and distributed logs on the same exact derived
        # identity. Primitive native terms are authoritative; independently
        # accumulated derived slots can differ by reduction-order rounding.
        result["image_loss"] = result["l1"] + result["ssim"]
        result["total_loss"] = sum(
            result[key] for key in (
                "l1", "ssim", "reg_opacity", "reg_scale", "gate",
                "color_cc",
            )
        )
        return result

    def render_train(self, view_index: int) -> np.ndarray:
        self._bind_train_rgb(int(view_index))
        self.native.render_dynamic_train_view(int(view_index))
        copy_rgb = getattr(self.native, "copy_pixel_rgb", None)
        if callable(copy_rgb):
            return np.asarray(copy_rgb(), dtype=np.float32)
        return np.array(self.native.pixel_state, dtype=np.float32, copy=True)[..., :3]

    def render_train_raw(self, view_index: int) -> np.ndarray:
        """Forward a training view without the per-camera affine corrector."""

        self._bind_train_rgb(int(view_index))
        self.native.render_dynamic_train_view(
            int(view_index), apply_color_correction=False,
        )
        copy_rgb = getattr(self.native, "copy_pixel_rgb", None)
        if callable(copy_rgb):
            return np.asarray(copy_rgb(), dtype=np.float32)
        return np.array(self.native.pixel_state, dtype=np.float32, copy=True)[..., :3]

    def set_sh_coefficients(self, sh_coeffs: np.ndarray) -> None:
        """Upload canonical (N, 16, 3) SH; native reorders before the device copy."""

        method = getattr(self.native, "set_sh_coefficients", None)
        if not callable(method):
            raise NativeBackendUnavailable(
                "the native module lacks set_sh_coefficients; rebuild "
                "with python -m ftgspp.build_native"
            )
        coeffs = np.ascontiguousarray(sh_coeffs, dtype=np.float32)
        if coeffs.ndim != 3 or coeffs.shape[1:] != (16, 3):
            raise ValueError(
                f"sh_coeffs must have shape (N, 16, 3), got {coeffs.shape}"
            )
        method(coeffs)

    def set_color_params(self, params: np.ndarray) -> None:
        """Upload per-camera affine color-corrector parameters only."""

        method = getattr(self.native, "set_color_correction_params", None)
        if not callable(method):
            raise NativeBackendUnavailable(
                "the native module lacks set_color_correction_params; rebuild "
                "with python -m ftgspp.build_native"
            )
        values = np.ascontiguousarray(params, dtype=np.float32).reshape(-1)
        method(values)

    def render_eval(self, view_index: int) -> np.ndarray:
        begin = getattr(self.native, "begin_eval_val_view", None)
        submit = getattr(self.native, "submit_eval_rgb_readback", None)
        complete = getattr(self.native, "complete_eval_rgb", None)
        drain = getattr(self.native, "drain_eval", None)
        if callable(begin) and callable(submit) and callable(complete):
            if callable(drain):
                drain()
            begin(int(view_index))
            submit()
            return np.asarray(complete(), dtype=np.float32)
        self.native.render_dynamic_val_view(int(view_index))
        copy_rgb = getattr(self.native, "copy_pixel_rgb", None)
        if callable(copy_rgb):
            return np.asarray(copy_rgb(), dtype=np.float32)
        return np.array(self.native.pixel_state, dtype=np.float32, copy=True)[..., :3]

    def render_flow_eval(self, view_index: int, end_timestamp: float) -> np.ndarray:
        self.native.render_flow_dynamic_val_view(int(view_index), float(end_timestamp))
        return np.array(self.native.pixel_state, dtype=np.float32, copy=True)[..., :2]

    def materialize(self, timestamp: float) -> dict[str, np.ndarray]:
        self.native.materialize_dynamic(float(timestamp))
        return {
            "means": np.array(self.native.xyz_ws, dtype=np.float32, copy=True),
            "scales": np.array(self.native.scales, dtype=np.float32, copy=True),
            "opacities": np.array(self.native.opacities, dtype=np.float32, copy=True),
        }

    def performance_snapshot(self) -> dict[str, object]:
        """Return JSON-safe cumulative native timing and memory counters."""

        raw_timing = dict(self.native.get_timing_breakdown())
        stages: dict[str, dict[str, int | float]] = {}
        for name, value in raw_timing.items():
            count, total_seconds = value
            count = int(count)
            total_seconds = float(total_seconds)
            stages[str(name)] = {
                "count": count,
                "total_seconds": total_seconds,
                "mean_seconds": total_seconds / count if count else 0.0,
            }
        workload_method = getattr(self.native, "get_workload_counters", None)
        workload = (
            {str(key): int(value) for key, value in workload_method().items()}
            if callable(workload_method) else {}
        )
        tile_status_method = getattr(
            self.native, "get_tile_invariant_status", None
        )
        if callable(tile_status_method):
            tile_status = int(tile_status_method())
            workload["tile_invariant_status"] = tile_status
            if tile_status:
                raise RuntimeError(
                    "exact tile count/emit invariant failed "
                    f"(native status={tile_status})"
                )
        packed_host_bytes = (
            0 if self._packed_gradient_host is None
            else int(self._packed_gradient_host.nbytes)
        )
        pipeline = {}
        telemetry = getattr(self.native, "get_pipeline_telemetry", None)
        if callable(telemetry):
            pipeline = {
                str(key): float(value) for key, value in dict(telemetry()).items()
            }
        return {
            "device_id": int(self.device_id),
            "vram_bytes": int(self.native.get_vram_usage()),
            "peak_vram_bytes": int(self.native.get_peak_vram_usage()),
            "host_staging_bytes": packed_host_bytes,
            "workload": workload,
            "stages": stages,
            "pipeline": pipeline,
        }

    def benchmark_pair_transfer_buckets(
        self, candidate_bytes: list[int] | tuple[int, ...], *, rounds: int = 3,
    ) -> list[dict[str, object]]:
        """Measure fixed-payload Vulkan round trips without model gradients."""

        method = getattr(self.native, "benchmark_pair_transfer_buckets", None)
        if not callable(method):
            raise NativeBackendUnavailable(
                "native module lacks pair bucket calibration; rebuild it"
            )
        return [
            {str(key): value for key, value in dict(item).items()}
            for item in method(
                [int(value) for value in candidate_bytes], int(rounds)
            )
        ]

    def model(self) -> NativeGaussians:
        values = self.native.get_dynamic_gaussians()
        # Native buffer_to_array returns owning, contiguous NumPy allocations.
        # Retain those owners directly; NativeGaussians slices keep them alive
        # through ndarray.base, avoiding a second full model-sized host copy.
        arrays = {
            str(key): np.ascontiguousarray(value, dtype=np.float32)
            for key, value in values.items()
        }
        return NativeGaussians.from_native(arrays, self.max_duration)

    def optimizer_state(self) -> dict[str, np.ndarray]:
        return {
            str(key): np.ascontiguousarray(value, dtype=np.float32)
            for key, value in self.native.get_optimizer_state().items()
        }

    def restore_optimizer(self, values: Mapping[str, np.ndarray]) -> None:
        self.native.set_optimizer_state({key: np.ascontiguousarray(value, dtype=np.float32) for key, value in values.items()})

    def relocation_statistics(self) -> np.ndarray:
        stats = np.array(self.native.relocation_stats, dtype=np.float32, copy=True)
        return stats[:, 0] / np.maximum(stats[:, 1], 1.0)

    def relocation_stats_state(self) -> np.ndarray:
        """Download raw ``(magnitude_sum, observation_count)`` statistics."""
        method = getattr(self.native, "relocation_stats_state", None)
        if method is not None:
            return np.ascontiguousarray(method(), dtype=np.float32)
        return np.array(self.native.relocation_stats, dtype=np.float32, copy=True)

    def set_relocation_stats_state(self, values: np.ndarray) -> None:
        """Upload raw relocation statistics without touching Adam state."""
        statistics = np.ascontiguousarray(values, dtype=np.float32)
        if statistics.ndim != 2 or statistics.shape[1] != 2:
            raise ValueError(
                f"relocation statistics must have shape (N, 2), got {statistics.shape}"
            )
        method = getattr(self.native, "set_relocation_stats_state", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks set_relocation_stats_state; rebuild "
                "the multi-device Vulkan target"
            )
        method(statistics)

    def clear_relocation_stats_state(self) -> None:
        method = getattr(self.native, "clear_relocation_stats_state", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks clear_relocation_stats_state; rebuild "
                "the multi-device Vulkan target"
            )
        method()

    def apply_planned_relocation(self, mode: str) -> None:
        method = getattr(self.native, "apply_planned_relocation", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks apply_planned_relocation; rebuild "
                "the multi-device Vulkan target"
            )
        method(str(mode))

    def plan_relocation_native(
        self, step: int, include_gate: bool, seed: int,
    ) -> dict[str, object]:
        """Build the authoritative relocation plan on this device."""
        method = getattr(self.native, "plan_relocation_dynamic_native", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks plan_relocation_dynamic_native; "
                "rebuild the multi-device Vulkan target"
            )
        values = method(
            int(step), bool(include_gate), int(seed) & 0xFFFFFFFF
        )
        result = {str(key): value for key, value in values.items()}
        if "map" in result and "mapping" not in result:
            result["mapping"] = result.pop("map")
        for key in ("mapping", "counts"):
            if key in result:
                result[key] = np.ascontiguousarray(result[key], dtype=np.int32)
        return result

    def relocate(
        self,
        mapping: np.ndarray,
        counts: np.ndarray,
        mode: str,
        *,
        require_dead_destinations: bool = True,
        require_alive_donors: bool = True,
    ) -> None:
        self.native.relocate_dynamic(
            np.ascontiguousarray(mapping, dtype=np.int32),
            np.ascontiguousarray(counts, dtype=np.int32),
            mode,
            bool(require_dead_destinations),
            bool(require_alive_donors),
        )

    def relocate_native(self, step: int, mode: str, include_gate: bool, seed: int) -> dict[str, int | bool]:
        values = self.native.relocate_dynamic_native(
            int(step), str(mode), bool(include_gate), int(seed) & 0xFFFFFFFF)
        return {str(key): value for key, value in values.items()}

    def begin_hard_separation(self) -> dict[str, int]:
        values = self.native.begin_hard_separation()
        return {str(key): int(value) for key, value in values.items()}

    def resume_hard_separation(self) -> dict[str, int]:
        """Restore separation-phase runtime config without splitting again.

        A mid-separation checkpoint already contains the topology produced by
        ``begin_hard_separation``. Reapplying that operation would corrupt the
        model, so the native runtime must expose a config-only resume entry.
        """
        method = getattr(self.native, "resume_hard_separation", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks resume_hard_separation; rebuild a "
                "target with mid-hard-separation resume support"
            )
        values = method()
        return {str(key): int(value) for key, value in values.items()}

    def hard_separation_digest(self) -> str:
        """Hash every Gaussian's binary group and exact gate value.

        This phase-boundary diagnostic downloads only the packed
        velocity/gate buffer (not the full model) and prevents equal aggregate
        counts from hiding different per-Gaussian static/dynamic assignments.
        """

        method = getattr(self.native, "hard_separation_gate_state", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks hard_separation_gate_state; rebuild "
                "a target with exact cross-replica gate verification support"
            )
        # The native entry returns an owning NumPy allocation and immediately
        # releases its D2H staging vectors.  Reading the pybind property here
        # would keep a second 4*N-float host copy alive indefinitely.
        values = np.asarray(method(), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 4:
            raise RuntimeError(
                "native velocities_gates must have shape (N,4) for hard-"
                f"separation verification, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError(
                "native hard-separation velocity/gate state contains a "
                "non-finite value"
            )
        gates = np.ascontiguousarray(values[:, 3], dtype=np.float32)
        groups = np.ascontiguousarray(gates > 0.0, dtype=np.uint8)
        digest = hashlib.sha256()
        digest.update(np.asarray([len(gates)], np.int64).tobytes())
        digest.update(gates.tobytes())
        digest.update(groups.tobytes())
        return digest.hexdigest()

    def close(self) -> None:
        native = getattr(self, "native", None)
        try:
            if native is not None:
                if self._binds_rgb_per_step() and self._bound_train_images:
                    for bound_index in list(self._bound_train_images):
                        try:
                            native.set_dynamic_view_cache_on_device(
                                int(bound_index), True, False,
                            )
                            native.release_dynamic_view_image(
                                int(bound_index), True,
                            )
                        except Exception:
                            pass
                native.cleanup()
        finally:
            # A failed native cleanup is not safely retryable. Always sever the
            # Python owners so a later destructor/second close cannot dispatch
            # against a partially destroyed VkDevice and so multi-GB staging
            # arrays are released while preserving the original exception.
            self.native = None
            self._dataset = None
            self._bound_train_index = None
            self._bound_train_images.clear()
            self._packed_gradient_host = None


def camera_extent(dataset: NativeDataset) -> float:
    positions = np.stack([np.linalg.inv(view.w2c)[:3, 3] for view in dataset.train_views], axis=0)
    center = positions.mean(axis=0)
    return max(float(np.linalg.norm(positions-center, axis=1).max()), 1e-8)
