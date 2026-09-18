"""Thin NumPy binding for the Vulkan-owned FTGSPP training engine."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from ftgspp.data.scene import Scene, View
from ftgspp.io.model import NativeGaussians

logger = logging.getLogger("ftgspp.native")


class NativeBackendUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _native_extension_identity(path_value: str) -> dict[str, object]:
    """Hash a loaded extension path once per process for V6 provenance."""

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
    if override:
        return Path(override).expanduser().resolve()
    import ftgspp
    return Path(ftgspp.__file__).resolve().parent / "shaders"


def resolve_view_load_options(config) -> dict[str, object]:
    """Resolve the lazy RGB cache and bounded lookahead options."""
    legacy_mode = os.environ.get("FTGSPP_VIEW_LOAD_MODE")
    if legacy_mode is not None and legacy_mode.strip().lower() != "lazy":
        raise ValueError(
            "FTGSPP_VIEW_LOAD_MODE is obsolete; native view loading is always "
            "lazy"
        )
    cache_raw = os.environ.get("FTGSPP_VIEW_RGB_CACHE")
    rgb_cache = int(cache_raw) if cache_raw is not None else int(config.run.view_rgb_cache)
    if os.environ.get("FTGSPP_VK_IMAGE_CACHE") is not None:
        raise ValueError(
            "FTGSPP_VK_IMAGE_CACHE is obsolete; device image-cache options "
            "have been removed"
        )
    workers_raw = os.environ.get("FTGSPP_VIEW_LOAD_WORKERS")
    workers = int(workers_raw) if workers_raw is not None else int(config.run.view_load_workers)
    if rgb_cache <= 0:
        raise ValueError("view_rgb_cache must be > 0")
    if workers < 0:
        raise ValueError("view_load_workers must be >= 0")
    return {
        "rgb_cache": rgb_cache,
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

    The cache is always bounded. Training never preloads the complete dataset.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._items: OrderedDict[tuple[bool, int], np.ndarray] = OrderedDict()
        self._pending: dict[tuple[bool, int], Future[np.ndarray]] = {}
        # Lookahead futures decode pixels off the critical path, but do not
        # enter/touch the deterministic LRU until their batch is consumed.
        self._scheduled: dict[tuple[bool, int], Future[np.ndarray]] = {}
        self._prefetched_reads: dict[tuple[bool, int], int] = {}
        self._lock = threading.RLock()

    def _store(self, key: tuple[bool, int], rgb: np.ndarray) -> np.ndarray:
        value = _canonicalize_reference_rgba(rgb)
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                evicted, _value = self._items.popitem(last=False)
                self._prefetched_reads.pop(evicted, None)
        return value

    def get_or_load(self, *, is_train: bool, index: int, view: View) -> np.ndarray:
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
        requests: list[tuple[bool, int, View]],
        *,
        workers: int,
        executor: Executor | None = None,
    ) -> None:
        """Decode a deterministic ordered request set before rank dispatch."""

        ordered: list[tuple[tuple[bool, int], View]] = []
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
        requests: list[tuple[bool, int, View]],
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
            from ftgspp import _ftgspp_vk as native_module
        except ImportError as exc:
            raise NativeBackendUnavailable(
                "_ftgspp_vk is not built; build the standalone native target on the Vulkan server"
            ) from exc
        FTGSPPVulkan = native_module.FTGSPPVulkan
        self._native_extension_metadata = _native_extension_identity(
            str(getattr(native_module, "__file__", ""))
        )
        shader_dir = _shader_dir()
        manifest = json.loads((shader_dir / "manifest.json").read_text())
        missing = [name for name in manifest["files"] if not (shader_dir / name).is_file()]
        if missing:
            raise NativeBackendUnavailable(f"installed shaders are incomplete: {missing}")
        self.native = FTGSPPVulkan()
        self.native.initialize(str(shader_dir) + os.sep, int(device_id))
        self.device_id = int(device_id)
        self.max_duration = 0.0
        self._dataset: Scene | None = None
        self._rgb_cache = SharedRgbViewCache(64)
        self._view_load_workers = 0
        self._bound_train_index: int | None = None
        self._num_gaussians = 0
        self._image_objective = "scene_rgb_v1"

    def configure(self, config, scene_extent: float) -> None:
        lrs, schedules = config.train.lrs, config.train.lr_schedules
        self.max_duration = float(config.model.max_duration)
        self._image_objective = str(config.pipeline.image_objective)
        self.native.configure_ftgspp({
            "max_steps": int(config.train.iterations),
            "max_sh_degree": int(config.init.sh_degree),
            "effective_pair_adam": (
                config.distributed.mode == "sync"
                and config.distributed.world_size_lr_scaling == "effective_batch_adam"
            ),
            "scene_scale": float(scene_extent),
            "ssim_lambda": 0.2, "means_lr": float(lrs["means"]),
            "image_objective": {
                "scene_rgb_v1": 0,
                "composite_v1": 1,
            }[self._image_objective],
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
            "marginal_threshold": float(config.model.marginal_threshold),
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

    def _camera_index_map(self, dataset: Scene) -> tuple[dict[int, int], list[int]]:
        camera_ids = sorted({int(c) for c in dataset.cameras})
        return {cid: i for i, cid in enumerate(camera_ids)}, camera_ids

    def _add_view_meta(
        self,
        view: View,
        is_train: bool,
        *,
        camera_index: int,
    ) -> None:
        height, width = view.size
        self.native.add_dynamic_view_meta(
            int(height), int(width), view.w2c, view.intrinsic, float(view.timestamp),
            int(camera_index), int(view.frame), bool(is_train),
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
        dataset: Scene,
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

    def upload_dataset(
        self,
        dataset: Scene,
        *,
        rgb_cache: int = 64,
        workers: int = 8,
        shared_rgb_cache: SharedRgbViewCache | None = None,
    ) -> None:
        self.native.clear_dynamic_views()
        self._dataset = dataset
        self._bound_train_index = None
        self._view_load_workers = int(workers)
        camera_to_index, camera_ids = self._camera_index_map(dataset)
        self._upload_dataset_lazy(
            dataset,
            rgb_cache=rgb_cache,
            camera_to_index=camera_to_index,
            camera_ids=camera_ids,
            shared_rgb_cache=shared_rgb_cache,
        )
        if self.native.num_train != len(dataset.train_views) or self.native.num_val != len(dataset.eval_views):
            raise RuntimeError("native dataset registration count mismatch")

    def _release_bound_train_image(self, view_index: int) -> None:
        self.native.release_dynamic_view_image(int(view_index), True)
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
    ) -> list[tuple[bool, int, View]]:
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
        """Return contiguous RGB only for three-channel metrics.

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

        if self._dataset is None:
            raise RuntimeError("upload_dataset must be called before train_view")
        view = self._dataset.train_views[int(view_index)]
        rgb = self._rgb_cache.get_or_load(is_train=True, index=int(view_index), view=view)
        if self._bound_train_index == int(view_index):
            return rgb
        if self._bound_train_index is not None:
            self._release_bound_train_image(int(self._bound_train_index))
        self.native.bind_dynamic_view_image(int(view_index), True, rgb)
        validity = np.ascontiguousarray(
            view.load_validity() >= 0.5, dtype=np.uint8,
        )
        if not bool(np.all(validity)):
            bind_validity = getattr(self.native, "bind_dynamic_view_validity", None)
            if not callable(bind_validity):
                raise NativeBackendUnavailable(
                    "the native module lacks validity-buffer support; rebuild "
                    "the Vulkan target"
                )
            bind_validity(int(view_index), True, validity)
        self._bound_train_index = int(view_index)
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
        background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, float] | None:
        """Run one native training step using the Vulkan image objective."""
        self._bind_train_rgb(int(view_index))
        background_rgb = np.asarray(background, dtype=np.float32)
        if background_rgb.shape != (3,) or not np.isfinite(background_rgb).all():
            raise ValueError("background must be a finite RGB vector")
        background_tuple = tuple(float(value) for value in background_rgb)
        self.native.train_dynamic_view(
            int(view_index), int(step), bool(collect_loss), background_tuple)
        self._maybe_log_pipeline_telemetry(int(step))
        return None

    def backward_view(
        self, view_index: int, step: int, *, collect_loss: bool = True,
        background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Compute one view's raw FP32 gradients without updating Adam.

        This split entry point is the synchronization boundary used by
        replicated multi-device training. Exactly one raw view gradient is
        finalized on each rank before the cross-rank mean and Adam update.
        """
        self._bind_train_rgb(int(view_index))
        method = getattr(self.native, "backward_dynamic_view", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks backward_dynamic_view; rebuild the "
                "multi-device Vulkan target"
            )
        background_array = np.asarray(background, np.float32)
        if background_array.shape != (3,) or not np.isfinite(background_array).all():
            raise ValueError("background must be a finite RGB vector")
        color = tuple(float(value) for value in background_array)
        method(int(view_index), int(step), bool(collect_loss), color)

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

    def supports_pair_exchange_gpu_merge(self) -> bool:
        """Probe the native two-device exchange contract."""

        return all(
            callable(getattr(self.native, name, None))
            for name in (
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
        self, *, world_size: int, bucket_bytes: int, sparse: bool, timeout_seconds: float = 120.0,
    ) -> dict[str, object]:
        method = getattr(self.native, "create_pair_exchange_gpu_merge_owner", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks the pair-exchange owner capability")
        return dict(method(int(world_size), int(bucket_bytes), bool(sparse), float(timeout_seconds)))

    def join_pair_session(self, leader: NativeVulkanBackend) -> None:
        self.native.join_pair_session(leader.native)

    def abort_pair_session(self, reason: str) -> None:
        self.native.abort_pair_session(reason)

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
        self, step: int, rank0_timestamp: float, rank1_timestamp: float,
    ) -> dict[str, float]:
        method = getattr(self.native, "optimizer_step_pair_exchange_gpu_merge", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks GPU peer-mean optimizer step")
        return {
            str(key): float(value)
            for key, value in method(
                int(step), float(rank0_timestamp), float(rank1_timestamp),
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
        method = getattr(self.native, "release_pair_exchange_gpu_merge_owner", None)
        if callable(method):
            method()

    def reset_pair_exchange_gpu_merge_attachments(self) -> None:
        method = getattr(self.native, "reset_pair_exchange_gpu_merge_attachments", None)
        if method is None:
            raise NativeBackendUnavailable(
                "the native module lacks mailbox attachment reset")
        method()

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
                "with uv build"
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
                "with uv build"
            )
        values = np.ascontiguousarray(params, dtype=np.float32).reshape(-1)
        method(values)

    def color_params(self) -> np.ndarray | None:
        """Download only the inference-time affine parameters, not all Adam state."""

        method = getattr(self.native, "get_color_correction_params", None)
        if not callable(method):
            raise NativeBackendUnavailable(
                "the native module lacks color snapshot support; rebuild the "
                "Vulkan target"
            )
        values = np.ascontiguousarray(method(), dtype=np.float32)
        return None if values.size == 0 else values

    def render_eval(self, view_index: int) -> np.ndarray:
        begin = getattr(self.native, "begin_eval_val_view", None)
        submit = getattr(self.native, "submit_eval_rgb_readback", None)
        complete = getattr(self.native, "complete_eval_rgb", None)
        drain = getattr(self.native, "drain_eval", None)
        if callable(begin) and callable(submit) and callable(complete):
            if callable(drain):
                drain()
            # Hold-out cameras report raw raster RGB even when CC is trained.
            begin(int(view_index), False)
            submit()
            return np.asarray(complete(), dtype=np.float32)
        self.native.render_dynamic_val_view(int(view_index), False)
        copy_rgb = getattr(self.native, "copy_pixel_rgb", None)
        if callable(copy_rgb):
            return np.asarray(copy_rgb(), dtype=np.float32)
        return np.array(self.native.pixel_state, dtype=np.float32, copy=True)[..., :3]

    def render_components(
        self, view_index: int, *, is_train: bool = False,
        apply_color_correction: bool = False,
    ) -> dict[str, np.ndarray]:
        """Return named raster components; the fourth state channel is T."""

        if is_train:
            self._bind_train_rgb(int(view_index))
            self.native.render_dynamic_train_view(
                int(view_index), apply_color_correction=bool(apply_color_correction),
            )
        else:
            begin = getattr(self.native, "begin_eval_val_view", None)
            submit = getattr(self.native, "submit_eval_rgb_readback", None)
            complete = getattr(self.native, "complete_eval_components", None)
            drain = getattr(self.native, "drain_eval", None)
            if all(callable(method) for method in (begin, submit, complete)):
                if callable(drain):
                    drain()
                begin(int(view_index), bool(apply_color_correction))
                submit()
                state = np.asarray(complete(), dtype=np.float32)
            else:
                self.native.render_dynamic_val_view(
                    int(view_index), bool(apply_color_correction),
                )
                state = np.array(
                    self.native.pixel_state, dtype=np.float32, copy=True,
                )
        if is_train:
            state = np.array(self.native.pixel_state, dtype=np.float32, copy=True)
        if state.ndim != 3 or state.shape[-1] != 4:
            raise RuntimeError(f"native component readback must be HxWx4, got {state.shape}")
        return {
            "foreground": np.ascontiguousarray(state[..., :3]),
            "transmittance": np.ascontiguousarray(state[..., 3]),
        }

    def evaluate_holdout_view(
        self,
        view_index: int,
        *,
        background: np.ndarray | list[float],
        bind_validity: bool,
        read_components: bool,
    ) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
        """Render one hold-out view and reduce ranking metrics on the GPU.

        Diagnostic F/T readback is opt-in because training-loop ranking does
        not need it. There is no CPU SSIM/PSNR fallback.
        """

        if self._dataset is None:
            raise RuntimeError("upload_dataset must be called before hold-out eval")
        begin = getattr(self.native, "begin_eval_val_view", None)
        complete_metrics = getattr(self.native, "complete_eval_metrics", None)
        bind_image = getattr(self.native, "bind_dynamic_view_image", None)
        bind_validity_fn = getattr(self.native, "bind_dynamic_view_validity", None)
        drain = getattr(self.native, "drain_eval", None)
        if not all(callable(fn) for fn in (begin, complete_metrics, bind_image)):
            raise RuntimeError(
                "hold-out eval requires native GPU complete_eval_metrics; rebuild"
            )
        color = np.asarray(background, dtype=np.float32).reshape(3)
        rgb = self.cached_view_image(is_train=False, index=int(view_index))
        bind_image(int(view_index), False, rgb)
        if bind_validity:
            if not callable(bind_validity_fn):
                raise RuntimeError(
                    "foreground eval requires bind_dynamic_view_validity"
                )
            views = self._dataset.eval_views
            validity = views[int(view_index)].load_validity()
            bind_validity_fn(
                int(view_index),
                False,
                np.ascontiguousarray(validity >= 0.5, dtype=np.uint8),
            )
        if callable(drain):
            drain()
        # Known validation cameras never consume trained per-camera affines.
        begin(int(view_index), False)
        gpu = dict(complete_metrics((float(color[0]), float(color[1]), float(color[2]))))
        required = (
            "squared_error_sum",
            "ssim_sum",
            "foreground_squared_error_sum",
            "foreground_samples",
            "valid_pixels",
            "valid_windows",
        )
        missing = [key for key in required if key not in gpu]
        if missing:
            raise RuntimeError(
                "complete_eval_metrics is missing " + ", ".join(missing)
            )
        gpu = {key: float(gpu[key]) for key in required}
        components = None
        if read_components:
            submit = getattr(self.native, "submit_eval_rgb_readback", None)
            complete = getattr(self.native, "complete_eval_components", None)
            if not callable(submit) or not callable(complete):
                raise RuntimeError(
                    "foreground diagnostic eval requires complete_eval_components"
                )
            submit()
            state = np.asarray(complete(), dtype=np.float32)
            if state.ndim != 3 or state.shape[-1] != 4:
                raise RuntimeError(
                    f"native component readback must be HxWx4, got {state.shape}"
                )
            components = {
                "foreground": np.ascontiguousarray(state[..., :3]),
                "transmittance": np.ascontiguousarray(state[..., 3]),
            }
        return gpu, components

    def render_flow_diagnostic(self, view_index: int, end_timestamp: float) -> np.ndarray:
        """Render model flow for diagnostics; default evaluation does not call it."""
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
            "host_staging_bytes": 0,
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

    def rasterizer_state(self) -> str:
        return self.native.rasterizer_state()

    def restore_rasterizer_state(self, state: str) -> None:
        self.native.restore_rasterizer_state(state)

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
                if self._bound_train_index is not None:
                    try:
                        native.release_dynamic_view_image(
                            int(self._bound_train_index), True,
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


def camera_extent(dataset: Scene) -> float:
    source_views = dataset.train_views or dataset.eval_views
    if not source_views:
        raise ValueError("camera extent requires at least one scene view")
    positions = np.stack([np.linalg.inv(view.w2c)[:3, 3] for view in source_views], axis=0)
    center = positions.mean(axis=0)
    return max(float(np.linalg.norm(positions-center, axis=1).max()), 1e-8)
