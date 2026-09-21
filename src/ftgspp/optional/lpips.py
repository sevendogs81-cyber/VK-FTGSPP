"""Optional LPIPS metric for the explicit locked final-test command."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

import numpy as np

from ftgspp.training.foreground_metrics import foreground_reference_roi


def _normalize_device_token(value: object) -> str:
    text = str(value).strip().lower()
    for prefix in ("uuid('", 'uuid("', "uuid("):
        if text.startswith(prefix) and text.endswith(")"):
            text = text[len(prefix):-1].strip("'\"")
            break
    return text.replace("-", "").replace(" ", "")


def _cuda_index_by_pci(pci_bdf: str) -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    wanted = pci_bdf.replace("0000:", "").lower()
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        bus = parts[1].lower()
        if bus == pci_bdf.lower() or bus.replace("0000:", "") == wanted:
            try:
                return int(parts[0])
            except ValueError:
                return None
    return None


def resolve_torch_device(
    requested: str = "auto",
    *,
    vulkan_metadata: dict[str, object] | None = None,
) -> str:
    """Prefer the CUDA device matching the Vulkan renderer, then CPU."""

    choice = str(requested).strip()
    lowered = choice.lower()
    if lowered == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS final-test evaluation requires the optional dependencies "
            "(install the project [lpips] extra)"
        ) from exc

    if lowered not in ("", "auto", "cuda"):
        if lowered.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"requested LPIPS device {choice!r} but CUDA is unavailable"
            )
        return choice
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        if lowered == "cuda":
            raise RuntimeError("requested LPIPS device 'cuda' but CUDA is unavailable")
        return "cpu"

    metadata = vulkan_metadata or {}
    vulkan_uuid = _normalize_device_token(
        metadata.get("uuid", metadata.get("device_uuid", "")),
    )
    vulkan_pci = str(
        metadata.get("pci_bdf", metadata.get("pci_bus_id", "")),
    ).strip().lower()
    if vulkan_uuid and vulkan_uuid != "unknown":
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_uuid = _normalize_device_token(getattr(props, "uuid", ""))
            if cuda_uuid and cuda_uuid == vulkan_uuid:
                return f"cuda:{index}"
    pci_index = (
        _cuda_index_by_pci(vulkan_pci)
        if vulkan_pci and vulkan_pci != "unknown" else None
    )
    if pci_index is not None and 0 <= pci_index < torch.cuda.device_count():
        return f"cuda:{pci_index}"

    free_index = 0
    free_bytes = -1
    for index in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(index)
        except (RuntimeError, TypeError):
            free = 0
        if int(free) > free_bytes:
            free_bytes = int(free)
            free_index = index
    return f"cuda:{free_index}"


def _is_cuda_oom(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


class LpipsVggEvaluator:
    """One reusable VGG LPIPS model with adaptive CUDA batching."""

    def __init__(
        self,
        *,
        device: str = "auto",
        vulkan_metadata: dict[str, object] | None = None,
    ) -> None:
        requested = str(device).strip() or "auto"
        try:
            import torch
            from torchmetrics.image.lpip import (
                LearnedPerceptualImagePatchSimilarity,
            )
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS final-test evaluation requires the optional dependencies "
                "(install the project [lpips] extra)"
            ) from exc

        self._torch = torch
        self._allow_cpu_fallback = requested.lower() == "auto"
        resolved = resolve_torch_device(
            requested, vulkan_metadata=vulkan_metadata,
        )
        self._device = torch.device(resolved)
        self._metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=False, reduction="none",
        )
        try:
            self._metric.to(self._device)
        except RuntimeError as exc:
            if not (
                self._allow_cpu_fallback
                and self._device.type == "cuda"
                and _is_cuda_oom(exc)
            ):
                raise
            torch.cuda.empty_cache()
            self._device = torch.device("cpu")
            self._metric.to(self._device)
        self._metric.eval()

    @property
    def device_name(self) -> str:
        return str(self._device)

    def evaluate(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
        *,
        batch_size: int,
    ) -> list[float]:
        if batch_size <= 0:
            raise ValueError("LPIPS batch size must be positive")
        if not pairs:
            return []
        shapes = {
            tuple(np.asarray(array).shape)
            for pair in pairs for array in pair
        }
        if len(shapes) != 1:
            raise ValueError("one LPIPS batch must contain one shared RGB shape")

        current_batch = min(int(batch_size), len(pairs))
        while True:
            try:
                return self._evaluate_batches(pairs, current_batch)
            except RuntimeError as exc:
                if not _is_cuda_oom(exc) or self._device.type != "cuda":
                    raise
                self._metric.reset()
                self._torch.cuda.empty_cache()
                if current_batch > 1:
                    current_batch = max(1, current_batch // 2)
                    continue
                if not self._allow_cpu_fallback:
                    raise
                self._metric.to("cpu")
                self._device = self._torch.device("cpu")
                self._torch.cuda.empty_cache()

    def _evaluate_batches(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
        batch_size: int,
    ) -> list[float]:
        torch = self._torch
        values: list[float] = []
        with torch.inference_mode():
            for offset in range(0, len(pairs), batch_size):
                chunk = pairs[offset:offset + batch_size]
                prediction = np.stack([
                    np.transpose(np.asarray(pred, dtype=np.float32), (2, 0, 1))
                    for pred, _reference in chunk
                ])
                reference = np.stack([
                    np.transpose(np.asarray(ref, dtype=np.float32), (2, 0, 1))
                    for _prediction, ref in chunk
                ])
                prediction_t = torch.from_numpy(prediction).to(
                    self._device, dtype=torch.float32,
                )
                reference_t = torch.from_numpy(reference).to(
                    self._device, dtype=torch.float32,
                )
                scores = self._metric(prediction_t, reference_t)
                self._metric.reset()
                values.extend(
                    float(value)
                    for value in torch.as_tensor(scores).reshape(-1).detach().cpu()
                )
        if len(values) != len(pairs):
            raise RuntimeError(
                f"LPIPS returned {len(values)} scores for {len(pairs)} frames"
            )
        return values


def evaluate_lpips_vgg_unnormalized(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    device: str = "auto",
    batch_size: int = 4,
    vulkan_metadata: dict[str, object] | None = None,
) -> list[float]:
    """Evaluate RGB ``[0, 1]`` pairs with VGG LPIPS and no input remapping."""

    if batch_size <= 0:
        raise ValueError("LPIPS batch size must be positive")
    if not pairs:
        return []
    groups: dict[
        tuple[int, ...], list[tuple[int, tuple[np.ndarray, np.ndarray]]]
    ] = {}
    for index, pair in enumerate(pairs):
        prediction, reference = pair
        shape = tuple(np.asarray(prediction).shape)
        if tuple(np.asarray(reference).shape) != shape:
            raise ValueError("LPIPS prediction/reference shapes differ")
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError("LPIPS inputs must be HxWx3 RGB arrays")
        groups.setdefault(shape, []).append((index, pair))
    evaluator = LpipsVggEvaluator(
        device=device, vulkan_metadata=vulkan_metadata,
    )
    ordered: list[float | None] = [None] * len(pairs)
    for group in groups.values():
        scores = evaluator.evaluate(
            [pair for _index, pair in group], batch_size=batch_size,
        )
        for (index, _pair), score in zip(group, scores, strict=True):
            ordered[index] = score
    if any(value is None for value in ordered):
        raise RuntimeError("LPIPS shape grouping did not score every frame")
    return [float(value) for value in ordered if value is not None]


class FinalTestLpipsMetric:
    """Stream final-test frames through LPIPS with bounded host memory."""

    requires_components = True

    def __init__(
        self,
        *,
        pipeline_mode: str,
        device: str = "auto",
        batch_size: int = 4,
        min_size: int = 64,
        vulkan_metadata: dict[str, object] | None = None,
        evaluator_factory: Callable[..., LpipsVggEvaluator] = LpipsVggEvaluator,
    ) -> None:
        if pipeline_mode not in {"scene", "foreground"}:
            raise ValueError(f"unsupported LPIPS pipeline mode: {pipeline_mode}")
        if int(batch_size) <= 0:
            raise ValueError("LPIPS batch size must be greater than zero")
        if int(min_size) < 32:
            raise ValueError("LPIPS minimum image size must be at least 32 pixels")
        if not str(device).strip():
            raise ValueError("LPIPS device must not be empty")
        self.pipeline_mode = pipeline_mode
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.min_size = int(min_size)
        self.vulkan_metadata = dict(vulkan_metadata or {})
        self._evaluator_factory = evaluator_factory
        self._evaluator: LpipsVggEvaluator | None = None
        self._records: list[dict[str, object]] = []
        self._buffers: dict[
            tuple[int, ...],
            list[tuple[int, np.ndarray, np.ndarray]],
        ] = {}
        self._setup_seconds = 0.0
        self._score_seconds = 0.0

    def _ensure_evaluator(self) -> LpipsVggEvaluator:
        if self._evaluator is None:
            started = time.perf_counter()
            self._evaluator = self._evaluator_factory(
                device=self.device,
                vulkan_metadata=self.vulkan_metadata,
            )
            self._setup_seconds += time.perf_counter() - started
        return self._evaluator

    @staticmethod
    def _reference_arrays(
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(image)
        scale = (
            np.float32(1.0 / 255.0)
            if raw.dtype == np.uint8 else np.float32(1.0)
        )
        rgb = np.asarray(raw[..., :3], np.float32) * scale
        alpha = (
            np.asarray(raw[..., 3], np.float32) * scale
            if raw.shape[-1] >= 4 else np.ones(raw.shape[:2], np.float32)
        )
        return np.clip(rgb, 0.0, 1.0), np.clip(alpha, 0.0, 1.0)

    def observe(
        self,
        *,
        view_index: int,
        view: object,
        background: np.ndarray,
        components: dict[str, np.ndarray],
    ) -> None:
        reference_image = components.get("reference_image")
        if reference_image is None:
            load_rgb = getattr(view, "load_rgb", None)
            if not callable(load_rgb):
                raise RuntimeError("LPIPS final test requires a reference image")
            reference_image = load_rgb()
        reference, alpha = self._reference_arrays(reference_image)
        foreground = np.asarray(components["foreground"], np.float32)
        transmittance = np.asarray(
            components["transmittance"], np.float32,
        ).squeeze()
        if (
            foreground.shape != reference.shape
            or transmittance.shape != reference.shape[:2]
        ):
            raise ValueError("LPIPS render/reference shapes differ")

        record: dict[str, object] = {
            "view": int(view_index),
            "camera": int(getattr(view, "camera_id")),
            "frame": int(getattr(view, "frame")),
            "lpips": None,
            "skip_reason": None,
        }
        self._records.append(record)
        record_index = len(self._records) - 1

        if self.pipeline_mode == "scene":
            prediction = np.clip(
                foreground + transmittance[..., None] * np.asarray(
                    background, np.float32,
                ),
                0.0,
                1.0,
            )
            target = reference
            y0, y1, x0, x1 = 0, reference.shape[0], 0, reference.shape[1]
            region = "full_frame"
        else:
            validity = components.get("validity")
            if validity is None:
                load_validity = getattr(view, "load_validity", None)
                validity = (
                    load_validity()
                    if callable(load_validity)
                    else np.ones(alpha.shape, np.float32)
                )
            validity = np.asarray(validity).squeeze() >= 0.5
            tolerance = max(1, round(2 * max(alpha.shape) / 1024))
            roi = foreground_reference_roi(
                alpha, validity, padding=tolerance,
            )
            if roi is None:
                record["skip_reason"] = "empty_reference_foreground"
                return
            y0, y1, x0, x1 = roi
            if not bool(np.all(validity[y0:y1, x0:x1])):
                record["skip_reason"] = "roi_not_fully_valid"
                return
            prediction = np.clip(
                foreground[y0:y1, x0:x1], 0.0, 1.0,
            )
            target = np.clip(
                reference[y0:y1, x0:x1] * alpha[y0:y1, x0:x1, None],
                0.0,
                1.0,
            )
            region = "reference_foreground_roi"

        record["region"] = {
            "kind": region,
            "y0": int(y0), "y1": int(y1),
            "x0": int(x0), "x1": int(x1),
        }
        if y1 - y0 < self.min_size or x1 - x0 < self.min_size:
            record["skip_reason"] = "region_too_small"
            return
        pair = (
            np.ascontiguousarray(prediction, dtype=np.float32),
            np.ascontiguousarray(target, dtype=np.float32),
        )
        shape = tuple(pair[0].shape)
        buffer = self._buffers.setdefault(shape, [])
        buffer.append((record_index, *pair))
        if len(buffer) >= self.batch_size:
            self._flush(shape, self.batch_size)
        elif sum(len(items) for items in self._buffers.values()) >= self.batch_size:
            fullest = max(self._buffers, key=lambda key: len(self._buffers[key]))
            self._flush(fullest)

    def _flush(self, shape: tuple[int, ...], count: int | None = None) -> None:
        buffer = self._buffers[shape]
        take = len(buffer) if count is None else min(int(count), len(buffer))
        if take <= 0:
            return
        chunk = buffer[:take]
        del buffer[:take]
        if not buffer:
            del self._buffers[shape]
        evaluator = self._ensure_evaluator()
        started = time.perf_counter()
        values = evaluator.evaluate(
            [(prediction, target) for _index, prediction, target in chunk],
            batch_size=self.batch_size,
        )
        self._score_seconds += time.perf_counter() - started
        for (record_index, _prediction, _target), value in zip(
            chunk, values, strict=True,
        ):
            self._records[record_index]["lpips"] = float(value)

    def finalize(
        self,
        *,
        vulkan_metadata: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, float]]:
        if not self.vulkan_metadata:
            self.vulkan_metadata = dict(vulkan_metadata)
        for shape in tuple(self._buffers):
            self._flush(shape)
        values = [
            float(record["lpips"])
            for record in self._records
            if record["lpips"] is not None
        ]
        actual_device = (
            self._evaluator.device_name
            if self._evaluator is not None else None
        )
        metrics: dict[str, object] = {
            "lpips": float(np.mean(values)) if values else None,
            "lpips_views": len(values),
            "lpips_skipped_views": len(self._records) - len(values),
            "lpips_view_metrics": self._records,
            "lpips_protocol": {
                "backbone": "vgg",
                "normalize": False,
                "rgb_range": [0.0, 1.0],
                "region": (
                    "full_frame" if self.pipeline_mode == "scene"
                    else "reference_foreground_roi"
                ),
                "minimum_side": self.min_size,
                "requested_device": self.device,
                "actual_device": actual_device,
            },
        }
        timings = {
            "lpips_model_setup": float(self._setup_seconds),
            "lpips_score": float(self._score_seconds),
        }
        return metrics, timings
