"""Optional Torch LPIPS implementation used exclusively during evaluation.

The native training objective stays entirely in Vulkan. Evaluation invokes
this module after rendering to compute scalar foreground ROI scores.
"""

from __future__ import annotations

import numpy as np


def _normalize_device_token(value: object) -> str:
    text = str(value).strip().lower()
    for prefix in ("uuid('", 'uuid("', "uuid("):
        if text.startswith(prefix) and text.endswith(")"):
            text = text[len(prefix):-1].strip("'\"")
            break
    return text.replace("-", "").replace(" ", "")


def resolve_torch_device(
    requested: str = "auto",
    *,
    vulkan_metadata: dict[str, object] | None = None,
) -> str:
    """Pick a Torch device for post-Vulkan LPIPS.

    ``auto`` prefers the CUDA ordinal whose UUID or PCI bus matches the Vulkan
    GPU that just rendered, then any CUDA device, then CPU. Explicit ``cpu``
    or ``cuda[:index]`` values are honored.
    """

    choice = str(requested).strip()
    lowered = choice.lower()
    if lowered == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        if lowered in ("", "auto"):
            return "cpu"
        raise RuntimeError(
            "LPIPS evaluation requires optional Torch deps "
            "(install with pip install -e '.[lpips]')"
        ) from exc

    if lowered not in ("", "auto", "cuda"):
        if lowered.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested LPIPS device {choice!r} but CUDA is unavailable")
        return choice
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return "cpu"

    metadata = vulkan_metadata or {}
    vulkan_uuid = _normalize_device_token(
        metadata.get("uuid", metadata.get("device_uuid", "")),
    )
    vulkan_pci = str(metadata.get("pci_bdf", metadata.get("pci_bus_id", ""))).strip().lower()
    if vulkan_uuid and vulkan_uuid != "unknown":
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_uuid = _normalize_device_token(getattr(props, "uuid", ""))
            if cuda_uuid and cuda_uuid == vulkan_uuid:
                return f"cuda:{index}"
    pci_index = _cuda_index_by_pci(vulkan_pci) if vulkan_pci and vulkan_pci != "unknown" else None
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


def _cuda_index_by_pci(pci_bdf: str) -> int | None:
    import subprocess
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


def evaluate_lpips_vgg_unnormalized(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    device: str = "auto",
    batch_size: int = 4,
    vulkan_metadata: dict[str, object] | None = None,
) -> list[float]:
    """3DGS Table-S2 LPIPS: VGG, ``normalize=False``, RGB in ``[0, 1]``.

    Runs after Vulkan rendering has produced the ROI pairs. ``reduction='none'``
    keeps per-frame scores; batch size falls back on OOM.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not pairs:
        return []
    shapes = {tuple(np.asarray(pred).shape) for pred, _ref in pairs}
    current_batch = 1 if len(shapes) > 1 else int(batch_size)
    torch_device_name = resolve_torch_device(
        device, vulkan_metadata=vulkan_metadata,
    )
    try:
        import torch
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS evaluation requires optional Torch deps "
            "(install with pip install -e '.[lpips]')"
        ) from exc

    torch_device = torch.device(torch_device_name)
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", normalize=False, reduction="none",
    ).to(torch_device)
    metric.eval()
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    while True:
        try:
            return _lpips_vgg_batches(
                pairs, metric=metric, torch_module=torch,
                torch_device=torch_device, batch_size=current_batch,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if current_batch <= 1 or "out of memory" not in message:
                raise
            current_batch = max(1, current_batch // 2)
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()


def _lpips_vgg_batches(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    metric,
    torch_module,
    torch_device,
    batch_size: int,
) -> list[float]:
    torch = torch_module
    values: list[float] = []
    with torch.inference_mode():
        for offset in range(0, len(pairs), batch_size):
            chunk = pairs[offset:offset + batch_size]
            pred = np.stack(
                [np.transpose(np.asarray(pred, dtype=np.float32), (2, 0, 1))
                 for pred, _ref in chunk],
                axis=0,
            )
            ref = np.stack(
                [np.transpose(np.asarray(ref, dtype=np.float32), (2, 0, 1))
                 for _pred, ref in chunk],
                axis=0,
            )
            pred_t = torch.from_numpy(pred).to(torch_device, dtype=torch.float32)
            ref_t = torch.from_numpy(ref).to(torch_device, dtype=torch.float32)
            scores = metric(pred_t, ref_t)
            if hasattr(metric, "reset"):
                metric.reset()
            scores = torch.as_tensor(scores).reshape(-1)
            values.extend(float(value) for value in scores.detach().cpu())
    if len(values) != len(pairs):
        raise RuntimeError(
            f"LPIPS returned {len(values)} scores for {len(pairs)} frames"
        )
    return values
