"""Optional Torch LPIPS plugin for native Vulkan training/eval.

LPIPS is intentionally *not* ported to Vulkan shaders. Native rasterization
produces RGB in ``[0, 1]``; this module computes ``λ · LPIPS`` and the
corresponding pixel gradient on a Torch device when training. Evaluation can
request a scalar-only forward pass and avoid constructing a gradient.

Requires the optional ``lpips`` extras (torch + torchmetrics). Color
correction applies before LPIPS when both are enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LpipsResult:
    loss: float
    # Training returns (H,W,4) float32 with a zero alpha channel. Scalar-only
    # evaluation deliberately returns None so callers cannot mistake a dummy
    # array for a valid gradient to inject into the Vulkan loss graph.
    pixel_grad: np.ndarray | None


class LpipsPlugin:
    """Lazy Torch LPIPS wrapper kept out of the default native import path."""

    def __init__(self, weight: float = 0.01, net_type: str = "alex", device: str | None = None):
        self.weight = float(weight)
        self.net_type = net_type
        self._device_name = device
        self._metric = None
        self._torch = None
        self._device = None

    def _ensure(self) -> None:
        if self._metric is not None:
            return
        try:
            import torch
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        except ImportError as exc:
            raise RuntimeError(
                "train.lpips_loss requires optional Torch deps "
                "(install with pip install -e '.[lpips]')"
            ) from exc
        self._torch = torch
        if self._device_name:
            self._device = torch.device(self._device_name)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._metric = LearnedPerceptualImagePatchSimilarity(
            net_type=self.net_type, normalize=True,
        ).to(self._device)
        self._metric.eval()

    def compute(
        self,
        prediction_rgb: np.ndarray,
        reference_rgb: np.ndarray,
        *,
        weight: float | None = None,
        require_gradient: bool = True,
        mask: np.ndarray | None = None,
    ) -> LpipsResult:
        """Return weighted LPIPS and optionally ∂L/∂prediction.

        Training uses the configured ``self.weight``. ``mask`` is the optional
        constant reference-alpha mask applied to both images inside the clamp
        graph. Evaluation passes
        ``weight=1, require_gradient=False`` so the reported LPIPS remains the
        standard unweighted reconstruction metric without building a backward
        graph or allocating a pixel-gradient image.
        """
        self._ensure()
        torch = self._torch
        assert self._metric is not None and torch is not None and self._device is not None

        pred = np.asarray(prediction_rgb, dtype=np.float32)
        ref = np.asarray(reference_rgb, dtype=np.float32)
        if pred.shape != ref.shape or pred.ndim != 3 or pred.shape[2] != 3:
            raise ValueError(f"expected matching (H,W,3) RGB arrays, got {pred.shape} vs {ref.shape}")

        if mask is None:
            mask_array = np.ones((*pred.shape[:2], 1), dtype=np.float32)
        else:
            mask_array = np.asarray(mask, dtype=np.float32)
            if mask_array.shape == pred.shape[:2]:
                mask_array = mask_array[..., None]
            if mask_array.shape != (*pred.shape[:2], 1):
                raise ValueError(
                    "LPIPS mask must have shape (H,W) or (H,W,1), got "
                    f"{mask_array.shape}"
                )
            if (
                not np.isfinite(mask_array).all()
                or np.any(mask_array < 0.0)
                or np.any(mask_array > 1.0)
            ):
                raise ValueError("LPIPS mask must be finite and in [0,1]")

        effective_weight = self.weight if weight is None else float(weight)
        if not np.isfinite(effective_weight) or effective_weight < 0.0:
            raise ValueError("LPIPS weight must be finite and non-negative")
        if not isinstance(require_gradient, bool):
            raise TypeError("require_gradient must be a bool")

        mask_t = (
            torch.as_tensor(mask_array, device=self._device)
            .permute(2, 0, 1).unsqueeze(0)
        )
        ref_t = (
            torch.as_tensor(ref, device=self._device)
            .permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        ) * mask_t
        if not require_gradient:
            # Evaluation consumes only the standard scalar. no_grad avoids the
            # full LPIPS activation graph and the HxWx4 host gradient transfer.
            with torch.no_grad():
                pred_t = (
                    torch.as_tensor(pred, device=self._device)
                    .permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
                ) * mask_t
                value = effective_weight * self._metric(pred_t, ref_t)
            loss = float(value.detach().cpu())
            if hasattr(self._metric, "reset"):
                self._metric.reset()
            return LpipsResult(loss=loss, pixel_grad=None)

        # Keep the unclamped prediction as the differentiation leaf. Detaching
        # after ``clamp`` would silently drop its Jacobian and inject a non-zero
        # gradient for values outside [0, 1], unlike FTGSPP's LPIPS graph.
        pred_raw = (
            torch.as_tensor(pred, device=self._device)
            .permute(2, 0, 1).unsqueeze(0)
            .detach().requires_grad_(True)
        )
        # RGBA datasets use the reference alpha as a constant RGB loss mask,
        # exactly like the native L1/SSIM path. Keeping the multiplication in
        # this graph applies both the alpha factor and clamp Jacobian to the
        # gradient injected back into Vulkan.
        pred_t = pred_raw.clamp(0, 1) * mask_t
        value = effective_weight * self._metric(pred_t, ref_t)
        value.backward()
        loss = float(value.detach().cpu())
        if hasattr(self._metric, "reset"):
            self._metric.reset()

        grad = (
            pred_raw.grad.detach().squeeze(0).permute(1, 2, 0)
            .cpu().numpy().astype(np.float32)
        )
        pixel_grad = np.zeros((pred.shape[0], pred.shape[1], 4), dtype=np.float32)
        pixel_grad[..., :3] = grad
        return LpipsResult(loss=loss, pixel_grad=pixel_grad)


def try_create(weight: float = 0.01) -> LpipsPlugin | None:
    """Return a plugin if Torch LPIPS is importable, else ``None``."""
    try:
        plugin = LpipsPlugin(weight=weight)
        plugin._ensure()
        return plugin
    except RuntimeError:
        return None
