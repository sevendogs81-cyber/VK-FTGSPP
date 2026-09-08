# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

"""Torch model shims used only by checkpoint conversion/interoperability."""

from ftgspp.compat.torch_models.colorcorrect import ColorCorrector, ColorCorrectors
from ftgspp.compat.torch_models.gaussians import Gaussians
from ftgspp.compat.torch_models.velocity import Bounds, ExplicitVelocities, VelocityField

__all__ = [
    "ColorCorrector",
    "ColorCorrectors",
    "Gaussians",
    "Bounds",
    "ExplicitVelocities",
    "VelocityField",
]
