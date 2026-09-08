# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from math import isqrt
from typing import Optional, Self

import torch
from torch import Tensor, nn

from ftgspp.compat.torch_models.velocity import ExplicitVelocities, VelocityField
from ftgspp.utils import PathLike

MARGINAL_T_THRESHOLD = 0.05
TEMPORAL_SCALE_MULTIPLIER = 2.0


class Gaussians(nn.Module):
    means: nn.Parameter  # (n, 3)
    scales: nn.Parameter  # (n, 3)
    quats: nn.Parameter  # (n, 4)
    opacities: nn.Parameter  # (n, 1)
    sh_0: nn.Parameter  # (n, 1, 3)
    sh_n: nn.Parameter  # (n, (sh_degree + 1) ** 2 - 1, 3)
    times: nn.Parameter  # (n, 1)
    durations: nn.Parameter  # (n, 1)
    velocity_model: VelocityField | ExplicitVelocities

    marginal_gates: nn.Parameter  # (n, 3)
    sh_degree: int
    max_duration: float

    def __init__(
        self,
        means: Tensor,
        scales: Tensor,
        quats: Tensor,
        opacities: Tensor,
        sh_0: Tensor,
        sh_n: Tensor,
        times: Tensor,
        durations: Tensor,
        velocity_model: VelocityField | Tensor,
        max_duration: float,
        marginal_gates: Optional[Tensor] = None,
    ):
        super().__init__()

        self.means = nn.Parameter(means.float())
        self.scales = nn.Parameter(scales.float())
        self.quats = nn.Parameter(quats.float())
        self.opacities = nn.Parameter(opacities.float())
        self.sh_0 = nn.Parameter(sh_0.float())
        self.sh_n = nn.Parameter(sh_n.float())
        self.times = nn.Parameter(times.float())
        self.durations = nn.Parameter(durations.float())

        if isinstance(velocity_model, Tensor):
            velocity_model = nn.Parameter(velocity_model)
        self.velocity_model = velocity_model

        if marginal_gates is None:
            self.marginal_gates = nn.Parameter(torch.full((len(self), 1), -1).float())
        else:
            self.marginal_gates = nn.Parameter(marginal_gates)

        self.sh_degree = isqrt(sh_n.shape[1] + 1) - 1
        self.max_duration = max_duration

    def means_t(self, t: float | Tensor) -> Tensor:
        return self.means + (t - self.times) * self.velocities_t(t)

    def velocities_t(self, t: float | Tensor):
        match self.velocity_model:
            case VelocityField():
                # The Vulkan-owned Gaussian parameters intentionally remain on
                # CPU, while the optional field may live on an auxiliary CUDA
                # device.  Keep this public rendering/evaluation method valid
                # across that split-device arrangement.
                field_device = next(self.velocity_model.parameters()).device
                velocity = self.velocity_model(self.means.to(field_device), t)
                return velocity.to(self.means.device)
            case ExplicitVelocities():
                return self.velocity_model

    def gate(self):
        return torch.sigmoid(20 * self.marginal_gates)

    def temporal_scale(self) -> Tensor:
        if self.max_duration == float("inf"):
            return torch.exp(self.durations)
        return self.max_duration * TEMPORAL_SCALE_MULTIPLIER * torch.sigmoid(self.durations)

    def marginal_t(self, t: float | Tensor) -> Tensor:
        return torch.exp(-0.5 * ((t - self.times) / self.temporal_scale()) ** 2)

    def temporal_opacity(self, t: float | Tensor):
        gate = self.gate()
        return gate + (1 - gate) * self.marginal_t(t)

    def opacities_t(
        self,
        t: float | Tensor,
    ) -> Tensor:
        return self.opacities.sigmoid() * self.temporal_opacity(t)

    def forward(
        self,
        t: float | Tensor,
        w2c: Tensor,
        intrinsic: Tensor,
        shape: tuple[int, int],
        clamp: bool = True,
        sh_degree: Optional[int] = None,
    ):
        raise RuntimeError(
            "Torch Gaussians.render is removed; use native Vulkan runtime "
            "(ftgspp.native_runtime) for training and rendering"
        )

    def render_oflow(
        self,
        t0: float | Tensor,
        t1: float | Tensor,
        w2c: Tensor,
        intrinsic: Tensor,
        shape: tuple[int, int],
    ):
        raise RuntimeError(
            "Torch Gaussians.render_oflow is removed; use native Vulkan runtime "
            "(ftgspp.native_runtime) for flow rendering"
        )

    def mask(self, mask: Tensor) -> Self:
        match self.velocity_model:
            case VelocityField():
                v = self.velocity_model
            case ExplicitVelocities():
                v = self.velocity_model[mask]

        return self.__class__(
            means=self.means[mask],
            scales=self.scales[mask],
            quats=self.quats[mask],
            opacities=self.opacities[mask],
            sh_0=self.sh_0[mask],
            sh_n=self.sh_n[mask],
            times=self.times[mask],
            durations=self.durations[mask],
            velocity_model=v,
            marginal_gates=self.marginal_gates[mask],
            max_duration=self.max_duration,
        )

    @classmethod
    def empty(cls, sh_degree: int, max_duration: float = float("inf")) -> Self:
        return cls(
            means=torch.empty((0, 3)),
            scales=torch.empty((0, 3)),
            quats=torch.empty((0, 4)),
            opacities=torch.empty((0, 1)),
            sh_0=torch.empty((0, 1, 3)),
            sh_n=torch.empty((0, (sh_degree + 1) ** 2 - 1, 3)),
            times=torch.empty((0, 1)),
            durations=torch.empty((0, 1)),
            velocity_model=torch.empty((0, 3)),
            marginal_gates=torch.empty((0, 1)),
            max_duration=max_duration,
        )

    def __or__(self, other: Self) -> Self:
        if len(self) == 0:
            max_duration = other.max_duration
        elif len(other) == 0:
            max_duration = self.max_duration
        elif self.max_duration != other.max_duration:
            raise ValueError(
                "cannot merge Gaussians with different max_duration "
                f"({self.max_duration} != {other.max_duration})"
            )
        else:
            max_duration = self.max_duration

        match self.velocity_model, other.velocity_model:
            case VelocityField(), VelocityField():
                v = self.velocity_model
            case ExplicitVelocities(), ExplicitVelocities():
                v = torch.cat([self.velocity_model, other.velocity_model])
            case _:
                raise ValueError

        return self.__class__(
            means=torch.cat([self.means, other.means]),
            scales=torch.cat([self.scales, other.scales]),
            quats=torch.cat([self.quats, other.quats]),
            opacities=torch.cat([self.opacities, other.opacities]),
            sh_0=torch.cat([self.sh_0, other.sh_0]),
            sh_n=torch.cat([self.sh_n, other.sh_n]),
            times=torch.cat([self.times, other.times]),
            durations=torch.cat([self.durations, other.durations]),
            velocity_model=v,
            marginal_gates=torch.cat([self.marginal_gates, other.marginal_gates]),
            max_duration=max_duration,
        )

    def __len__(self) -> int:
        return len(self.means)

    def save(self, path: PathLike):
        # Native Vulkan handles are process-local and must never enter a
        # checkpoint.  The persistent parameters have already been synced.
        backend = self.__dict__.pop("_vulkan_backend", None)
        try:
            torch.save(self, path)
        finally:
            if backend is not None:
                self.__dict__["_vulkan_backend"] = backend

    @classmethod
    def load(cls, path: PathLike, map_location: str | torch.device = "cpu") -> Self:
        # Checkpoints are portable across servers even when the optional
        # velocity field was trained on a CUDA auxiliary device.
        return torch.load(path, map_location=map_location, weights_only=False)
