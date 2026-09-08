# FreeTimeGS++ checkpoint compatibility model. Not used by native training.
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from __future__ import annotations

from typing import Callable, Sequence, TypeAlias

import torch
from torch import Tensor, nn

ExplicitVelocities: TypeAlias = nn.Parameter


class HashGridEncoding(nn.Module):
    """Packed HashGrid matching tiny-cuda-nn FTGSPP configuration.

    Layout: 16 levels × 2 features, log2_hashmap_size=19, base_resolution=16,
    per_level_scale=2. Levels whose ``resolution**3 <= hashmap`` use dense
    indexing; larger levels use the tcnn spatial hash. Parameter vector length
    is 14_229_504 (not the naive 16×2^19×2 full table).
    """

    def __init__(
        self,
        *,
        levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 2.0,
    ):
        super().__init__()
        self.levels = levels
        self.features_per_level = features_per_level
        self.hashmap_size = 1 << log2_hashmap_size
        resolutions = [int(round(base_resolution * per_level_scale**i)) for i in range(levels)]
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.long))
        offsets = []
        cursor = 0
        for resolution in resolutions:
            offsets.append(cursor)
            cursor += min(self.hashmap_size, resolution**3)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.n_params = cursor * features_per_level
        self.table = nn.Parameter(torch.empty(self.n_params))
        nn.init.uniform_(self.table, -1e-4, 1e-4)

    @staticmethod
    def _hash(coords: Tensor) -> Tensor:
        # tiny-cuda-nn primes
        x, y, z = coords.unbind(-1)
        return (x * 1) ^ (y * 2654435761) ^ (z * 805459861)

    def forward(self, xyz: Tensor) -> Tensor:
        original_shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3).clamp(0.0, 1.0)
        encoded = []
        for level, resolution in enumerate(self.resolutions.tolist()):
            scale = float(resolution - 1)
            # tcnn pos_fract: fmaf(scale, input, 0.5)
            scaled = xyz * scale + 0.5
            base = torch.floor(scaled).long()
            fraction = scaled - base.float()
            value = torch.zeros(
                (len(xyz), self.features_per_level),
                dtype=self.table.dtype,
                device=xyz.device,
            )
            hashmap_size = min(self.hashmap_size, resolution**3)
            offset0 = int(self.offsets[level])
            for corner in range(8):
                corner_offset = torch.tensor(
                    [(corner >> 0) & 1, (corner >> 1) & 1, (corner >> 2) & 1],
                    device=xyz.device,
                    dtype=torch.long,
                )
                coords = base + corner_offset
                weight_axis = torch.where(corner_offset.bool(), fraction, 1.0 - fraction)
                weight = weight_axis.prod(dim=-1, keepdim=True)
                local = coords[:, 0] + resolution * (coords[:, 1] + resolution * coords[:, 2])
                if hashmap_size < resolution**3:
                    local = self._hash(coords)
                local = torch.remainder(local, hashmap_size)
                index = (offset0 + local) * self.features_per_level
                feats = torch.stack(
                    [self.table[index], self.table[index + 1]], dim=-1,
                )
                value = value + weight * feats
            encoded.append(value)
        result = torch.cat(encoded, dim=-1)
        return result.reshape(*original_shape, self.levels * self.features_per_level)


class Bounds(nn.Module):
    """N-Dimensional bounds, used for normalizing coordinates"""

    _bounds: nn.Parameter  # (dim, 2) in min, max order

    def __init__(self, bounds: Sequence[float] | Sequence[Sequence[float]] | Tensor):
        super().__init__()
        if not isinstance(bounds, Tensor):
            bounds = torch.tensor(bounds)
        assert bounds.shape[-1] == 2 and bounds.dim() <= 2
        self._bounds = nn.Parameter(bounds.reshape(-1, 2), False)

    def dim(self) -> int:
        return len(self._bounds)

    def normalize(
        self,
        input: Tensor,  # (..., dim)
        range: tuple[float, float] = (0, 1),
    ):
        assert input.shape[-1] == self.dim()
        min, max = range

        scale = (self._bounds[:, 1] - self._bounds[:, 0]).clamp_min(1e-8)
        shift = self._bounds[:, 0]
        z = (input - shift) / scale

        return (max - min) * z + min


class MLP(nn.Module):
    """Basic MLP"""

    layers: nn.ModuleList
    activation: Callable[[Tensor], Tensor]

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: int,
        hidden_dims: int,
        hidden_layers: int,
        activation: Callable[[Tensor], Tensor],
    ):
        super().__init__()

        layers = [
            nn.Linear(input_dims, hidden_dims),
            *[nn.Linear(hidden_dims, hidden_dims) for _ in range(hidden_layers - 1)],
            nn.Linear(hidden_dims, output_dims),
        ]
        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, input: Tensor):
        z = input
        for linear in self.layers[:-1]:
            z = linear(z)
            z = self.activation(z)
        z = self.layers[-1](z)

        return z


class HyperMLP(nn.Module):
    """MLP with biases modulated with hypernetworks"""

    layers: nn.ModuleList
    bias_nets: nn.ModuleList

    activation: Callable[[Tensor], Tensor]
    hidden_dims: int
    hidden_layers: int

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: int,
        hidden_dims: int,
        hidden_layers: int,
        activation: Callable[[Tensor], Tensor],
        bias_net_input_dims: int,
        bias_net_hidden_dims: int,
        bias_net_hidden_layers: int,
        bias_net_activation: Callable[[Tensor], Tensor],
    ):
        super().__init__()

        layers = [
            nn.Linear(input_dims, hidden_dims, bias=False),
            *[
                nn.Linear(hidden_dims, hidden_dims, bias=False)
                for _ in range(hidden_layers - 1)
            ],
            nn.Linear(hidden_dims, output_dims, bias=False),
        ]
        self.layers = nn.ModuleList(layers)
        self.activation = activation

        bias_nets = [
            MLP(
                input_dims=bias_net_input_dims,
                output_dims=hidden_dims,
                hidden_dims=bias_net_hidden_dims,
                hidden_layers=bias_net_hidden_layers,
                activation=bias_net_activation,
            )
            for _ in range(hidden_layers)
        ]
        bias_nets.append(
            MLP(
                input_dims=bias_net_input_dims,
                output_dims=output_dims,
                hidden_dims=bias_net_hidden_dims,
                hidden_layers=bias_net_hidden_layers,
                activation=bias_net_activation,
            )
        )
        self.bias_nets = nn.ModuleList(bias_nets)

        self.hidden_dims = hidden_dims
        self.hidden_layers = hidden_layers

    def forward(self, input: Tensor, z_bias: Tensor):
        z = input.float()
        for i, linear in enumerate(self.layers[:-1]):
            z = linear(z) + self.bias_nets[i](z_bias)
            z = self.activation(z)

        z = self.layers[-1](z) + self.bias_nets[-1](z_bias)

        return z


class VelocityField(nn.Module):
    enc_xyz: HashGridEncoding
    hyper_mlp: nn.Module

    bounds_xyz: Bounds
    bounds_t: Bounds

    @torch.no_grad()
    def __init__(self, bounds_xyz: Bounds, bounds_t: Bounds):
        super().__init__()

        self.bounds_xyz = bounds_xyz
        self.bounds_t = bounds_t

        self.enc_xyz = HashGridEncoding()
        self.hyper_mlp = HyperMLP(
            input_dims=32,
            output_dims=3,
            hidden_dims=128,
            hidden_layers=3,
            activation=nn.functional.relu,
            bias_net_input_dims=1,
            bias_net_hidden_dims=128,
            bias_net_hidden_layers=3,
            bias_net_activation=nn.functional.relu,
        )

    def forward(self, xyz: Tensor, t: float | Tensor):
        if not isinstance(t, Tensor):
            t = torch.tensor(t, device=xyz.device, dtype=xyz.dtype)
        t = t.to(device=xyz.device, dtype=xyz.dtype).reshape(-1, 1)
        if len(t) == 1 and len(xyz) != 1:
            t = t.expand(len(xyz), 1)

        xyz = self.bounds_xyz.normalize(xyz, (0, 1))
        t = self.bounds_t.normalize(t, (0, 1))
        z_xyz = self.enc_xyz(xyz).float()
        z_t = t.float()

        v = self.hyper_mlp(z_xyz, z_t)

        return v
