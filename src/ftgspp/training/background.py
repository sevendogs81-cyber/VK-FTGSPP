"""Deterministic solid-background contexts for the composite image objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


BACKGROUND_SAMPLER_VERSION = "splitmix64-rgb24-v1"
BACKGROUND_DOMAIN_TAG = 0x4654475350504247  # ASCII-ish "FTGSPPBG"
_MASK64 = (1 << 64) - 1
_PHASE_IDS = {"main": 0x6D61696E, "hard_separation": 0x68617264}


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _unit_float24(word: int) -> np.float32:
    return np.float32((word >> 40) & 0xFFFFFF) * np.float32(1.0 / 16777216.0)


@dataclass(frozen=True)
class BackgroundContext:
    color: tuple[float, float, float]
    version: str
    phase: str
    logical_step: int
    rank_policy: str
    rank: int | None

    def array(self) -> np.ndarray:
        return np.asarray(self.color, dtype=np.float32)


def sample_background(
    options,
    *,
    phase: str,
    logical_step: int,
    rank: int = 0,
) -> BackgroundContext:
    """Resolve one immutable color without consuming any process RNG state."""

    if phase not in _PHASE_IDS:
        raise ValueError(f"unknown background phase: {phase!r}")
    if logical_step < 0:
        raise ValueError("background logical_step must be nonnegative")
    rank_policy = str(getattr(options, "rank_policy", "shared"))
    hard_policy = str(getattr(options, "hard_phase", "inherit"))
    configured = tuple(float(value) for value in getattr(options, "color", (0, 0, 0)))
    mode = str(getattr(options, "mode", "fixed"))
    if phase == "hard_separation" and hard_policy == "fixed_black":
        mode = "fixed"
        configured = (0.0, 0.0, 0.0)
    sampled_rank = int(rank) if rank_policy == "independent" else None
    if mode == "fixed":
        color = configured
    elif mode == "random_solid":
        seed = int(getattr(options, "seed", 0)) & _MASK64
        key = seed ^ BACKGROUND_DOMAIN_TAG
        key ^= (_PHASE_IDS[phase] * 0xD6E8FEB86659FD93) & _MASK64
        key ^= (int(logical_step) * 0xA0761D6478BD642F) & _MASK64
        if sampled_rank is not None:
            key ^= (sampled_rank * 0xE7037ED1A0B428DB) & _MASK64
        color = tuple(
            float(_unit_float24(_splitmix64(key ^ lane)))
            for lane in (
                0x243F6A8885A308D3,
                0x13198A2E03707344,
                0xA4093822299F31D0,
            )
        )
    else:
        raise ValueError(f"unsupported background mode: {mode!r}")
    return BackgroundContext(
        color=(float(color[0]), float(color[1]), float(color[2])),
        version=BACKGROUND_SAMPLER_VERSION,
        phase=phase,
        logical_step=int(logical_step),
        rank_policy=rank_policy,
        rank=sampled_rank,
    )


def sample_rank_backgrounds(
    options, *, phase: str, logical_step: int, world_size: int,
) -> tuple[BackgroundContext, ...]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return tuple(
        sample_background(
            options, phase=phase, logical_step=logical_step, rank=rank,
        )
        for rank in range(world_size)
    )


def background_metadata(options) -> dict[str, object]:
    mode = str(getattr(options, "mode", "fixed"))
    return {
        "sampler_version": BACKGROUND_SAMPLER_VERSION,
        "domain_tag": f"0x{BACKGROUND_DOMAIN_TAG:016x}",
        "mode": mode,
        "color": [float(v) for v in getattr(options, "color", (0, 0, 0))],
        "seed": int(getattr(options, "seed", 0)),
        "rank_policy": str(getattr(options, "rank_policy", "shared")),
        "hard_phase": str(getattr(options, "hard_phase", "inherit")),
        "key": "seed/domain/phase/phase-local-logical-step[/rank]",
        "float_mapping": "top-24-bits-times-2^-24",
    }


def ensure_color(value: Sequence[float]) -> tuple[float, float, float]:
    color = tuple(float(item) for item in value)
    if len(color) != 3 or not np.isfinite(color).all():
        raise ValueError("background color must contain three finite values")
    return color
