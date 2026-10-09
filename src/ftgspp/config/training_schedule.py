"""Input-only task budgets; effective plans contain concrete step counts.

Reference clips have 121 foreground / 60 scene frames. Scale exposure, not camera
count: flat sampling visits each timestamp with probability 1 / frame_count.
This is a starting heuristic, not a claim of equal quality or model capacity.
"""

import math


TRAINING_PRESETS = {
    "foreground": {"preview": 35_000, "balanced": 70_000, "extended": 105_000},
    "scene": {"preview": 15_000, "balanced": 30_000, "extended": 45_000},
}
REFERENCE_FRAMES = {"foreground": 121, "scene": 60}


def training_iterations(mode: str, preset: str, start: int, stop: int) -> int:
    if mode not in TRAINING_PRESETS:
        raise ValueError("pipeline.mode must be scene or foreground")
    if preset not in TRAINING_PRESETS[mode]:
        raise ValueError("train.preset must be preview, balanced, or extended")
    frames = int(stop) - int(start)
    if frames <= 0:
        raise ValueError("data.frames.stop must be greater than data.frames.start")
    reference = TRAINING_PRESETS[mode][preset]
    # A half-reference floor avoids a tiny optimizer horizon for short clips.
    # No hidden upper cap: long clips must opt into a smaller budget explicitly.
    return 1000 * math.ceil(reference * max(0.5, frames / REFERENCE_FRAMES[mode]) / 1000)


def training_step(mode: str, kind: str, iterations: int) -> int:
    steps = int(iterations)
    if steps <= 0:
        raise ValueError("train.iterations must be greater than zero")
    if mode == "scene":
        if kind == "stop":
            return math.ceil(steps * 5 / 6)
        if kind == "every":
            return max(1, round(steps / 300))
        if kind == "interval":
            return max(1, round(steps / 15))
        if kind == "start":
            return min(steps - 1, round(steps / 60))
        if kind == "warmup":
            return min(steps - 1, round(steps / 10))
        if kind == "hard_steps":
            return max(1, round(steps / 10))
    elif mode != "foreground":
        raise ValueError("pipeline.mode must be scene or foreground")
    if kind == "stop":
        return math.ceil(steps / 2)
    if kind == "every":
        return max(1, round(steps / 140))
    if kind == "interval":
        return max(1, round(steps / 35))
    if kind == "start":
        return min(steps - 1, round(steps / 35))
    raise ValueError(f"unknown {mode} schedule field: {kind}")


def hard_relocation_steps(steps: int) -> list[int]:
    """A phase-local midpoint; one-step debug phases have no interior step."""
    return [int(steps) // 2] if int(steps) > 1 else []


def scene_keyframe_stride(start: int, stop: int, fps: float) -> int:
    """Keep roughly 1/6-second geometry spacing; bound very short clips."""
    from ftgspp.config.schema import DEFAULT_FPS

    fps = float(fps)
    if not math.isfinite(fps):
        raise ValueError("data.fps must be finite")
    fps = fps if fps > 0 else DEFAULT_FPS
    return min(max(1, int(stop) - int(start)), max(1, round(fps / 6)))
