"""Small OS helpers shared by native orchestration (no Torch)."""

from __future__ import annotations

import os

type PathLike = str | os.PathLike


def num_workers() -> int:
    return len(os.sched_getaffinity(0))
