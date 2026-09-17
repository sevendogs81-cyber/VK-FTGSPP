"""Stage wall-clock accounting that does not treat GPU sync as a hidden default."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class StageTimer:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        begin = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - begin)

    def report(self) -> dict[str, object]:
        return {
            "wall_seconds": float(time.perf_counter() - self.started),
            "stages_seconds": {name: float(value) for name, value in self.stages.items()},
        }
