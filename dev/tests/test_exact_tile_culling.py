from __future__ import annotations

import math
import unittest

import numpy as np


TILE = 16
WIDTH = 64
HEIGHT = 64


def _rectangle(center: np.ndarray, inverse: np.ndarray) -> tuple[int, int, int, int]:
    covariance = np.linalg.inv(inverse)
    radii = np.sqrt(np.diag(covariance))
    return (
        max(0, math.floor((center[0] - radii[0]) / TILE)),
        max(0, math.floor((center[1] - radii[1]) / TILE)),
        min(WIDTH // TILE, math.floor((center[0] + radii[0]) / TILE) + 1),
        min(HEIGHT // TILE, math.floor((center[1] + radii[1]) / TILE) + 1),
    )


def _range_bound(
    inverse: np.ndarray, lower: float, upper: float,
) -> tuple[float, float] | None:
    a, b, c = inverse[0, 0], inverse[0, 1], inverse[1, 1]
    determinant = a * c - b * b
    if not (a > 0 and c > 0 and determinant > 0):
        return None
    radius = math.sqrt(a / determinant)
    lower = max(lower, -radius)
    upper = min(upper, radius)
    if lower > upper:
        return None
    extreme = -b / c * math.sqrt(c / determinant)
    v0 = np.clip(-extreme, lower, upper)
    v1 = np.clip(extreme, lower, upper)
    bv0, bv1 = -b * v0, -b * v1
    d0 = max(bv0 * bv0 - a * (c * v0 * v0 - 1), 0)
    d1 = max(bv1 * bv1 - a * (c * v1 * v1 - 1), 0)
    return (
        (bv0 - math.sqrt(d0)) / a - 1e-4,
        (bv1 + math.sqrt(d1)) / a + 1e-4,
    )


def _emitted_tiles(center: np.ndarray, inverse: np.ndarray) -> set[tuple[int, int]]:
    min_x, min_y, max_x, max_y = _rectangle(center, inverse)
    scan_rows = max_y - min_y <= max_x - min_x
    emitted: set[tuple[int, int]] = set()
    lines = range(min_y, max_y) if scan_rows else range(min_x, max_x)
    for line in lines:
        oriented_center = center if scan_rows else center[::-1]
        oriented_inverse = inverse if scan_rows else inverse[::-1, ::-1]
        bounds = _range_bound(
            oriented_inverse,
            line * TILE - oriented_center[1],
            (line + 1) * TILE - oriented_center[1],
        )
        if bounds is None:
            continue
        range_min, range_max = (
            (min_x, max_x) if scan_rows else (min_y, max_y)
        )
        begin = min(max(math.floor((bounds[0] + oriented_center[0]) / TILE), range_min), range_max)
        end = min(max(math.floor((bounds[1] + oriented_center[0]) / TILE) + 1, range_min), range_max)
        for cross in range(begin, max(begin, end)):
            emitted.add((cross, line) if scan_rows else (line, cross))
    return emitted


class ExactTileCullingTests(unittest.TestCase):
    def _assert_no_pixel_false_negative(
        self, center: tuple[float, float], inverse: np.ndarray,
    ) -> None:
        center_array = np.asarray(center, np.float64)
        emitted = _emitted_tiles(center_array, inverse)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                delta = np.asarray([x, y], np.float64) - center_array
                if float(delta @ inverse @ delta) <= 1.0 + 1e-12:
                    self.assertIn((x // TILE, y // TILE), emitted)

    def test_edges_tangencies_and_thin_rotated_ellipses_have_no_false_negative(self):
        cases = [
            ((0.0, 8.0), np.diag([1 / 0.05**2, 1 / 20.0**2])),
            ((16.0, 16.0), np.diag([1 / 16.0**2, 1 / 3.0**2])),
            ((63.0, 40.0), np.diag([1 / 5.0**2, 1 / 18.0**2])),
        ]
        angle = math.radians(37)
        rotation = np.asarray(
            [[math.cos(angle), -math.sin(angle)],
             [math.sin(angle), math.cos(angle)]],
            np.float64,
        )
        cases.append((
            (31.5, 31.5),
            rotation @ np.diag([1 / 0.08**2, 1 / 28.0**2]) @ rotation.T,
        ))
        for center, inverse in cases:
            with self.subTest(center=center):
                self._assert_no_pixel_false_negative(center, inverse)

    def test_random_spd_ellipses_have_no_pixel_false_negative(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            angle = rng.uniform(-math.pi, math.pi)
            rotation = np.asarray(
                [[math.cos(angle), -math.sin(angle)],
                 [math.sin(angle), math.cos(angle)]],
                np.float64,
            )
            radii = np.exp(rng.uniform(math.log(0.05), math.log(30.0), 2))
            inverse = rotation @ np.diag(1.0 / (radii * radii)) @ rotation.T
            center = tuple(rng.uniform(-8.0, 72.0, 2))
            self._assert_no_pixel_false_negative(center, inverse)


if __name__ == "__main__":
    unittest.main()
