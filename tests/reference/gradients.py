"""Numerical packed-gradient layout oracle; not part of the runtime API."""
from typing import Mapping
import numpy as np


class GradientLayoutReference:
    def __init__(self, layout, arena=None):
        self.packed_gradient_layout = lambda: layout
        self.packed_gradient_state = lambda: arena

    @staticmethod
    def _unpack_reordered_sh(
        values: np.ndarray, num_gaussians: int, reorder_size: int,
    ) -> np.ndarray:
        packed = np.asarray(values, dtype=np.float32).reshape(-1, 4)
        canonical = np.empty((num_gaussians, 48), dtype=np.float32)
        for index in range(num_gaussians):
            lane = index % reorder_size
            block = index - lane
            for coefficient4 in range(12):
                packed_index = (
                    12 * block + coefficient4 * reorder_size + lane
                )
                canonical[index, 4 * coefficient4:4 * coefficient4 + 4] = (
                    packed[packed_index]
                )
        return canonical.reshape(num_gaussians, 16, 3)

    @staticmethod
    def _pack_reordered_sh(
        values: np.ndarray, padded_gaussians: int, reorder_size: int,
    ) -> np.ndarray:
        canonical = np.asarray(values, dtype=np.float32)
        if canonical.ndim != 3 or canonical.shape[1:] != (16, 3):
            raise ValueError(
                "canonical SH gradient must have shape (N, 16, 3), got "
                f"{canonical.shape}"
            )
        packed = np.zeros((12 * padded_gaussians, 4), dtype=np.float32)
        flat = canonical.reshape(canonical.shape[0], 48)
        for index in range(canonical.shape[0]):
            lane = index % reorder_size
            block = index - lane
            for coefficient4 in range(12):
                packed_index = (
                    12 * block + coefficient4 * reorder_size + lane
                )
                packed[packed_index] = flat[
                    index, 4 * coefficient4:4 * coefficient4 + 4
                ]
        return packed.reshape(-1)

    @staticmethod
    def _packed_region(
        arena: np.ndarray, layout: Mapping[str, object], name: str,
    ) -> np.ndarray:
        region = layout.get(name)
        if not isinstance(region, (tuple, list)) or len(region) != 2:
            raise RuntimeError(f"packed gradient layout is missing {name}")
        offset, count = (int(value) for value in region)
        return np.asarray(arena[offset:offset + count], dtype=np.float32)

    def canonical_gradient_state(self) -> dict[str, np.ndarray]:
        """Return raw native gradients in framework-neutral parameter shapes."""

        layout = self.packed_gradient_layout()
        arena = np.array(self.packed_gradient_state(), copy=True)
        xyz = self._packed_region(
            arena, layout, "grad_base_xyz_ws",
        )
        if xyz.size % 3:
            raise RuntimeError("packed mean gradient size is not divisible by 3")
        num_gaussians = xyz.size // 3
        scale_opacity = self._packed_region(
            arena, layout, "grad_base_log_scales_opacity",
        ).reshape(num_gaussians, 4)
        time_duration = self._packed_region(
            arena, layout, "grad_times_durations",
        ).reshape(num_gaussians, 2)
        velocity_gate_region = layout.get("grad_velocities_gates")
        if isinstance(velocity_gate_region, (tuple, list)):
            velocity_gate = self._packed_region(
                arena, layout, "grad_velocities_gates",
            ).reshape(num_gaussians, 4)
        else:
            velocity_gate = np.zeros((num_gaussians, 4), dtype=np.float32)
        sh_native = self._packed_region(
            arena, layout, "grad_sh_coeffs_reordered",
        )
        if sh_native.size % 48:
            raise RuntimeError("packed SH gradient size is not divisible by 48")
        padded_gaussians = sh_native.size // 48
        reorder_size = int(layout.get("sh_reorder_size", 16))
        rotations = self._packed_region(
            arena, layout, "grad_rotations",
        ).reshape(num_gaussians, 4)
        return {
            "means": xyz.reshape(num_gaussians, 3).copy(),
            "log_scales": scale_opacity[:, :3].copy(),
            "opacity_logits": scale_opacity[:, 3:4].copy(),
            "times": time_duration[:, :1].copy(),
            "duration_raw": time_duration[:, 1:2].copy(),
            "velocities": velocity_gate[:, :3].copy(),
            "gate_raw": velocity_gate[:, 3:4].copy(),
            "sh_coefficients": self._unpack_reordered_sh(
                sh_native, num_gaussians, reorder_size,
            ),
            "quaternions": rotations.copy(),
            "_packed_padded_gaussians": np.asarray(
                [padded_gaussians], dtype=np.int64,
            ),
        }

    def canonical_gradient_to_packed(
        self,
        values: Mapping[str, np.ndarray],
        *,
        base: np.ndarray | None = None,
    ) -> np.ndarray:
        """Pack canonical Gaussian gradients for a controlled Adam reference."""

        layout = self.packed_gradient_layout()
        total = int(layout["total_floats"])
        arena = (
            np.zeros(total, dtype=np.float32)
            if base is None
            else np.array(base, dtype=np.float32, copy=True)
        )
        if arena.shape != (total,):
            raise ValueError(
                f"packed gradient base must have shape {(total,)}, got {arena.shape}"
            )
        means = np.asarray(values["means"], dtype=np.float32)
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError(
                f"canonical mean gradient must have shape (N, 3), got {means.shape}"
            )
        num_gaussians = means.shape[0]

        def write(name: str, value: np.ndarray) -> None:
            region = layout.get(name)
            if not isinstance(region, (tuple, list)) or len(region) != 2:
                raise RuntimeError(f"packed gradient layout is missing {name}")
            offset, count = (int(item) for item in region)
            flat = np.asarray(value, dtype=np.float32).reshape(-1)
            if flat.size != count:
                raise ValueError(
                    f"{name} expects {count} floats, got {flat.size}"
                )
            arena[offset:offset + count] = flat

        write("grad_base_xyz_ws", means)
        write("grad_base_log_scales_opacity", np.concatenate((
            np.asarray(values["log_scales"], dtype=np.float32),
            np.asarray(values["opacity_logits"], dtype=np.float32),
        ), axis=1))
        write("grad_times_durations", np.concatenate((
            np.asarray(values["times"], dtype=np.float32),
            np.asarray(values["duration_raw"], dtype=np.float32),
        ), axis=1))
        if "grad_velocities_gates" in layout:
            write("grad_velocities_gates", np.concatenate((
                np.asarray(values["velocities"], dtype=np.float32),
                np.asarray(values["gate_raw"], dtype=np.float32),
            ), axis=1))
        sh_region = layout["grad_sh_coeffs_reordered"]
        sh_count = int(sh_region[1])
        padded_gaussians = sh_count // 48
        write("grad_sh_coeffs_reordered", self._pack_reordered_sh(
            np.asarray(values["sh_coefficients"], dtype=np.float32),
            padded_gaussians,
            int(layout.get("sh_reorder_size", 16)),
        ))
        write(
            "grad_rotations",
            np.asarray(values["quaternions"], dtype=np.float32),
        )
        return arena
