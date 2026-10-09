"""Generate a data-free installation check, not a quality benchmark.

Run with the core environment's Python. The output contains a 32x32, two-frame,
three-camera synthetic target, calibration, keyframe points and a short config.
No model downloads, GPU access or files from dev/ are needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import cv2
import numpy as np
from omegaconf import OmegaConf


def make_scene(output: Path) -> Path:
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ftgspp-tiny-", dir=output.parent) as temporary:
        root = Path(temporary) / "capture"
        root.mkdir()
        for kind in ("intri", "extri"):
            storage = cv2.FileStorage(str(root / f"{kind}.yml"), cv2.FILE_STORAGE_WRITE)
            try:
                storage.write("names", ["0", "1", "2"])
                for camera in range(3):
                    if kind == "intri":
                        storage.write(f"K_{camera}", np.array([[30., 0, 16], [0, 30., 16], [0, 0, 1]]))
                        storage.write(f"D_{camera}", np.zeros((5, 1)))
                        storage.write(f"H_{camera}", 32)
                        storage.write(f"W_{camera}", 32)
                    else:
                        storage.write(f"Rot_{camera}", np.eye(3))
                        storage.write(f"T_{camera}", np.array([[0.1 * camera], [0.], [0.]]))
            finally:
                storage.release()
        yy, xx = np.mgrid[:32, :32]
        for camera in range(3):
            directory = root / "images" / str(camera)
            directory.mkdir(parents=True)
            for frame in range(2):
                alpha = ((xx - (16 + frame + camera)) ** 2 + (yy - 16) ** 2 < 6 ** 2)
                bgra = np.zeros((32, 32, 4), np.uint8)
                bgra[alpha] = [120, 170, 210, 255]
                if not cv2.imwrite(str(directory / f"{frame:06d}.png"), bgra):
                    raise OSError("failed to write synthetic image")
        points = root / "points_stride1"
        points.mkdir()
        rng = np.random.default_rng(5)
        angle = rng.uniform(0, 2 * np.pi, 32)
        radius = 0.5 * np.sqrt(rng.uniform(0, 1, 32))
        xyz = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.full(32, 3.)))
        header = (
            "ply\nformat ascii 1.0\nelement vertex 32\nproperty float x\n"
            "property float y\nproperty float z\nproperty uchar red\n"
            "property uchar green\nproperty uchar blue\nend_header\n"
        )
        for frame in range(2):
            shifted = xyz + np.array([frame * 0.1, 0, 0])
            (points / f"f{frame:06d}.ply").write_text(
                header + "".join(f"{x} {y} {z} 210 170 120\n" for x, y, z in shifted),
            )
        config = {
            "pipeline": {"mode": "foreground"},
            "data": {"name": "tiny", "root": ".", "fps": 2.0, "scale": 1.0,
                     "frames": {"start": 0, "stop": 2},
                     "train_cameras": [0, 1], "eval_cameras": [2],
                     "masks": {"source": "alpha"}},
            "init": {"backend": "points", "points_path": "points_stride1",
                     "seed": 0, "num_gaussians": 64, "keyframe_stride": 1},
            "train": {"iterations": 100, "checkpoint_interval": 50,
                      "log_interval": 25, "hard_separation": False,
                      "color_correction": False,
                      "relocation": {"start": 1, "stop": 100, "every": 10}},
            "eval": {"frame_interval": 1},
            "run": {"output_path": "run", "view_load_workers": 0},
        }
        OmegaConf.save(OmegaConf.create(config), root / "ftgspp.yaml")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output appeared during generation: {output}")
        root.rename(output)
    return output / "ftgspp.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = make_scene(args.output)
    print(f"Created synthetic workflow fixture: {config}")
    print("Use the README commands to train and render. This is not a quality benchmark.")


if __name__ == "__main__":
    main()
