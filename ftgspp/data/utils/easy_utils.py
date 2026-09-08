# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ftgspp.utils import PathLike


class FileStorage:
    def __init__(self, filename: PathLike):
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        self.fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)

    def close(self):
        cv2.FileStorage.release(self.fs)

    def read(self, key: str, dt: str = "mat"):
        if dt == "mat":
            return self.fs.getNode(key).mat()
        if dt == "list":
            values = []
            node = self.fs.getNode(key)
            for i in range(node.size()):
                value = node.at(i).string()
                if value == "":
                    value = str(int(node.at(i).real()))
                if value != "none":
                    values.append(value)
            return values
        if dt == "real":
            return self.fs.getNode(key).real()
        raise NotImplementedError(dt)


@dataclass
class EasyCamera:
    name: str
    K: np.ndarray
    D: np.ndarray
    H: int
    W: int
    R: np.ndarray
    T: np.ndarray
    Rvec: np.ndarray

    @property
    def w2c(self) -> np.ndarray:
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = self.R
        out[:3, 3] = self.T.reshape(3)
        return out


def _read_distortion(intri: FileStorage, cam: str) -> np.ndarray:
    D = intri.read(f"D_{cam}")
    if D is None:
        D = intri.read(f"dist_{cam}")
    if D is None:
        D = np.zeros((5, 1), dtype=np.float64)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    if len(D) < 5:
        D = np.pad(D, (0, 5 - len(D)))
    return D[:5].reshape(5, 1)


def read_camera(intri_path: PathLike, extri_path: Optional[PathLike] = None) -> dict[str, EasyCamera]:
    if extri_path is None:
        root = Path(intri_path)
        intri_path = root / "intri.yml"
        extri_path = root / "extri.yml"

    intri = FileStorage(intri_path)
    extri = FileStorage(extri_path)
    try:
        cam_names = intri.read("names", dt="list")
        cameras: dict[str, EasyCamera] = {}

        for cam in cam_names:
            K = np.asarray(intri.read(f"K_{cam}"), dtype=np.float64)
            H = int(intri.read(f"H_{cam}", dt="real")) or -1
            W = int(intri.read(f"W_{cam}", dt="real")) or -1

            T = np.asarray(extri.read(f"T_{cam}"), dtype=np.float64).reshape(3, 1)
            Rvec = extri.read(f"R_{cam}")
            R = extri.read(f"Rot_{cam}")
            if R is not None:
                R = np.asarray(R, dtype=np.float64)
                Rvec = cv2.Rodrigues(R)[0]
            elif Rvec is not None:
                Rvec = np.asarray(Rvec, dtype=np.float64)
                R = cv2.Rodrigues(Rvec)[0]
            else:
                raise ValueError(f"Either R_{cam} or Rot_{cam} must be provided")

            cameras[cam] = EasyCamera(
                name=cam,
                K=K,
                D=_read_distortion(intri, cam),
                H=H,
                W=W,
                R=R,
                T=T,
                Rvec=Rvec,
            )

        return cameras
    finally:
        intri.close()
        extri.close()
