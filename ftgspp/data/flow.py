# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import argparse
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ftgspp.config import Config
from ftgspp.data.utils import load_multiview_dataset
from ftgspp.data.flow_names import keyframe_indices, to_flow_cam_filename, to_flow_pair_dir


def _parse_cameras(spec: str, *, max_camera: int) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(max_camera))

    cams: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                raise ValueError(f"invalid camera range: {token}")
            cams.update(range(lo, hi + 1))
        else:
            cams.add(int(token))

    out = sorted(cams)
    if not out:
        raise ValueError("no cameras selected")
    if out[0] < 0 or out[-1] >= max_camera:
        raise ValueError(f"camera index out of range [0, {max_camera - 1}]")
    return out


def _to_numpy(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _as_hw2_flow(flow: Any, *, h: int, w: int) -> np.ndarray:
    flow = _to_numpy(flow)
    if flow.ndim == 4 and flow.shape[0] == 1:
        flow = flow[0]
    if flow.ndim != 3:
        raise ValueError(f"unexpected flow ndim={flow.ndim}, shape={flow.shape}")

    if flow.shape == (h, w, 2):
        return flow.astype(np.float32, copy=False)
    if flow.shape == (2, h, w):
        return np.moveaxis(flow, 0, -1).astype(np.float32, copy=False)
    if flow.shape == (h, 2, w):
        return np.moveaxis(flow, 1, -1).astype(np.float32, copy=False)
    raise ValueError(f"unexpected flow shape={flow.shape}, expected (H,W,2) or (2,H,W)")


def _as_hw_covis(covis: Any, *, h: int, w: int) -> np.ndarray:
    covis = _to_numpy(covis)
    if covis.ndim == 4 and covis.shape[0] == 1:
        covis = covis[0]
    if covis.ndim == 3 and covis.shape[0] == 1:
        covis = covis[0]
    if covis.ndim == 3 and covis.shape[-1] == 1:
        covis = covis[..., 0]
    if covis.ndim != 2:
        raise ValueError(f"unexpected covis ndim={covis.ndim}, shape={covis.shape}")
    if covis.shape != (h, w):
        raise ValueError(f"unexpected covis shape={covis.shape}, expected {(h, w)}")
    return covis.astype(np.float32, copy=False)


class UFMRunner:
    def __init__(self, *, device: str, model_name: str):
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[ufm] CUDA is not available; falling back to CPU.")
            device = "cpu"
        self._device = torch.device(device)

        load_errors: list[Exception] = []
        model: Any | None = None
        try:
            from ufm.models import UFM

            model = UFM.from_pretrained(model_name)
        except Exception as e:
            load_errors.append(e)
            try:
                from uniflowmatch.models.ufm import UniFlowMatchConfidence

                model = UniFlowMatchConfidence.from_pretrained(model_name)
            except Exception as e2:
                load_errors.append(e2)

        if model is None:
            details = " | ".join(f"{type(e).__name__}: {e}" for e in load_errors)
            raise ModuleNotFoundError(
                "UFM package not found or failed to initialize. "
                "Install UFM in this environment first. "
                f"(details: {details})"
            )

        self._model = model
        if hasattr(self._model, "to"):
            self._model = self._model.to(self._device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    @staticmethod
    def _extract_flow_and_covis(preds: Any, *, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            flow = preds.flow.flow_output[0]
            covis = preds.covisibility.mask[0]
        except Exception as e:
            raise ValueError(
                "Cannot parse UFM outputs. Expected attrs: "
                "`preds.flow.flow_output` and `preds.covisibility.mask`."
            ) from e

        flow = _as_hw2_flow(flow, h=h, w=w)
        covis = _as_hw_covis(covis, h=h, w=w)
        return flow, covis

    def _predict_signature_kwargs(self, *, h: int, w: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(self._model.predict_correspondences_batched)
            params = sig.parameters
            if "output_resolution" in params:
                kwargs["output_resolution"] = (h, w)
            if "query_points" in params:
                kwargs["query_points"] = None
        except Exception:
            pass
        return kwargs

    def predict(self, img_a: Image.Image, img_b: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        src = np.asarray(img_a).copy()
        dst = np.asarray(img_b).copy()
        src_t = torch.from_numpy(src).to(self._device, non_blocking=True)
        dst_t = torch.from_numpy(dst).to(self._device, non_blocking=True)

        kwargs = self._predict_signature_kwargs(h=img_a.height, w=img_a.width)
        with torch.inference_mode():
            try:
                preds = self._model.predict_correspondences_batched(
                    source_image=src_t,
                    target_image=dst_t,
                    **kwargs,
                )
            except Exception:
                try:
                    preds = self._model.predict_correspondences_batched(src_t, dst_t, **kwargs)
                except Exception:
                    preds = self._model.predict_correspondences_batched([img_a], [img_b], **kwargs)
        return self._extract_flow_and_covis(preds, h=img_a.height, w=img_a.width)


def precompute_ufm_flow(
    *,
    config_path: Path,
    out: Path | None,
    model: str,
    device: str,
    cameras: str,
    keyframe_stride: int | None,
    max_pairs: int | None,
    overwrite: bool,
    bidirectional: bool,
) -> None:
    config = Config.load(config_path)
    dataset = load_multiview_dataset(config)
    dataset = dataset.at(config.data.frames.into_slice())

    out_root = out if out is not None else config.init.temporal_flow_path
    out_root = out_root.expanduser().resolve()

    kfs = keyframe_indices(dataset.num_frames, keyframe_stride or config.init.keyframe_stride)
    pairs = list(zip(kfs[:-1], kfs[1:]))
    if bidirectional:
        pairs += [(b, a) for (a, b) in pairs]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if len(pairs) == 0:
        raise ValueError("no frame pairs to process")

    selected_cams = _parse_cameras(cameras, max_camera=dataset.num_cameras)
    runner = UFMRunner(device=device, model_name=model)

    out_root.mkdir(parents=True, exist_ok=True)
    total = len(pairs) * len(selected_cams)
    num_done = 0
    num_skip = 0

    with tqdm(total=total, desc="UFM flow precompute") as bar:
        for idx_0, idx_1 in pairs:
            frame_0 = int(dataset[idx_0, 0].frame)  # type: ignore
            frame_1 = int(dataset[idx_1, 0].frame)  # type: ignore
            pair_dir = out_root / to_flow_pair_dir(frame_0, frame_1)
            pair_dir.mkdir(parents=True, exist_ok=True)

            for cam in selected_cams:
                out_path = pair_dir / to_flow_cam_filename(cam)
                if out_path.exists() and not overwrite:
                    num_skip += 1
                    bar.update(1)
                    continue

                data_0 = dataset[idx_0, cam].load()
                data_1 = dataset[idx_1, cam].load()
                img_0 = Image.fromarray(data_0.rgb.cpu().numpy(), mode="RGB")  # type: ignore
                img_1 = Image.fromarray(data_1.rgb.cpu().numpy(), mode="RGB")  # type: ignore
                flow, covis = runner.predict(img_0, img_1)

                np.savez_compressed(
                    out_path,
                    flow=flow,
                    covis=covis,
                    frame_0=np.int32(frame_0),
                    frame_1=np.int32(frame_1),
                    camera=np.int16(cam),
                    height=np.int32(img_0.height),
                    width=np.int32(img_0.width),
                )
                num_done += 1
                bar.update(1)

    print(
        f"[ufm] saved={num_done} skipped={num_skip} pairs={len(pairs)} "
        f"cameras={len(selected_cams)} out={out_root}"
    )


def main(args: argparse.Namespace):
    precompute_ufm_flow(
        config_path=args.config,
        out=args.out,
        model=args.model,
        device=args.device,
        cameras=args.cameras,
        keyframe_stride=args.keyframe_stride,
        max_pairs=args.max_pairs,
        overwrite=args.overwrite,
        bidirectional=args.bidirectional,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", type=str, default="infinity1096/UFM-Base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cameras", type=str, default="all")
    parser.add_argument("--keyframe-stride", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)
