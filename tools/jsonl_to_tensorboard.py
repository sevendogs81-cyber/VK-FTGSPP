#!/usr/bin/env python3
"""Convert FTGSPP train_metrics.jsonl / metrics.jsonl (+ optional eval PNGs) to TensorBoard."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

SKIP_TRAIN_KEYS = {
    "step",
    "rank_view_audit",
    "views",
    "sampling_policy",
    "transport",
    "world_size_lr_scaling",
}

TAG_PREFIX = {
    "backward_seconds": "timing",
    "backward_seconds_imbalance": "timing",
    "backward_seconds_max": "timing",
    "backward_seconds_min": "timing",
    "backward_seconds_rank0": "timing",
    "backward_seconds_rank1": "timing",
    "data_seconds": "timing",
    "optimizer_seconds": "timing",
    "reduction_seconds": "timing",
    "wall_seconds": "timing",
    "wave_backward_download_seconds": "timing",
    "gradient_download_seconds": "exchange",
    "gradient_exchange_overlap": "exchange",
    "gradient_exchange_wall_seconds": "exchange",
    "gradient_host_memcpy_seconds": "exchange",
    "gradient_peer_wait_seconds": "exchange",
    "gradient_peer_wait_seconds_mean": "exchange",
    "gradient_sparse_pack_seconds": "exchange",
    "gradient_transfer_overlap": "exchange",
    "gradient_transfer_wall_seconds": "exchange",
    "gradient_transfer_wall_seconds_mean": "exchange",
    "gradient_upload_seconds": "exchange",
    "gradient_async_pipeline": "exchange",
    "gradient_persistent_host_mapping": "exchange",
    "pair_bucket_count": "pair",
    "pair_local_visible": "pair",
    "pair_payload_bytes": "pair",
    "pair_peer_payload_bytes": "pair",
    "pair_peer_visible": "pair",
    "grad_num_splats": "grad",
    "grad_oc_l2": "grad",
    "grad_oc_rms": "grad",
    "grad_rotation_l2": "grad",
    "grad_rotation_rms": "grad",
    "grad_scaling_l2": "grad",
    "grad_scaling_rms": "grad",
    "grad_sh_dc_l2": "grad",
    "grad_sh_dc_rms": "grad",
    "grad_sh_rest_l2": "grad",
    "grad_sh_rest_rms": "grad",
    "grad_xyz_l2": "grad",
    "grad_xyz_rms": "grad",
    "total_loss": "loss",
    "image_loss": "loss",
    "l1": "loss",
    "ssim": "loss",
    "color_cc": "loss",
    "gate": "loss",
    "reg_opacity": "loss",
    "reg_scale": "loss",
}


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _tag_for(key: str) -> str:
    prefix = TAG_PREFIX.get(key, "train")
    return f"{prefix}/{key}"


class TbWriter:
    def __init__(self, logdir: Path) -> None:
        logdir.mkdir(parents=True, exist_ok=True)
        self._writer = EventFileWriter(str(logdir))

    def add_scalar(self, tag: str, value: float, step: int, wall_time: float | None = None) -> None:
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
        event = Event(wall_time=wall_time or time.time(), step=int(step), summary=summary)
        self._writer.add_event(event)

    def add_image_png(self, tag: str, png_bytes: bytes, step: int, wall_time: float | None = None) -> None:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(png_bytes)) as im:
            w, h = im.size
        image = Summary.Image(
            height=h,
            width=w,
            colorspace=3,
            encoded_image_string=png_bytes,
        )
        summary = Summary(value=[Summary.Value(tag=tag, image=image)])
        event = Event(wall_time=wall_time or time.time(), step=int(step), summary=summary)
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def convert_train(run_dir: Path, writer: TbWriter) -> int:
    path = run_dir / "train_metrics.jsonl"
    if not path.exists():
        print(f"skip train: missing {path}")
        return 0
    n = 0
    for row in _iter_jsonl(path):
        step = int(row.get("step", n))
        for key, value in row.items():
            if key in SKIP_TRAIN_KEYS:
                continue
            if _is_number(value):
                writer.add_scalar(_tag_for(key), float(value), step)
        n += 1
    print(f"train scalars: {n} steps from {path.name}")
    return n


def convert_eval(run_dir: Path, writer: TbWriter) -> int:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        print(f"skip eval: missing {path}")
        return 0
    n = 0
    for row in _iter_jsonl(path):
        step = int(row["step"])
        for key in (
            "psnr",
            "ssim",
            "worst_camera_psnr",
            "worst_camera_ssim",
            "views",
            "views_seen",
            "wall_seconds",
            "camera_count",
            "optimizer_steps",
        ):
            if key in row and _is_number(row[key]):
                writer.add_scalar(f"eval/{key}", float(row[key]), step)
        cams = row.get("camera_metrics") or {}
        for cam_id, metrics in cams.items():
            for mk in ("psnr", "ssim", "views"):
                if mk in metrics and _is_number(metrics[mk]):
                    writer.add_scalar(f"eval/camera_{cam_id}/{mk}", float(metrics[mk]), step)
        n += 1
    print(f"eval scalars: {n} checkpoints from {path.name}")
    return n


def convert_eval_images(run_dir: Path, writer: TbWriter, max_side: int) -> int:
    eval_dir = run_dir / "eval"
    if not eval_dir.is_dir():
        print("skip images: no eval/")
        return 0

    from PIL import Image
    import io

    patterns = ("pred", "gt", "flow")
    by_step: dict[int, dict[str, Path]] = {}
    for path in sorted(eval_dir.glob("*.png")):
        # e.g. 001000_pred.png
        stem = path.stem
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        step_s, kind = parts
        if kind not in patterns or not step_s.isdigit():
            continue
        by_step.setdefault(int(step_s), {})[kind] = path

    n = 0
    for step, files in sorted(by_step.items()):
        for kind, path in files.items():
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                scale = min(1.0, max_side / max(w, h))
                if scale < 1.0:
                    im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                writer.add_image_png(f"eval_images/{kind}", buf.getvalue(), step)
            n += 1
    print(f"eval images: {n} pngs (max_side={max_side})")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="FTGSPP run directory with jsonl logs")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="TensorBoard logdir (default: <run_dir>/tensorboard)",
    )
    parser.add_argument("--no-images", action="store_true", help="Skip eval PNG images")
    parser.add_argument("--max-side", type=int, default=1024, help="Max image side for TB (default 1024)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out = (args.out or (run_dir / "tensorboard")).resolve()
    if out.exists():
        for p in out.glob("events.out.tfevents.*"):
            p.unlink()

    writer = TbWriter(out)
    try:
        convert_train(run_dir, writer)
        convert_eval(run_dir, writer)
        if not args.no_images:
            convert_eval_images(run_dir, writer, args.max_side)
        writer.flush()
    finally:
        writer.close()

    print(f"wrote TensorBoard logs -> {out}")
    print(f"launch: tensorboard --logdir {out} --bind_all --port 6006")


if __name__ == "__main__":
    main()
