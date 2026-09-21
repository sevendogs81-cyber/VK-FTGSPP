"""Load export_selection.json and restore the selected color-correction snapshot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ftgspp.io.hashing import file_sha256


def export_selection_path(run: Path) -> Path:
    return Path(run) / "export_selection.json"


def load_export_selection(run: Path) -> dict[str, Any]:
    path = export_selection_path(run)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("export_selection.json must contain a JSON object")
    return payload


def restore_exported_color_correction(
    backend: Any,
    run: Path,
    selection: Mapping[str, Any],
    dataset: Any,
) -> str | None:
    raw = selection.get("color_correction")
    if not isinstance(raw, dict):
        return None
    artifact = Path(run) / str(raw["artifact"])
    digest = file_sha256(artifact)
    declared = raw.get("sha256")
    if declared is not None and str(declared) != digest:
        raise ValueError("color correction snapshot hash mismatch")
    with np.load(artifact, allow_pickle=False) as archive:
        color_step = int(np.asarray(archive["snapshot_step"]))
        params = np.asarray(archive["params"], np.float32)
        camera_ids = np.asarray(archive["camera_ids"], np.int64).tolist()
    if color_step != int(selection["step"]):
        raise ValueError(
            "exported color correction does not match the selected "
            "Gaussian snapshot"
        )
    if camera_ids != sorted(int(value) for value in dataset.cameras):
        raise ValueError(
            "exported color correction camera map does not match the dataset"
        )
    backend.set_color_params(params)
    return digest
