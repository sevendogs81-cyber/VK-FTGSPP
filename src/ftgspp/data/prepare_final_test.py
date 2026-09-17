"""Validate an independent final-test manifest without creating a GPU run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ftgspp.config.compose import load_run_config
from ftgspp.data.scene import Scene, load_final_test_scene


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ftgspp prepare final-test",
        description=(
            "verify hashes, geometry, training identity, and non-overlap with "
            "checkpoint-selection views"
        ),
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    config = load_run_config(run)
    training_scene = Scene.from_config(config)
    final_scene = load_final_test_scene(
        training_scene, args.manifest.expanduser().resolve(),
    )
    if (
        (final_scene.evaluation_identity or {}).get("schema")
        == "ftgspp-final-test-v2"
    ):
        selection_path = run / "export_selection.json"
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "final-test-v2 requires a readable export_selection.json"
            ) from exc
        locked = selection.get("training_dataset_fingerprint")
        current = training_scene.checkpoint_metadata()["fingerprint"]
        if locked is None or str(locked) != str(current):
            raise ValueError(
                "current training data does not match the exported model's "
                "locked dataset fingerprint"
            )
    print(json.dumps(final_scene.evaluation_identity, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
