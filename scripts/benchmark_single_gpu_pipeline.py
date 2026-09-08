#!/usr/bin/env python3
"""Steady-state single-GPU pipeline telemetry helper.

Writes JSONL rows when FTGSPP_PIPELINE_TELEMETRY points at a file. The full
30k experiment is launched separately with the same env var.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/ftgspp_pipeline_telemetry.jsonl"),
    )
    args = parser.parse_args()
    os.environ["FTGSPP_PIPELINE_TELEMETRY"] = str(args.output)
    print(json.dumps({
        "telemetry_path": str(args.output),
        "hint": "export FTGSPP_PIPELINE_TELEMETRY before python -m ftgspp.run",
    }))


if __name__ == "__main__":
    main()
