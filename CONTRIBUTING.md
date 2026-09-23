# Maintainer notes

This repository is maintained by a single author. Code contributions are not
solicited for this release. Read LICENSE before using or redistributing, and
keep third-party attribution and the research-only licensing boundary.

Users should install with `bash scripts/setup.sh` from a clone. Maintainers
use Python 3.12, NumPy 2 and pybind11 3.0.1. CPU tests run with
`uv sync --python 3.12 --extra dev --no-install-project`. Native or shader
changes need a clean `uv pip install .` (or `uv build --python 3.12`) plus
the relevant GPU fixtures. Keep Slang 2026.10 and uv 0.11.29 pins in
`scripts/setup.sh` identical to `.github/workflows/artifacts.yml`.
Shader layout changes must update the C++ push constants/descriptors and
behavior tests together.

The supported surface is the installed CLI plus the YAML keys documented in
README, `docs/usage.md`, and `docs/architecture.md`, on **one GPU**. Two-device
pair code, `dev/tests/gpu/` scripts, and `.github/workflows/gpu.yml` stay in the
tree for engineering; they are in development and not a user-facing
first-release path. `dev/tools/` scripts are unsupported research helpers and
may change without notice.

Do not commit generated SPIR-V, extension binaries, datasets, weights, runs, or
the local-only notes `docs/foreground_reconstruction_plan.md`,
`docs/initialization_pipeline_improvement_plan.md`, `docs/validation.md`,
`docs/viewer_plan.md`, `docs/viewer_performance.md`, `docs/viewer_rgba8.md`, or
the raw `docs/viewer*_benchmark_*.json` measurements.
Keep optional Torch imports out of the ordinary CLI/training path.

Report issues with the command/config, doctor output, package/build versions and
a small reproducer. Remove private paths/data before sharing logs. Do not upload
confidential scenes. GPU CI runs only on explicitly controlled hosts.
