# Maintainer notes

This repository is maintained by a single author. Code contributions are not
solicited for this release. Read LICENSE before using or redistributing, and
keep third-party attribution and the research-only licensing boundary.

Users should install with `bash scripts/setup.sh` from a clone. Maintainers
use Python 3.12, NumPy 2 and pybind11 3.0.1. CPU tests run with
`uv sync --locked --python 3.12 --extra dev --no-install-project`, followed by
`uv run --no-sync pytest`. After native or shader changes, rebuild and install
with `uv pip install --python .venv/bin/python --reinstall --no-deps .`, then run
the relevant GPU fixtures. `uv build --python 3.12` creates a wheel without
installing it. Keep Slang 2026.10 and uv 0.11.29 pins in
`scripts/setup.sh` identical to `.github/workflows/artifacts.yml`.
Shader layout changes must update the C++ push constants/descriptors and
behavior tests together.

The supported surface is the installed CLI plus the YAML keys documented in
`docs/first-run.md`, `docs/arguments.md`, and `docs/architecture.md`, on **one GPU**. Two-device
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

## Optional inference dependencies

The pinned BiRefNet-HR repository publishes a broad
[training/application requirements file](https://huggingface.co/ZhengPeng7/BiRefNet_HR/blob/a7a562f6fd16021180f2f4348f4de003a2d3d1e1/requirements.txt)
that includes `numpy<2`. This project does not install that file. Its
`birefnet` extra declares the direct imports used by the pinned inference
module (`torchvision`, timm, Kornia, and einops) while retaining the project's
tested NumPy 2/OpenCV 4.12 core.

The audited upstream `depth-anything-3==0.1.1` distribution declares its full
demo/server/training environment as hard dependencies, including `numpy<2`.
FTGSPP uses only the inference API and is tested with NumPy 2/OpenCV 4.12. The
uv lock therefore supplies reviewed static metadata for that exact DA3 version
and explicitly declares the smaller inference dependency set in the `da3`
extra. This avoids unrelated Open3D, web-server, pre-commit, and
duplicate OpenCV packages while keeping DA3 itself version-pinned.
