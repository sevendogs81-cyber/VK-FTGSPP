# Maintainer tree

This directory is **not** part of the user path (`scripts/setup.sh`,
`ftgspp`, and `examples/`). Keep it for CI, agents, and local research.
Unsupported `tools/` are omitted from release source distributions.

- `tests/` — CPU contracts and GPU engineering scripts. Official evaluation
  is still `ftgspp eval`.
- `tools/` — unsupported research helpers; they may change without notice.

## Viewer verification

The reproducible checks are:

```bash
PYTHONPATH=src .venv/bin/python -m pytest dev/tests/test_viewer.py dev/tests/test_remote_viewer.py -q
node --test dev/tests/viewer_browser.test.cjs
.venv/bin/python dev/tests/gpu/check_viewer.py --device 0
.venv/bin/python dev/tests/gpu/check_remote_viewer.py \
  --model build/viewer/acceptance/gaussians.npz --device 0
# On an accessible NVIDIA Vulkan desktop:
.venv/bin/python dev/tests/gpu/check_viewer.py --device 0 --desktop \
  --output build/viewer/desktop-acceptance

# With FTGSPP_BUILD_VIEWER=ON and BUILD_TESTING=ON:
build/viewer/viewer_display_test build/viewer/shaders 0
build/viewer/viewer_pack_test build/viewer/shaders 0
.venv/bin/python dev/tests/gpu/check_viewer_rgba8.py --model /path/to/run \
  --device 0 --output build/viewer/rgba8/my-scene-quality

# Run directory recovers calibration; use a bare NPZ for an inferred free view.
.venv/bin/python dev/tests/gpu/benchmark_viewer.py --model /path/to/run \
  --camera 0 --time 1 --time-step 0.033333333 --device 0 \
  --output build/viewer/performance/my_scene.json
```

The GPU script checks all four SH levels at three timestamps against the native
evaluation renderer, forced tile-overflow replay, repeat upload, resizing and an
empty model. Its `--desktop` branch additionally checks idle reuse, window
resizing, reload, playback and capture equivalence. The display executable uses
the **production graphics pipeline** with a test-only offscreen color attachment
to check orientation, 1:1 / bilinear sampling, backgrounds, exposure and Alpha/T
without a display server.

The contracts cover manifest validation, image composition, browser protocol,
authorization, caching, resizing, path/session persistence and export failure
handling. GPU acceptance compares all four SH levels and multiple timestamps
with the training renderer, exercises tile-overflow replay and validates display
and RGBA8 packing. A real desktop session is still required to verify WSI and
human interaction on each target driver/window-system combination. Throughput
depends on scene density, view, resolution, encoding and the SSH link; the
reported per-stage timings are diagnostic rather than an FPS guarantee.
Training preview, GT/depth diagnostics and multi-GPU presentation remain outside
this version.
