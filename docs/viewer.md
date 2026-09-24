# Viewer: local window and SSH browser

The viewer supports a local GLFW / Dear ImGui window and a display-free browser
mode for an SSH server. Both use the same native Vulkan 4DGS forward renderer.
The local window reads the resident `float4(F.rgb, T)` buffer directly; the remote
mode reads back, encodes and sends frames to the browser. The model stays on one
server GPU. Headless export remains available without either UI.

## SSH browser viewer (no server display)

On the GPU server, install the small remote extra and start the server:

```bash
bash scripts/setup.sh --with-remote
source .venv/bin/activate
ftgspp viewer --remote --run /path/to/run --device 0 --port 8765
```

On your own computer, create an SSH tunnel, then open the **exact tokenized URL**
printed by `ftgspp viewer` in a browser:

```bash
ssh -L 8765:127.0.0.1:8765 USER@SERVER
# Browser on your computer: http://127.0.0.1:8765/?token=...  (use the printed token)
```

The server binds to loopback and accepts one browser at a time. No server X11,
Wayland, NVIDIA display session, GLFW, or desktop build is required. The server
still needs a supported NVIDIA **Vulkan compute** device and driver. `--device`
is the Vulkan ID from `ftgspp doctor`. The browser computer needs only a current
browser and SSH; it does not need an NVIDIA GPU. The remote extra adds
`websockets` but no frontend build system or Python GUI package. The default
headless installation continues to omit the remote extra; `--with-viewer` is
only for the local native window. Both flags can be used together.

The remote UI has orbit, pan, look/fly, optical-axis roll, dolly, fit/reset, calibrated capture
cameras, playback/speed/loop/sample snap, RGB/Alpha/T, background/exposure,
SH/threshold, faithful/interactive quality, PNG screenshot, camera path
keyframes/preview/save/load/export, direct MP4 export, and session save/load.
Session and path files are stored on the **server** beside the model by default.
PNG screenshots, path image sequences, and MP4 videos are saved under
`RUN/exports/`. Remote screenshots download automatically, and completed MP4
files have a download link. The remote page shows the absolute server export
directory below the screenshot controls.
Use `--session`, `--camera-path`, and
`--output` for alternate server paths. A supplied `--camera-path` is loaded on
start and must already exist. Automatic session load/save works as in the native
viewer; `--no-session` disables it. The image always keeps the original camera
aspect ratio, with centered letterboxing as the browser or local window changes.
Orbit, pan, dolly, fit/reset, playback and old sessions cannot unlock the aspect.
Orbit, pan, look/fly, dolly and fit preserve scene playback: a playing scene
keeps advancing while navigating, and a paused scene stays paused.
Use the camera panel's `Roll -90°`, `Flip 180°`, and `Roll +90°` buttons to
correct sideways or upside-down datasets exactly. `Ctrl + left drag` adjusts
roll continuously. Orbit, look, pan, dolly, and fit preserve the chosen roll;
reset or selecting a calibrated camera restores that camera's stored pose.
Selecting another calibrated camera uses that camera's original aspect.
`--width`/`--height` set the initial remote bounding box; browser resizes choose
later boxes up to 1920×1080 for viewing. Screenshots and path exports fit the
original aspect into the full browser viewport even when interactive quality
reduces moving frames to 75%; their bounding box is capped at 3840×2160.
Pixel dimensions are rounded to whole pixels. The model's SH degree and dynamic threshold are preserved in both
quality modes. Interactive frames are JPEG quality 88; PNG output is lossless.
Frames keep FP32 F/T on the GPU. A Vulkan compute shader composes display
options and packs RGBA8; packing and D2H share one submission and completion
wait. Readback writes a reusable uint8 NumPy buffer for existing JPEG/PNG
encoding (4 bytes/pixel plus a 4-byte non-finite status, instead of 16
bytes/pixel). Unchanged render inputs reuse GPU F/T;
exposure/background/mode changes only pack, read back and encode;
identical image requests reuse the encoded bytes. The browser sends state
deltas, waits for image decode before requesting the next frame, and stops its
animation loop while paused. The server keeps only the latest pending view.
The status bar reports host-observed total/forward/pack+readback time, cache reuse,
renderer buffer allocation and encoded size. The frame protocol reports
`rgba8_ms`, `readback_bytes` and `encode_ms`; composition and D2H have a combined
host timing, not separate GPU timestamps. These are not end-to-end latency or FPS.

Remote access is protected by a random per-process URL token and a same-origin
WebSocket check. Keep the printed URL private. The server rejects non-loopback
binds; SSH forwarding is the intended network transport.

## Install and open

Linux x86_64, Python 3.12, a supported NVIDIA Vulkan GPU, and an X11 desktop
(or XWayland providing Vulkan presentation) are required for the window.

```bash
bash scripts/setup.sh --with-viewer
source .venv/bin/activate
ftgspp viewer --run /path/to/run --device 0
```

`--with-viewer` adds GLFW 3.4 and Dear ImGui v1.91.9b at native build time. Their
archives are pinned by SHA-256. No Torch, Qt, browser, CUDA/OpenGL interop, or
Python GUI package is added. The default installation builds with
`FTGSPP_BUILD_VIEWER=OFF`. Dependencies for the optional X11 build are
`libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`.

For an existing development environment, rebuild only the project:

```bash
uv pip install --python .venv/bin/python --reinstall --no-deps . \
  -C cmake.define.FTGSPP_BUILD_VIEWER=ON
```

Device IDs are the **Vulkan IDs**, as printed by `ftgspp doctor`, and may differ
from `nvidia-smi` indices. `--device` takes precedence over device environment
variables. The window requires one queue supporting compute, graphics and
presentation to the chosen display. A plain Xvfb display may not provide NVIDIA
Vulkan presentation. If there is no accessible GPU desktop, use
`ftgspp viewer --remote` or `ftgspp render`.

## Assets

A new training run publishes `viewer_manifest.json` beside `gaussians.npz` after
final model selection. It includes the actual post-initialization camera
matrices/intrinsics/sizes, capture timestamps, SH degree, marginal threshold,
model SHA-256 and camera-metadata SHA-256. Static camera poses are deduplicated.
Copy these **two files** to view the result without the original dataset.

```bash
# Old run: recover cameras from its existing calibration or verified DA3 cache.
ftgspp prepare viewer --run /path/to/old/run
# Replace stale metadata explicitly after replacing its model.
ftgspp prepare viewer --run /path/to/old/run --force

# Native NPZ alone: free camera and explicitly specified playback settings.
ftgspp viewer --model /path/to/gaussians.npz --device 0 \
  --time-range 0 10 --fps 30 --sh-degree 3 --marginal-threshold 0.05
```

Old-run conversion reads existing calibration and one image-size reference per
camera; it does not run training preflight, download models, regenerate masks,
or hash every image. For uncalibrated runs it requires an existing verified
completed DA3 camera cache. If an old run cannot recover cameras, opening it
falls back to the NPZ's inferred free view and reports that limitation;
`prepare viewer` fails explicitly instead of publishing placeholder calibration.

For a bare NPZ, bounds and time limits are estimated from a bounded deterministic
sample. SH=3, threshold=0.05 and FPS=30 are defaults, not recovered training
settings. Gaussian `max_duration` is read from the model and is **not video
length**. A mismatched manifest is rejected. PLY is not accepted because this
repository's PLY export omits the gate needed to reproduce the native model.

## Controls and quality

| Action | Control |
|---|---|
| Orbit | Left drag in the viewport |
| Pan | Middle drag or Shift + left drag |
| Look / fly | Right drag; right + WASD / Q/E; Shift accelerates |
| Camera roll | Ctrl + left drag; Camera → Roll −90 / Flip 180 / Roll +90 |
| Persistent fly mode | Camera → Fly |
| Dolly | Mouse wheel |
| Fit / reset | F / R, or Camera buttons |
| Capture camera | Camera selector; fixed original aspect and optional capture timestamp |
| Playback | Space, timeline Play/Pause; Loop, speed, continuous seconds slider |
| Previous / next actual sample | Left / right arrows |
| Snap to actual timestamps | Timeline → Snap |
| Screenshot | Ctrl+S or Screenshot button |
| Fullscreen / resize | F11 / normal window resizing |
| Overlays | Axes, approximate bounds, capture frustums |

The sidebar can be hidden. `Open` accepts a run directory or NPZ path; `Reload`
replaces the resident model. It uses a text path field, not an OS file picker.
The overlays are simple unoccluded guides; bounds are a conservative display
cube derived from sampled model centers, not an exact Gaussian-support hull.

**Faithful is the default.** The local window renders at the physical viewport resolution,
using the model's FP32 parameters, selected SH degree and original dynamic gate /
threshold calculation. Both capture and free cameras retain the original image
aspect and field of view during window resizing; unused space is letterboxed.
RGB, Alpha and Transmittance are
available, with black/white/checker/custom background and exposure in EV.
Background, exposure and visualization changes reuse F/T without another
splatting pass. Paused scenes reuse the cached forward result.

```bash
ftgspp viewer --run /path/to/run --quality interactive --target-fps 30
```

Interactive mode explicitly allows render scales **1.0 → 0.85 → 0.75** when the
measured forward budget is exceeded, with hysteresis. It does not reduce SH,
quantize parameters or cull extra Gaussians. Pausing and leaving the camera still
restores full resolution. The UI reports the actual render extent and scale.
Screenshots and exports always rerender at full output resolution.

`Forward` is host-observed forward submission plus GPU completion time.
`Display + UI` is a GPU timestamp measurement. FPS measures the window loop,
including idle waiting and presentation. `Buffers` counts renderer allocations;
it excludes swapchain/driver/ImGui memory and is not total process VRAM.

## Screenshots, sessions and paths

Screenshots go to `RUN/exports/capture_TIMESTAMP.png`, plus a JSON sidecar
containing the effective camera/time, model hash and render/display settings.
Names are unique and existing PNGs are never overwritten. The saved image
contains the scene; it excludes the sidebar and overlays. Transparent output is
supported in RGB mode: PNG stores **straight alpha**, with `alpha=1-T` and
`RGB=F/alpha`; fully transparent RGB is zero. No extra gamma curve is applied.

On normal exit, the viewer saves `RUN/viewer_session.json`; opening the same
model restores it paused. Explicit CLI flags override the session. A session is
bound to the model hash. Use `--session FILE` to choose another location or
`--no-session` to disable automatic load/save. `Save session` is an explicit UI
save even with automatic persistence disabled. A failed/mismatched auto-load is
reported and ignored. Read-only model directories can use writable `--session`
and `--output` locations.

Add keyframes from the current camera, then select, replace or delete them.
`Key path seconds` changes their position along the camera path; `Preview path`
and `Path seconds` preview the interpolation. Translation and intrinsics
interpolate linearly; orientation uses quaternion SLERP. For video and preview,
4D scene time is configured separately: either freeze it at an explicit time or
advance it over an explicit start/end range. Camera path seconds never
implicitly drive scene seconds.

`Save path` / `Load path` use `RUN/viewer_path.json`, or the file chosen by
`--camera-path`. A path uses schema `ftgspp-camera-path-v1` and a nonempty
`keyframes` array, each with `path_time` and a `camera` containing `width`,
`height`, `K[3][3]`, `w2c[4][4]`, and `timestamp`. Path times must strictly
increase; camera poses follow OpenCV (+X right, +Y down, +Z forward).

The remote viewer exports MP4 directly from the browser in three modes:

- **Recorded interaction** is the default. Press `Start recording`, operate the
  viewer normally with orbit, pan, look/fly and wheel zoom, optionally control
  scene Play/Pause, then stop recording. Server receive time becomes camera-path
  time, while each sample retains the 4D scene time actually displayed. Export
  interpolates the captured poses and observed scene times at the requested FPS.
- **Keyframed camera path** samples the user-created spatial keyframes. Its 4D
  timeline is explicit and separate: freeze at one scene time, or advance over a
  selected scene-time range for the duration of the camera path.
- **Fixed camera + scene time** is the earlier fixed-view behavior and is labeled
  as such. It advances scene time without moving the camera.

The video panel controls FPS and output width; height follows the recorded/path
camera aspect ratio. The UI reports effective extent, duration, and frame count
before export.

Remote MP4 export snapshots the current camera/path and display settings when it
starts. It sends the GPU-packed RGBA8 frames directly to an FFmpeg raw-video pipe,
then atomically publishes an H.264/CRF18/yuv420p file and JSON manifest under
`RUN/exports/`. It does not write an intermediate PNG sequence. The page shows
progress, measured frame throughput and estimated remaining time, supports
cancellation, and provides inline playback plus a token-protected download after
completion. MP4 is opaque and uses the selected background. Encoding uses the
resource-bounded x264 `veryfast`/`zerolatency` pipeline with four threads so
high-resolution exports do not accumulate a large lookahead queue. An installed
`ffmpeg` with `libx264` is required on the server; encoder failures are reported
with FFmpeg's diagnostic instead of a generic pipe error.

The remote worker serializes export with interactive rendering because both use
one renderer. View updates received during export are coalesced and the newest
view is rendered afterward. A video export is limited to 3840×2160, 240 FPS and
100,000 frames. Cancellation, disconnects and encoder errors remove the partial
MP4; completed videos and their manifests remain. Use the headless CLI for
unattended batch export. JSON is the implemented camera-path exchange format;
EasyMocap path import/export is not included.

## Headless export

Works with the default headless installation; no GLFW or display is needed.
The headless export path also uses GPU RGBA8 packing and reuses its output array.
The native window continues to display F/T directly; its existing FP32 screenshot
interface and the Python CPU compositor remain available as reference paths.

```bash
# Single calibrated view at a specified scene time.
ftgspp render --run /path/to/run --camera 0 --time 1.25 --device 0 \
  --width 1920 --transparent --output frames/still

# Deterministic trajectory and optional video.
ftgspp render --run /path/to/run --camera-path /path/to/viewer_path.json \
  --width 1920 --height 1080 --fps 30 --device 0 --output frames/orbit
ftgspp render --run /path/to/run --camera-path /path/to/viewer_path.json \
  --width 1920 --fps 30 --device 0 --video --output frames/orbit-video
```

The output directory must be new. It contains `000000.png`, etc., and
`render.json` with every sampled camera/time and relevant settings. A static
camera defaults to one frame; `--duration` repeats it for a specified duration.
For a static camera observing animation, create two keys with identical camera
poses and different scene timestamps. Paths sample `start + i/fps`, excluding
the endpoint at exactly `duration`, with at least one frame. A longer explicit
`--duration` holds the last key after the path ends. A single width or height
preserves aspect; specifying both rescales the camera intrinsics to that extent.

`--video` invokes an installed `ffmpeg` using libx264/CRF18/yuv420p and keeps the
lossless PNGs. Video is lossy, pads odd dimensions to even, and does not support
transparent alpha. PNG export has no FFmpeg dependency. Failed or interrupted
exports retain the completed frames; choose a new directory to rerun.

## Verification and current limits

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
