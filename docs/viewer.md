# Viewer: local window and SSH browser

The viewer supports a local GLFW / Dear ImGui window and a display-free browser
mode for an SSH server. Both use the same native Vulkan 4DGS forward renderer.
Choose the browser mode for an SSH server, the desktop mode for a Linux display,
or headless export for unattended rendering. All three need a supported GPU on
the machine rendering the model.

## SSH browser viewer (no server display)

On the GPU server, install the small remote extra if needed, then start the
server. The command below is for a core + remote environment. **Repeat your
other setup flags** (for example `--with-init --with-remote`) to keep those
components; setup replaces the selected profile. Stop existing jobs before setup:

```bash
bash scripts/setup.sh --with-remote
source .venv/bin/activate
ftgspp viewer --remote --run /path/to/run --device 0 --port 8765
```

On your own computer, create an SSH tunnel, then open the **exact tokenized URL**
printed by `ftgspp viewer` in a browser:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
# Browser on your computer: http://127.0.0.1:8765/?token=...  (use the printed token)
```

The server binds to loopback and accepts one browser at a time. No server X11,
Wayland, NVIDIA display session, GLFW, or desktop build is required. The server
still needs a supported NVIDIA **Vulkan compute** device and driver. `--device`
is the Vulkan ID from `ftgspp doctor`; the server GPU must meet the
[hardware compatibility gate](hardware.md). The browser computer needs only a
current browser and SSH; it does not need an NVIDIA GPU. The remote extra adds
`websockets` but no frontend build system or Python GUI package. The default
headless installation continues to omit the remote extra; `--with-viewer` is
only for the local native window. Both flags can be used together.

To switch between completed DNA-Rendering and 4DAnyone runs in one browser,
pass their parent directory with `--catalog`:

```bash
ftgspp viewer --remote --run /path/to/runs/0012_09 --catalog /path/to/runs
```

The catalog includes immediate subdirectories with `gaussians.npz`,
`viewer_manifest.json`, and a resolved config identifying either dataset.
Finish exports and stop recording before switching. Commands from the previous
scene are discarded or rejected while switching. Sessions, camera paths, and
exports follow the selected run unless their locations were explicitly supplied.
Failed model uploads attempt to restore the previous model from disk.

### Browser controls and image size

The remote UI has orbit, pan, look/fly, optical-axis roll, dolly, fit/reset,
calibrated capture cameras, playback/speed/loop/sample snap, RGB/Alpha/T,
background/exposure, SH/threshold, faithful/interactive quality, PNG screenshot,
camera path keyframes/preview/save/load/export, direct MP4 export, and session save/load.
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
Pixel dimensions are rounded to whole pixels. The model's SH degree and dynamic
threshold are preserved in both quality modes. Interactive frames use JPEG
quality 88; PNG output is lossless.

Remote access is protected by a random per-process URL token and a same-origin
WebSocket check. Keep the printed URL private. The server rejects non-loopback
binds; SSH forwarding is the intended network transport.

## Local desktop installation

If adding the desktop viewer to an existing environment, repeat its other
setup flags alongside `--with-viewer` to retain those components.

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
- **Fixed camera + scene time** advances scene time without moving the camera.

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
with FFmpeg's diagnostic instead of a generic pipe error. `scripts/setup.sh`
installs and checks these tools for the supported setup path.

The remote worker serializes export with interactive rendering because both use
one renderer. View updates received during export are coalesced and the newest
view is rendered afterward. A video export is limited to 3840×2160, 240 FPS and
100,000 frames. PNG path sequences have the same frame-count cap and are
published only after all frames and their manifest are complete; interrupted
sequences are removed. Interaction recording stops automatically at 10,000
samples, preserving the captured path for export. Cancellation, disconnects and
encoder errors remove the partial MP4; completed videos and their manifests
remain. Use the headless CLI for unattended batch export. Odd image dimensions
are padded to even dimensions for H.264; the video manifest records the encoded
dimensions and each frame retains
its original render camera. JSON is the implemented camera-path exchange format;
EasyMocap path import/export is not included.

## Headless export

Works with the default headless installation; no GLFW or display is needed.

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

## Troubleshooting

| Symptom | What to check |
|---|---|
| Missing `websockets` | Rerun setup with **all** required flags plus `--with-remote` |
| Browser cannot connect | Keep the server and SSH tunnel running; use matching ports and the exact printed URL |
| Invalid token or forbidden request | Use the current process's URL; restarting changes the token; do not replace localhost with the server IP |
| Viewer already connected | Close the other tab/client; only one client is supported |
| Missing display / desktop module | Use `--remote` on an SSH server; desktop needs a viewer build and X11/XWayland |
| Black or cropped view | Select a calibrated camera, fit/reset, choose a recorded timestamp, and check exposure/background |
| Motion looks soft | Pause and take a faithful PNG before judging training quality; interactive motion and JPEG/video can reduce detail |
| Cannot export to an existing folder | Choose a fresh output directory; keep completed exports |
| Missing/stale viewer metadata | See [assets](#assets); keep the native model and matching manifest together |

For implementation and profiling details see [architecture](architecture.md#viewer-rendering).
Maintainer acceptance commands are in [dev](../dev/README.md#viewer-verification).
