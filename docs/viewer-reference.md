# Viewer exports and saved sessions

For connection and navigation, start with [the viewer guide](viewer.md).
Paths below are on the machine running the viewer, usually the GPU server.

## Screenshots

Use **Screenshot** or **Ctrl+S**. Images go to `RUN/exports/` and download in the
browser. Each PNG has a JSON sidecar recording its camera, time and settings.
Screenshots exclude UI and overlays and use full output resolution even in
interactive quality mode. Choose transparent output in RGB mode for straight
alpha PNGs. MP4 uses an opaque background.

The original camera aspect ratio is preserved. `--width` and `--height` specify
a bounding box rather than stretching the image. Browser viewing is capped at
1920×1080; screenshots and video export can reach 3840×2160.

## Export a video

The browser video panel offers three routes:

| Mode | Steps |
|---|---|
| Recorded interaction | Start recording, move around and play/pause, stop, then export |
| Keyframed camera path | Add camera keyframes, preview the path, choose scene timing, then export |
| Fixed camera + scene time | Choose a view, scene start/end time and FPS, then export |

For keyframed paths, camera-path time and scene time are separate: you can
orbit a frozen subject or advance scene time while moving. Set the output width
and FPS, check the displayed duration/frame count, then start export.

The page shows progress, cancellation and a download link. Files remain under
`RUN/exports/`. Export occupies the renderer, so interactive updates wait until
it finishes. FFmpeg with `libx264` is required; setup installs/checks it.

## Save a session or camera path

The viewer normally saves `RUN/viewer_session.json` on exit and restores the
same model paused. Use the UI's **Save session** before ending a remote session.
Explicit CLI flags override saved settings. To reset the view without loading
a session, start with `--no-session`.

**Save path** and **Load path** use `RUN/viewer_path.json`. A supplied
`--camera-path FILE` must already exist. Use these options for writable or
alternate locations:

```bash
ftgspp viewer --remote --run /path/to/run \
  --session /path/to/viewer_session.json --output /path/to/exports
```

## Render without opening the viewer

Save a camera path in the viewer, then render it as frames or a video:

```bash
ftgspp render --run /path/to/run --camera-path /path/to/viewer_path.json \
  --width 1920 --height 1080 --fps 30 --device 0 --output frames/orbit
ftgspp render --run /path/to/run --camera-path /path/to/viewer_path.json \
  --width 1920 --fps 30 --device 0 --video --output frames/orbit-video
```

Choose a new output directory. `--video` keeps the PNG frames and adds an MP4.
For a single frame, use the [still-render command](viewer.md#save-images-and-video).

## Open older models

Current runs include `viewer_manifest.json`. For an older run, recover metadata
from its saved configuration and existing calibration or verified DA3 cache:

```bash
ftgspp prepare viewer --run /path/to/old/run
ftgspp viewer --remote --run /path/to/old/run
```

A standalone NPZ opens with `--model /path/to/gaussians.npz`. Without matching
metadata it uses an estimated free view; supply playback settings explicitly
with `--time-range START END --fps FPS`.

## Switch between captures

For completed DNA-Rendering or 4DAnyone runs, add `--catalog /path/to/runs` to
the viewer command. Eligible immediate subdirectories appear in the selector.
Finish exports and stop recording before switching. Sessions and exports follow
the selected run unless you explicitly set other paths.
