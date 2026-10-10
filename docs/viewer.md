# View and export

## Open in your browser

On the GPU machine, with `.venv` activated:

```bash
ftgspp viewer --remote --run /path/to/run --device 0
```

If browser support is missing, install it with
`bash scripts/setup.sh --with-remote`. Open the printed URL, including its token.
For a remote GPU server, first run this on your own computer:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

Leave the viewer and tunnel running. The viewer listens on localhost and
accepts one browser client; your computer only needs a browser and SSH.

## Explore

| Action | Control |
|---|---|
| Orbit | Left drag |
| Pan | Middle drag or Shift + left drag |
| Look / fly | Right drag; right + WASD / Q/E |
| Zoom | Scroll |
| Fit / reset | F / R |
| Play / pause | Space or the playback button |
| Select a recorded view | Camera selector |
| Correct orientation | Camera roll buttons |
| Screenshot | Screenshot button |

Use **Faithful** quality to inspect details. **Interactive** can reduce preview
resolution during motion; screenshots and exports use full output resolution.
Browser playback and exports align scene time to captured samples when available.

## Save images and videos

Screenshots download as PNG and remain in `RUN/exports/` on the server.
Choose transparent output in RGB mode for alpha PNGs; MP4 uses a solid background.

In the browser's video panel, choose one route:

| Route | What to do |
|---|---|
| Record your interaction | Start recording, navigate/play, stop, then export |
| Camera path | Add camera keyframes, preview, choose scene timing, then export |
| Fixed camera | Choose a view and scene start/end time, then export |

Set width and FPS, check the displayed duration, then export. A download link
appears on completion; files stay in `RUN/exports/`. Navigation waits during
export. For a camera path, choose whether the subject stays frozen or animates
while the camera moves.

## Save a view or path

Use **Save session** to keep the view; the same model restores paused.
Start with `--no-session` to ignore a saved session. **Save path** and
**Load path** use `RUN/viewer_path.json`; `--camera-path FILE` loads another
existing path. Use `--output DIR` for an alternate export directory.

## Render from the command line

Use a new output directory. For a still at a recorded time:

```bash
ftgspp render --run /path/to/run --camera 0 --time 0 \
  --device 0 --output frames/still
```

This writes `000000.png`. `--camera` is a zero-based viewer camera-list index.
For a saved camera path, export PNG frames and an MP4:

```bash
ftgspp render --run /path/to/run --camera-path /path/to/viewer_path.json \
  --width 1920 --fps 30 --device 0 --video --output frames/orbit
```

MP4 needs FFmpeg with `libx264`, provided by setup. Headless paths follow their
explicit scene times. See `ftgspp render --help` for additional options.

## Desktop window

On Linux with an X11/XWayland display:

```bash
bash scripts/setup.sh --with-viewer
source .venv/bin/activate
ftgspp viewer --run /path/to/run --device 0
```

## Troubleshooting

- **Cannot connect:** keep the server and tunnel running, with matching ports.
- **Forbidden:** use the current printed URL; restarting changes the token.
- **Already connected:** close the other viewer tab.
- **Blank view:** select a capture camera and timestamp, then fit/reset.
- **Old run missing metadata:** run `ftgspp prepare viewer --run /path/to/run`.
- **After a server upgrade:** reload the browser page.
