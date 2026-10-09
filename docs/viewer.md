# View and export a model

## Open in your browser

Install browser support once with `bash scripts/setup.sh --with-remote`.
On the GPU machine, start the viewer with your completed run:

```bash
ftgspp viewer --remote --run /path/to/run --device 0
```

On the same machine, open the link printed in the terminal. For an SSH server,
run this on your own computer first and leave both commands running:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

Open the exact printed URL, including its token. Only one browser client can
connect at a time. Rendering runs on the server GPU; your computer needs only a
browser. The server listens on localhost; SSH provides remote access.

## Explore

| Action | Control |
|---|---|
| Rotate around the subject | Left drag |
| Pan | Middle drag or Shift + left drag |
| Zoom | Scroll |
| Fit / reset | F / R |
| Play / pause | Space or the playback button |
| Select a recorded view | Camera selector |
| Correct a sideways view | Camera roll buttons |
| Save a screenshot | Screenshot button |

Pause on a frame and select a capture camera when comparing with the source
images. Use **Faithful** quality for inspection; **Interactive** can reduce
resolution during motion to improve responsiveness.

## Save images and video

Screenshots download as PNG. For video, use the camera-path controls to add
views and preview the path, then export MP4. Files are also saved under
`RUN/exports/` on the server. PNG supports transparency; MP4 uses a solid background.

For an unattended still render, use a new output directory and a time within
your capture's range:

```bash
ftgspp render --run /path/to/run --camera 0 --time 0 \
  --device 0 --output /path/to/new-render
```

This writes `000000.png`. `--camera` selects a viewer camera-list index.
Use the native run or `gaussians.npz`; PLY exports omit dynamic gate information.
See `ftgspp render --help` for camera-path and video options.

## Desktop window

On a Linux machine with an X11/XWayland display:

```bash
bash scripts/setup.sh --with-viewer
ftgspp viewer --run /path/to/run --device 0
```

## Troubleshooting

- **Cannot connect:** keep the server and tunnel open; check their ports match.
- **Forbidden:** reopen the current printed URL; restarting changes the token.
- **Already connected:** close the other viewer tab.
- **Blank view:** choose a capture camera and recorded timestamp, then fit/reset.
- **Missing metadata:** run `ftgspp prepare viewer --run /path/to/run` for an old run.

For sessions, custom camera paths and file formats, see [viewer reference](viewer-reference.md).
