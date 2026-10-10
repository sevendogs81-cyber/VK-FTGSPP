# Train your first capture

This guide reconstructs a foreground subject from synchronized multi-view RGB
images. Complete [installation](../README.md#install) first. For a scene including
the background, use the [scene route](#scene-mode-input-routes) at the end.

## 1. Point to your images

Arrange images as `images/<camera>/<frame>.jpg` or `.png`, with the same frames
in every camera folder. The camera rig should stay fixed during the capture.
See [data formats](data.md) if your files use another layout.

Run the following from the repository, replacing the path, FPS and camera ID:

```bash
CAPTURE=/absolute/path/to/my_capture
CONFIG="$CAPTURE/ftgspp.yaml"
RUN="$CAPTURE/runs/foreground"
FPS=30
EVAL_CAMERA=2
DEVICE=0

ftgspp prepare dataset --images "$CAPTURE/images" --fps "$FPS" \
  --pipeline foreground --eval-cameras "$EVAL_CAMERA" --config "$CONFIG"
```

Use the actual capture FPS. `EVAL_CAMERA` must exist in your images; it is held
out to check the result, while the other cameras train the model. The command
writes `ftgspp.yaml` and indexes the images without moving them.

Have calibration? Put `intri.yml` and `extri.yml` beside `images/`, or add
`--calibration /path/to/cameras` to the command above.

Have masks? Add `--mask-root /path/to/masks` above and skip step 2. For RGBA
images use `--mask-source alpha`. Have point clouds too? Follow the
[existing-points route](data.md#use-existing-point-clouds) before training.

## 2. Create foreground masks

```bash
ftgspp prepare masks --images "$CAPTURE/images" \
  --view-manifest "$CAPTURE/manifests/dataset.json" \
  --output "$CAPTURE/masks/birefnet_hr"
```

BiRefNet downloads its weights on first use. Open a few generated masks and check
that the subject is complete, especially hands, hair and fast-moving parts.

## 3. Prepare the geometry

```bash
ftgspp prepare points --method carving --config "$CONFIG" \
  --output "$CAPTURE/points_vhull_ibr_stride1"
```

This creates colored point clouds for initialization. Existing calibration is
used when available; otherwise DA3 estimates the camera rig and may download
weights. Keep the generated cache directories with the capture.

If you change the output path, use the matching training command printed by
preparation. Otherwise continue with the same `CONFIG` below.

## 4. Train

```bash
ftgspp train --config "$CONFIG" --print-config
ftgspp train --config "$CONFIG" --output "$RUN" --devices "$DEVICE"
```

The first command shows the effective settings without training. Check the
frame range, FPS, camera split and iteration count. The default `balanced`
budget adapts to clip length. For a shorter diagnostic run, add
`--set train.preset=preview` to the training command. See
[common settings](usage.md#change-settings) when you need to adjust memory or quality.

Use a new `RUN` directory for each experiment. On completion, it contains
`gaussians.npz` and the camera information needed for viewing.

## 5. Open the result

```bash
ftgspp viewer --remote --run "$RUN" --device "$DEVICE"
```

Open the printed link. For a remote GPU server, follow the
[SSH connection instructions](viewer.md#open-in-your-browser).
Select the held-out camera and inspect playback at the beginning, middle and
end, including fast motion. Compare with the original images at the same time.

For interrupted training, metrics or unexpected results, see
[resume and troubleshooting](usage.md).

## Scene-mode input routes

To reconstruct the whole image, including the background, choose `scene` when
indexing. This route needs DA3 (`--with-da3` or `--with-init`) if points or
calibration are missing. Skip the foreground mask and carving steps:

```bash
CONFIG="$CAPTURE/scene.yaml"
RUN="$CAPTURE/runs/scene"
ftgspp prepare dataset --images "$CAPTURE/images" --fps "$FPS" \
  --pipeline scene --eval-cameras "$EVAL_CAMERA" --config "$CONFIG" \
  --output-manifest "$CAPTURE/manifests/scene.json"
ftgspp train --config "$CONFIG" --print-config
ftgspp train --config "$CONFIG" --output "$RUN" --devices "$DEVICE"
```

Then open `RUN` in the viewer as above. Existing RGB PLYs can be supplied with
`--points DIR --keyframe-stride N` during indexing; see [point-cloud inputs](data.md#use-existing-point-clouds).
