# Data formats

For the complete workflow, follow [Train your first capture](first-run.md).
This page covers input formats and reusing masks, calibration or point clouds.

## Images and cameras

Use synchronized images from a fixed camera rig. Each camera must have every
frame in the selected range, with numeric camera IDs and frame filenames:

```text
my_capture/
├── intri.yml                 # optional EasyMocap calibration
├── extri.yml
├── images/
│   ├── 0/000000.jpg
│   ├── 0/000001.jpg
│   ├── 1/000000.jpg
│   └── 1/000001.jpg
└── fmasks/                   # optional existing masks
    ├── 0/000000.png
    ├── 0/000001.png
    ├── 1/000000.png
    └── 1/000001.png
```

A flat directory with `c<camera>-f<six-digit-frame>.*` filenames also works.
For arbitrary paths, use a [manifest](#custom-image-manifest).
`--eval-cameras` takes camera IDs, not list positions; the remaining cameras
train the model. Leave at least one training camera.

Set `--fps` to the actual indexed image rate. `--start` includes its frame and
`--stop` excludes it. Without these flags, indexing uses the available span.
For example, `--start 0 --stop 225` selects frames 0 through 224.

Indexing validates camera splits and output paths before writing. `--overwrite`
allows replacing generated files; it rejects collisions with input images,
calibration, supplied mask directories and point-cloud files. Each output file
is published atomically.

## Calibration and synchronization

Place EasyMocap/OpenCV FileStorage `intri.yml` and `extri.yml` beside `images/`,
or pass `--calibration DIR` during indexing. Camera names must match the images.
Intrinsics contain `names`, `K_<name>` and optionally distortion `D_<name>`
(or `dist_<name>`), `H_<name>` and `W_<name>`. Extrinsics contain `T_<name>` and
either `Rot_<name>` (matrix) or `R_<name>` (Rodrigues vector).
The [tiny example generator](../examples/make_tiny_scene.py) writes a working pair.

Extrinsics map world points to OpenCV camera coordinates: +X right, +Y down,
+Z forward. Supply intrinsics for the original image resolution. Camera
translations and point clouds must share one world frame and unit of distance.
Without calibration, DA3 estimates the static rig during initialization/preparation.

Cameras are synchronized by default, with time `frame / fps`. To account for
known offsets, put `timecode.json`, `t_offsets.json` or `time_offsets.json` in
the capture root, e.g. `{"0": 0.0, "1": 0.002}`. Values are seconds, covering
all selected cameras; scene time becomes `frame / fps - offset`.

## Use existing masks

Add `--mask-root /path/to/fmasks` when indexing foreground images, then skip
BiRefNet. Use grayscale PNGs with the same dimensions, camera/frame names and
pixel alignment as the images: 0 is background, 255 is foreground, and values
between them represent soft edges. For RGBA images, use `--mask-source alpha`.

RGB is straight/unpremultiplied by default. If your files are premultiplied,
set `data.masks.rgb_encoding: premultiplied`. For arbitrary mask paths,
`--mask-manifest FILE` accepts JSON `views` entries with `frame`, `camera_id`
and `mask_path`; paths resolve relative to that manifest.

## Use existing point clouds

Supply one RGB PLY per initialization keyframe, named `f000000.ply`, etc.,
including the last selected frame. Points need world-space XYZ and RGB values.
Use their actual frame spacing as `N` below:

```bash
ftgspp prepare dataset --images "$CAPTURE/images" --fps "$FPS"   --pipeline foreground --eval-cameras "$EVAL_CAMERA" --config "$CONFIG"   --mask-root "$CAPTURE/fmasks" --points /path/to/rgb_points --keyframe-stride N
```

This selects `init.backend: points`; complete calibration and foreground masks
let you train without BiRefNet or DA3. For scene mode, use `--pipeline scene`
and omit `--mask-root`. The default model size follows the selected PLY point
count; inspect it with `ftgspp train --config "$CONFIG" --print-config`.

For **XYZ-only clouds**, first index the images and calibration as usual. Set
`init.keyframe_stride` to the clouds' actual frame spacing, then add image colors:

```bash
ftgspp prepare points --method colorize --config "$CONFIG"   --input-points /path/to/xyz_points --output "$CAPTURE/points_rgb"
```

Use the training command printed on completion. Add `--color-mode gray` for
neutral colors instead. To also filter a foreground cloud against its masks,
use `--method point_filtering`. These commands preserve user point counts
unless filtering removes points or you explicitly set `--max-points`.

Generated carving/filtering directories include integrity manifests. Keep
those files with their PLYs; arbitrary PLYs cannot substitute for a prepared
cache. Choose scale, frames and camera split before preparation. If they change,
regenerate matching geometry in a new directory.

## Custom image manifest

For example, save this as `views.json`:

```json
{
  "schema": "ftgspp-dataset-v1",
  "fps": 30,
  "views": [
    {"frame": 0, "camera_id": 0, "rgb_path": "rgb/left.png"},
    {"frame": 0, "camera_id": 1, "rgb_path": "rgb/right.png"}
  ]
}
```

Paths are relative to the manifest. Optional `timestamp` values are seconds and
must cover all selected views. Index it with
`ftgspp prepare dataset --manifest views.json --eval-cameras 1`.
