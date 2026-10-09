# Prepare your data

Start with synchronized multi-view images. The easiest supported layout uses
numeric camera directories and numeric frame filenames. The training guide
uses a capture called `my_capture`; replace it with your own path.

## Required and optional inputs

| Input | Scene mode | Foreground mode |
|---|---|---|
| RGB images, capture FPS and camera split | Required | Required |
| Camera calibration | Supply it, or estimate with DA3 | Required for carving; DA3 can estimate it on the direct DA3 route |
| Foreground masks | Not used | Required: existing files, image alpha, or BiRefNet output |
| Initialization geometry | Existing keyframe RGB PLYs, or DA3 | Prepared carving/filtering cache, existing points, or direct DA3 |

Indexing images does not generate calibration, masks or points. It does not
verify that the images describe a well-synchronized, calibrated capture.

## Directory layout

```text
my_capture/
├── intri.yml                    # EasyMocap intrinsics, when available
├── extri.yml                    # EasyMocap world-to-camera extrinsics
├── images/
│   ├── 0/000000.jpg
│   ├── 0/000001.jpg
│   ├── 1/000000.jpg
│   └── 1/000001.jpg              # every camera has every selected frame
└── fmasks/                      # optional existing external masks
    ├── 0/000000.png
    ├── 0/000001.png
    ├── 1/000000.png
    └── 1/000001.png
```

Use unique camera/frame pairs and a contiguous, rectangular selected frame
range. PNG and JPEG are convenient RGB formats. Use a lossless format for masks.
A flat `c<camera>-f<six-digit-frame>.*` image directory is also supported.
For other filenames, use a dataset manifest as described below.

`--eval-cameras` takes **camera IDs**, not list positions. The remaining cameras
train the model unless an explicit `--train-cameras` list is supplied. Leave at
least one training camera. Omit the eval flag only when you intentionally want
no held-out camera evaluation; training views cannot establish novel-view quality.

## Frames, FPS and synchronization

- Set `--fps` to the capture rate, not a preview video's encoding rate. Omitting
  it on image indexing uses 60 FPS; the tool does not measure capture FPS.
- `--start` is inclusive and `--stop` is exclusive. `0..225` selects frames
  0 through 224. Without these flags, indexing selects the available frame span.
- With no per-view timestamps or time-offset file, cameras are treated as
  synchronized and scene time is `frame / fps` seconds.
- A `timecode.json`, `t_offsets.json` or `time_offsets.json` in a searched data
  root supplies camera-to-offset seconds, for example `{"0": 0.0, "1": 0.002}`.
  The convention is `frame / fps - offset`. A supplied file must cover selected
  cameras under the default strict policy. Do not add offsets merely to make a
  validation error disappear; establish synchronization from the capture.

FPS also determines the default initial Gaussian duration:
`init.keyframe_stride / data.fps`. Changing FPS changes motion semantics.

## Camera calibration

Use an EasyMocap/OpenCV FileStorage `intri.yml` and `extri.yml` pair. Both refer
to the same camera names as the images. Intrinsics contain `names`, `K_<name>`
and optionally `D_<name>` (or `dist_<name>`), `H_<name>` and `W_<name>`.
Extrinsics contain `T_<name>` and either a `Rot_<name>` rotation matrix or an
`R_<name>` Rodrigues vector. The tiny [example generator](../examples/make_tiny_scene.py)
writes a minimal working pair.

Extrinsics map world points into OpenCV camera coordinates: +X right, +Y down,
+Z forward. Supply intrinsics for the source image resolution. The loader
handles its configured scaling and calibrated undistortion; keep RGB and masks
aligned in the same source-image coordinates.

Camera translations, input PLY positions and explicit carving bounds must use
one world frame and consistent units. Bounds are six numbers in the order
`xmin ymin zmin xmax ymax zmax` and must contain the subject throughout the
selected sequence. Do not copy bounds from another capture.

## Existing masks

For the directory above, index with `--pipeline foreground --mask-root
/path/to/my_capture/fmasks`. This avoids BiRefNet entirely. For RGBA images use
`--pipeline foreground --mask-source alpha` instead.

External masks are scalar images: 0 is background, 1 is foreground after
normalization. Use grayscale uint8 PNG with values 0–255 (or uint16 with
0–65535); intermediate values represent soft edges. Floating maps must be
finite and in [0, 1]. RGB masks with disagreeing channels are rejected.
Masks must match their RGB image's original dimensions and pixel alignment.
RGB is straight/unpremultiplied by default; declare `rgb_encoding:
premultiplied` explicitly if that is how your files were produced.

For nonstandard paths, set `data.masks.manifest_path` to a JSON file with
`views` containing `frame`, `camera_id` and `mask_path` entries. Paths are
relative to that manifest. Generated BiRefNet manifests additionally include
hashes and completion markers; do not edit those caches manually.

## Custom image manifest

A minimal `views.json` for one frame and two cameras is:

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

Paths are relative to the manifest. Optional `timestamp` values, in seconds,
must cover all selected views or none. Optional `rgb_sha256` values are checked.
Run `ftgspp prepare dataset --manifest views.json --eval-cameras 1` to generate
a starter config. See `ftgspp prepare dataset --help` for custom output paths.

## Existing point clouds

Scene initialization with `init.backend: auto` looks for a complete RGB PLY set
at `points_stride10/f000000.ply`, `f000010.ply`, and so on, including the last
selected frame. Adjust the directory and frame spacing to `init.keyframe_stride`.
Each point needs world-space XYZ and RGB. An incomplete set can trigger DA3;
use `init.backend: points` when existing points are an explicit requirement.

Foreground `carving` and `point_filtering` caches additionally need the matching
integrity manifest produced by `ftgspp prepare points`. Arbitrary PLY files are
not interchangeable with these caches. Existing PLYs can instead use the explicit
`points` backend or supply `--input-points` during geometry preparation.

Continue with [training](usage.md).
