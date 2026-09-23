"""CLI/I/O layer. Interactive camera and GPU frames stay in the native module."""

from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import time

from .asset import atomic_json, load_asset, legacy_metadata
from .offline import display_options, export_frames, native_module, save_png, shader_dir
from .path import PATH_SCHEMA, from_state, load_path, save_path
from .session import load_session, save_session, validate_state


def parser_for(command):
    p = argparse.ArgumentParser(prog=f"ftgspp {command}")
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument("--run", type=Path, help="completed run with gaussians.npz")
    sources.add_argument("--model", type=Path, help="native gaussians.npz")
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="Vulkan ID from ftgspp doctor; overrides environment selection",
    )
    p.add_argument("--time-range", nargs=2, type=float, metavar=("START", "END"))
    p.add_argument(
        "--fps", type=float, help="timeline/export FPS; defaults to asset FPS"
    )
    p.add_argument("--sh-degree", type=int, choices=range(4))
    p.add_argument("--marginal-threshold", type=float)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--camera", type=int, help="capture camera index (0 based)")
    p.add_argument("--time", type=float, help="scene time in training seconds")
    p.add_argument("--background", choices=("black", "white", "checker"))
    p.add_argument("--mode", choices=("rgb", "alpha", "transmittance"))
    p.add_argument("--exposure", type=float)
    p.add_argument("--transparent", action="store_true", default=None)
    p.add_argument("--camera-path", type=Path)
    if command == "viewer":
        p.add_argument("--remote", action="store_true", help="serve browser viewer without a display")
        p.add_argument("--host", default="127.0.0.1", help="remote bind address (loopback only)")
        p.add_argument("--port", type=int, default=8765, help="remote HTTP/WebSocket port")
        p.add_argument("--quality", choices=("faithful", "interactive"))
        p.add_argument("--target-fps", type=float)
        p.add_argument("--no-vsync", action="store_true")
        p.add_argument(
            "--session",
            type=Path,
            help="load/save session; default: run/viewer_session.json",
        )
        p.add_argument(
            "--no-session",
            action="store_true",
            help="do not automatically load/save session",
        )
        p.add_argument(
            "--output", type=Path, help="capture directory; default: run/viewer_output"
        )
        p.add_argument(
            "--frames", type=int, default=0, help="close after N frames (automation)"
        )
        p.add_argument(
            "--screenshot", type=Path, help="save a full-quality PNG when closing"
        )
        p.add_argument("--stats-json", type=Path, help="write final runtime statistics")
    else:
        p.add_argument(
            "--output", type=Path, required=True, help="new PNG sequence directory"
        )
        p.add_argument(
            "--duration",
            type=float,
            help="output duration in seconds; static camera defaults to one frame",
        )
        p.add_argument(
            "--video", action="store_true", help="also encode MP4 using optional ffmpeg"
        )
    return p


def options(args):
    state = {}
    if args.background:
        state.update(
            background=[1, 1, 1] if args.background == "white" else [0, 0, 0],
            checker=args.background == "checker",
        )
    if args.mode is not None:
        state["mode"] = ("rgb", "alpha", "transmittance").index(args.mode)
    for name in (
        "exposure",
        "transparent",
        "sh_degree",
        "marginal_threshold",
        "time",
        "target_fps",
    ):
        value = getattr(args, name, None)
        if value is not None:
            state[name] = value
    if getattr(args, "quality", None):
        state["interactive"] = args.quality == "interactive"
    return validate_state(state)


def select_camera(asset, args):
    if args.camera is None:
        c = asset.metadata["initial_camera"]
    else:
        if not 0 <= args.camera < len(asset.metadata["cameras"]):
            raise ValueError("Capture camera index outside available cameras")
        c = asset.metadata["cameras"][args.camera]
    return dict(c, timestamp=args.time if args.time is not None else c["timestamp"])


def unique_path(root, stem, suffix=""):
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return root / f"{stem}_{stamp}{suffix}"


def screenshot(app, asset, path):
    state = app.state()
    c = app.output_camera()
    ft = app.capture()
    save_png(path, ft, **display_options(state))
    atomic_json(
        Path(path).with_suffix(".json"),
        dict(
            schema="ftgspp-capture-v1",
            model_sha256=asset.sha256,
            camera=c,
            display=display_options(state),
            sh_degree=state["sh_degree"],
            marginal_threshold=state["marginal_threshold"],
        ),
        overwrite=False,
    )


def run_desktop(args, asset):
    module = native_module("_ftgspp_viewer")
    app = module.Desktop(
        shader_dir(),
        args.device,
        args.width or 1280,
        args.height or 800,
        not args.no_vsync,
    )
    output = args.output or asset.model_path.parent / "viewer_output"
    session = args.session or asset.model_path.parent / "viewer_session.json"
    path_file = args.camera_path or asset.model_path.parent / "viewer_path.json"
    last_report = time.monotonic()
    try:
        app.load_scene(asset.model.arrays(), asset.metadata)
        asset.model = None  # Do not retain a second host copy during normal browsing.
        if session.is_file() and not args.no_session:
            try:
                app.restore_state(load_session(session, asset))
            except (ValueError, KeyError, OSError) as exc:
                app.status = f"Session ignored: {exc}"
        if args.camera is not None:
            app.restore_state(
                dict(
                    camera=select_camera(asset, args),
                    selected_camera=args.camera,
                    lock_camera=True,
                )
            )
        app.restore_state(options(args))
        if args.camera_path:
            data = load_path(path_file)
            app.restore_state(
                dict(
                    bookmarks=[k["camera"] for k in data["keyframes"]],
                    path_times=[k["path_time"] for k in data["keyframes"]],
                )
            )
        while not app.should_close():
            ev = app.frame()
            try:
                if ev["open"]:
                    src = (
                        asset.model_path.parent
                        if ev["open"] == "@reload"
                        else Path(ev["open"])
                    )
                    next_asset = load_asset(src)
                    app.load_scene(next_asset.model.arrays(), next_asset.metadata)
                    asset = next_asset
                    asset.model = None
                    if not args.session:
                        session = asset.model_path.parent / "viewer_session.json"
                    if not args.output:
                        output = asset.model_path.parent / "viewer_output"
                    if not args.camera_path:
                        path_file = asset.model_path.parent / "viewer_path.json"
                if ev["screenshot"]:
                    file = unique_path(output, "capture", ".png")
                    screenshot(app, asset, file)
                    app.status = f"Saved {file}"
                if ev["save_session"]:
                    save_session(session, asset, app.state())
                    app.status = f"Saved {session}"
                if ev["save_path"]:
                    save_path(path_file, from_state(app.state()))
                    app.status = f"Saved {path_file}"
                if ev["load_path"]:
                    data = load_path(path_file)
                    app.restore_state(
                        dict(
                            bookmarks=[k["camera"] for k in data["keyframes"]],
                            path_times=[k["path_time"] for k in data["keyframes"]],
                        )
                    )
                    app.status = f"Loaded {path_file}"
                if ev["export_path"]:
                    state = app.state()
                    file = unique_path(output, "path")
                    export_frames(
                        asset,
                        from_state(state),
                        file,
                        fps=asset.metadata["fps"],
                        display=display_options(state),
                        desktop=app,
                    )
                    app.status = f"Exported PNG sequence: {file}"
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                app.status = f"Error: {exc}"
                print(app.status, flush=True)
            if args.frames and app.state()["frames"] >= args.frames:
                break
            if args.frames and time.monotonic() - last_report > 20:
                print(f"Viewer frame {app.state()['frames']}/{args.frames}", flush=True)
                last_report = time.monotonic()
        if args.screenshot:
            screenshot(app, asset, args.screenshot)
        if args.stats_json:
            atomic_json(args.stats_json, app.state())
        if not args.no_session:
            save_session(session, asset, app.state())
    finally:
        app.close()


def main(argv=None, *, command="viewer"):
    p = parser_for(command)
    args = p.parse_args(argv)
    try:
        if args.device < 0:
            raise ValueError("Device must be a non-negative Vulkan ID")
        for name in ("width", "height"):
            n = getattr(args, name)
            if n is not None and not 1 <= n <= 16384:
                raise ValueError(f"{name} must be 1..16384")
        if command == "viewer" and (
            (not args.remote and args.width is not None and args.width < 640)
            or (not args.remote and args.height is not None and args.height < 360)
            or (not args.remote and args.frames < 0)
        ):
            raise ValueError(
                "Viewer window must be at least 640x360; frames must be non-negative"
            )
        if command == "viewer" and args.remote and (
            args.no_vsync or args.frames or args.screenshot or args.stats_json
        ):
            raise ValueError(
                "--no-vsync, --frames, --screenshot and --stats-json are desktop-only; "
                "use the remote UI for PNG captures"
            )
        asset = load_asset(
            args.run or args.model,
            time_range=args.time_range,
            fps=args.fps,
            sh_degree=args.sh_degree,
            marginal_threshold=args.marginal_threshold,
        )
        if command == "viewer":
            if args.remote:
                from .remote import run_remote
                run_remote(args, asset)
            else:
                run_desktop(args, asset)
        else:
            state = dict(
                mode=0,
                background=[0, 0, 0],
                checker=False,
                exposure=0.0,
                transparent=False,
            )
            state.update(options(args))
            path_data = (
                load_path(args.camera_path)
                if args.camera_path
                else dict(
                    schema=PATH_SCHEMA,
                    keyframes=[dict(path_time=0.0, camera=select_camera(asset, args))],
                )
            )
            report = export_frames(
                asset,
                path_data,
                args.output,
                device=args.device,
                width=args.width,
                height=args.height,
                fps=args.fps or asset.metadata["fps"],
                duration=args.duration,
                display=display_options(state),
                video=args.video,
                progress=lambda i, n: (
                    print(f"Rendered {i}/{n}", flush=True)
                    if i == n or i % 30 == 0
                    else None
                ),
            )
            print(f"Saved {report['frame_count']} full-quality frames to {args.output}")
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        p.exit(2, f"error: {exc}\n")


def prepare_main(argv=None):
    p = argparse.ArgumentParser(prog="ftgspp prepare viewer")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--force", action="store_true", help="replace existing viewer_manifest.json"
    )
    args = p.parse_args(argv)
    from ftgspp.io.viewer_manifest import write_manifest

    try:
        run = args.run.expanduser().resolve()
        manifest = run / "viewer_manifest.json"
        if manifest.exists() and not args.force:
            raise FileExistsError(f"{manifest} exists; use --force to replace")
        # Read the raw model without accepting a stale manifest during regeneration.
        import numpy as np
        from ftgspp.io.model import NativeGaussians
        from ftgspp.io.hashing import file_sha256
        from .asset import inferred_metadata

        model = run / "gaussians.npz"
        with np.load(model, allow_pickle=False) as z:
            meta = inferred_metadata(NativeGaussians.from_mapping(z))
        meta.update(legacy_metadata(run))
        write_manifest(model, meta, file_sha256(model))
        print(f"Saved {manifest}")
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        p.exit(2, f"error: {exc}\n")
