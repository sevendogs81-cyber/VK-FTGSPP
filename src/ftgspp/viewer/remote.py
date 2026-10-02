"""Single-client, display-free browser viewer over a loopback WebSocket."""

from __future__ import annotations

import asyncio
import json
import math
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from .asset import atomic_bytes, fit_camera_extent, json_value, scaled_camera
from .offline import display_options, native_module, shader_dir
from .path import CameraPath, from_state, load_path, save_path
from .session import load_session, save_session, validate_state

STATE_FIELDS = frozenset({
    "camera", "time", "speed", "playing", "loop", "snap", "lock_camera",
    "selected_camera", "mode", "background", "checker", "exposure",
    "transparent", "sh_degree", "marginal_threshold", "interactive",
    "target_fps", "bookmarks", "path_times", "world_up",
})

MAX_VIDEO_FRAMES = 100_000
MAX_RECORDED_KEYS = 10_000
VIDEO_ENCODER_PRESET = "veryfast"
VIDEO_ENCODER_THREADS = 4


class VideoExportCancelled(RuntimeError):
    pass


def _bounded_frame_count(duration, fps, *, label):
    """Return a deterministic frame count without allowing numeric overflow."""

    duration, fps = float(duration), float(fps)
    if not math.isfinite(duration) or duration < 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid {label.lower()} duration or FPS")
    samples = duration * fps
    if not math.isfinite(samples) or samples - 1e-9 > MAX_VIDEO_FRAMES:
        raise ValueError(f"{label} exceeds {MAX_VIDEO_FRAMES:,} frames")
    count = max(1, math.ceil(samples - 1e-9))
    if count > MAX_VIDEO_FRAMES:
        raise ValueError(f"{label} exceeds {MAX_VIDEO_FRAMES:,} frames")
    return count


def remote_state(state):
    # Desktop sessions also carry window overlays and runtime counters. These
    # must not round-trip into the browser's control messages.
    state = validate_state({k: v for k, v in state.items() if k in STATE_FIELDS})
    if len(state.get("bookmarks", [])) > 256:
        raise ValueError("Too many path keyframes")
    # Retain the field for old sessions/clients, but aspect is always fixed.
    return dict(state, lock_camera=True)


def wire(value):
    return json.loads(json.dumps(value, default=json_value, allow_nan=False))


def initial_state(asset, args):
    from .app import options, select_camera

    meta = asset.metadata
    c = select_camera(asset, args)
    state = dict(
        camera=c, time=c["timestamp"], speed=1.0, playing=False, loop=True,
        snap=False, lock_camera=True,
        selected_camera=args.camera if args.camera is not None else -1,
        mode=0, background=[0, 0, 0], checker=False, exposure=0.0,
        transparent=False, sh_degree=meta["sh_degree"],
        marginal_threshold=meta["marginal_threshold"],
        interactive=args.quality == "interactive", target_fps=args.target_fps or 30.0,
        bookmarks=[], path_times=[], world_up=meta["world_up"],
    )
    session = args.session or asset.model_path.parent / "viewer_session.json"
    if session.is_file() and not args.no_session:
        try:
            state.update(remote_state(load_session(session, asset)))
        except (ValueError, KeyError, OSError, TypeError, OverflowError) as exc:
            print(f"Remote viewer ignored session: {exc}", flush=True)
    if args.camera is not None:
        state.update(camera=c, time=c["timestamp"], selected_camera=args.camera, lock_camera=True)
    state.update(options(args))
    if args.camera_path:
        path = load_path(args.camera_path)
        state.update(
            bookmarks=[k["camera"] for k in path["keyframes"]],
            path_times=[k["path_time"] for k in path["keyframes"]],
        )
    if not -1 <= state["selected_camera"] < len(meta["cameras"]):
        state["selected_camera"] = -1
    return remote_state(state)


class RemoteViewer:
    def __init__(self, args, asset):
        from .app import default_export_root

        self.args = args
        self.asset = asset
        self.state = initial_state(asset, args)
        self.output = args.output or default_export_root(asset)
        self.session = args.session or asset.model_path.parent / "viewer_session.json"
        self.path_file = args.camera_path or asset.model_path.parent / "viewer_path.json"
        self.token = secrets.token_urlsafe(32)
        self.client = None
        self.renderer = None
        self.render_width = args.width or min(1280, self.state["camera"]["width"])
        self.render_height = args.height or min(720, self.state["camera"]["height"])
        if (self.render_width > 3840 or self.render_height > 3840
                or self.render_width * self.render_height > 3840 * 2160):
            raise ValueError("Remote render area exceeds 3840x2160")
        self.capture_width = self.render_width
        self.capture_height = self.render_height
        # F/T stays on GPU; one reusable CPU RGBA8 image and one encoded image.
        self._rgba = None
        self._forward_key = self._encoded_key = self._encoded = None
        self.last_timings = {}
        self.exports: dict[str, Path] = {}
        self.recording_started = None
        self.recording_keys = []
        self.recorded_path = None
        self.catalog: list[Path] = []

    def start_renderer(self):
        self._rgba = None
        self._forward_key = self._encoded_key = self._encoded = None
        self.renderer = native_module("_ftgspp_vk").ViewRenderer(shader_dir(), self.args.device)
        try:
            self.renderer.upload(self.asset.model.arrays())
            self.asset.model = None
        except BaseException:
            self.renderer.close()
            self.renderer = None
            raise

    def render_image(self, state, width, height, png=False):
        start = time.perf_counter()
        width, height = fit_camera_extent(state["camera"], width, height)
        c = scaled_camera(state["camera"], width, height)
        key = (
            width, height, c["K"].tobytes(), c["w2c"].tobytes(),
            float(state["time"]), int(state["sh_degree"]), float(state["marginal_threshold"]),
        )
        timing = dict(forward_ms=0.0, readback_ms=0.0, compose_ms=0.0, encode_ms=0.0,
                      rgba8_ms=0.0, readback_bytes=0,
                      forward_cached=key == self._forward_key, encoded_cached=False)
        settings = display_options(state)
        if not png:
            settings["transparent"] = False
        display_key = (
            key, png, settings.get("mode", 0), tuple(settings.get("background", (0, 0, 0))),
            settings.get("checker", False), settings.get("exposure", 0),
            settings.get("transparent", False),
        )
        if display_key == self._encoded_key:
            timing["encoded_cached"] = True
            self.last_timings = timing
            return self._encoded, (time.perf_counter() - start) * 1000
        if key != self._forward_key:
            self._forward_key = self._encoded_key = None
            mark = time.perf_counter()
            self.renderer.render(c, state["time"], state["sh_degree"], state["marginal_threshold"])
            timing["forward_ms"] = (time.perf_counter() - mark) * 1000
        mark = time.perf_counter()
        shape = (height, width, 4)
        if self._rgba is None or self._rgba.shape != shape:
            self._rgba = np.empty(shape, dtype=np.uint8)
        try:
            self.renderer.readback_rgba8_into(self._rgba, **settings)
        except Exception:
            self._forward_key = self._encoded_key = None
            raise
        self._forward_key = key
        # Composition and D2H share one GPU submission/wait; do not invent
        # separate GPU durations from a single host-observed interval.
        timing["rgba8_ms"] = (time.perf_counter() - mark) * 1000
        timing["readback_bytes"] = width * height * 4 + 4
        mark = time.perf_counter()
        image = cv2.cvtColor(self._rgba, cv2.COLOR_RGBA2BGRA if settings.get("transparent")
                            else cv2.COLOR_RGBA2BGR)
        ok, encoded = cv2.imencode(
            ".png" if png else ".jpg", image,
            [] if png else [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        if not ok:
            raise RuntimeError("Remote image encoding failed")
        self._encoded = encoded.tobytes()
        self._encoded_key = display_key
        timing["encode_ms"] = (time.perf_counter() - mark) * 1000
        self.last_timings = timing
        elapsed = (time.perf_counter() - start) * 1000
        return self._encoded, elapsed

    async def send(self, ws, message):
        await ws.send(json.dumps(message, default=json_value, separators=(",", ":"), allow_nan=False))

    def recording_summary(self):
        keys = self.recording_keys if self.recording_started is not None else (
            self.recorded_path or {}
        ).get("keyframes", [])
        duration = float(keys[-1]["path_time"]) if keys else 0.0
        return dict(
            active=self.recording_started is not None,
            sample_count=len(keys), duration=duration,
        )

    def record_interaction_sample(self):
        if self.recording_started is None:
            return
        if len(self.recording_keys) >= MAX_RECORDED_KEYS:
            raise ValueError(f"Interaction recording exceeds {MAX_RECORDED_KEYS:,} samples")
        elapsed = 0.0 if not self.recording_keys else max(
            0.0, time.monotonic() - self.recording_started
        )
        if self.recording_keys:
            elapsed = max(elapsed, self.recording_keys[-1]["path_time"] + 1e-6)
        camera = wire(dict(self.state["camera"], timestamp=float(self.state["time"])))
        self.recording_keys.append(dict(path_time=elapsed, camera=camera))

    def start_interaction_recording(self):
        if self.recording_started is not None:
            raise ValueError("Interaction recording is already active")
        self.recording_keys = []
        self.recorded_path = None
        self.recording_started = time.monotonic()
        self.record_interaction_sample()
        return self.recording_summary()

    def stop_interaction_recording(self):
        if self.recording_started is None:
            raise ValueError("Interaction recording is not active")
        # Hitting the cap must still leave a recording that can be stopped and
        # exported.  The update that reached the cap already captured the most
        # recent accepted state.
        if len(self.recording_keys) < MAX_RECORDED_KEYS:
            self.record_interaction_sample()
        self.recording_started = None
        if len(self.recording_keys) < 2:
            raise ValueError("Interaction recording needs at least two samples")
        self.recorded_path = dict(
            schema="ftgspp-camera-path-v1", keyframes=list(self.recording_keys),
        )
        return self.recording_summary()

    def clear_interaction_recording(self):
        if self.recording_started is not None:
            raise ValueError("Stop interaction recording before clearing it")
        self.recording_keys = []
        self.recorded_path = None
        return self.recording_summary()

    def apply_update(self, message):
        updates = message.get("state", {})
        if not isinstance(updates, dict) or set(updates) - STATE_FIELDS:
            raise ValueError("Unknown or invalid state fields")
        updates = dict(updates)
        if "selected_camera" in updates:
            i = updates["selected_camera"]
            if type(i) is not int or not -1 <= i < len(self.asset.metadata["cameras"]):
                raise ValueError("Capture camera index outside available cameras")
        if "bookmarks" in updates or "path_times" in updates:
            updates.setdefault("bookmarks", self.state["bookmarks"])
            updates.setdefault("path_times", self.state["path_times"])
            if not isinstance(updates["bookmarks"], list):
                raise ValueError("Bookmarks must be an array")
            if len(updates["bookmarks"]) > 256:
                raise ValueError("Too many path keyframes")
        # Validate only modified values. Playback must not revalidate hundreds
        # of already validated path cameras on each frame.
        state = dict(self.state, **validate_state(updates))
        state["lock_camera"] = True
        if state["transparent"] and state["mode"] != 0:
            raise ValueError("Transparent output requires RGB mode")
        if len(state.get("bookmarks", [])) > 256:
            raise ValueError("Too many path keyframes")
        dimensions = {}
        for key in ("width", "height", "fullWidth", "fullHeight"):
            if key in message:
                value = message[key]
                if type(value) is not int or not 1 <= value <= 3840:
                    raise ValueError(f"Invalid render {key}")
                dimensions[key] = value
        rw = dimensions.get("width", self.render_width)
        rh = dimensions.get("height", self.render_height)
        cw = dimensions.get("fullWidth", self.capture_width)
        ch = dimensions.get("fullHeight", self.capture_height)
        if rw * rh > 3840 * 2160 or cw * ch > 3840 * 2160:
            raise ValueError("Render area exceeds 3840x2160")
        self.state = state
        self.render_width, self.render_height = rw, rh
        self.capture_width, self.capture_height = cw, ch

    def hello_message(self):
        current = self.asset.model_path.parent.name
        return dict(
            type="hello", metadata=self.asset.metadata, state=self.state,
            model_sha256=self.asset.sha256, recording=self.recording_summary(),
            output_path=str(self.output.resolve()),
            scenes=[path.name for path in self.catalog],
            scene=current if self.catalog else None,
        )

    def activate_scene(self, name: str) -> None:
        matches = [path for path in self.catalog if path.name == name]
        if len(matches) != 1:
            raise ValueError(f"Unknown scene: {name}")
        target = matches[0]
        if target == self.asset.model_path.parent.resolve():
            return
        if not self.args.no_session:
            try:
                save_session(self.session, self.asset, self.state)
            except OSError as exc:
                print(f"Remote viewer could not save session: {exc}", flush=True)
        from .app import default_export_root
        from .asset import load_asset

        asset = load_asset(
            target,
            time_range=self.args.time_range,
            fps=self.args.fps,
            sh_degree=self.args.sh_degree,
            marginal_threshold=self.args.marginal_threshold,
        )
        # Prepare every fallible piece before replacing the resident GPU model.
        # This keeps the old asset, renderer, and browser state consistent when
        # a scene has an incompatible camera/session/path configuration.
        state = initial_state(asset, self.args)
        output = self.args.output or default_export_root(asset)
        session = self.args.session or asset.model_path.parent / "viewer_session.json"
        path_file = self.args.camera_path or asset.model_path.parent / "viewer_path.json"
        # Native upload releases the old buffers before allocating the new ones.
        # Invalidate cached pixels even when upload fails, then reload the old
        # model on failure without keeping two models resident during browsing.
        self._rgba = None
        self._forward_key = self._encoded_key = self._encoded = None
        try:
            self.renderer.upload(asset.model.arrays())
        except Exception as exc:
            asset.model = None
            try:
                previous = load_asset(self.asset.model_path)
                if previous.sha256 != self.asset.sha256:
                    raise ValueError("Previous model changed on disk")
                self.renderer.upload(previous.model.arrays())
            except Exception as recovery:
                raise RuntimeError(
                    f"Scene upload failed ({exc}); restoring the previous model "
                    f"also failed ({recovery}). Reload a scene or restart the viewer."
                ) from exc
            raise RuntimeError(f"Scene upload failed ({exc}); previous model restored") from exc
        asset.model = None
        self.asset = asset
        self.state = state
        self.output = output
        self.session = session
        self.path_file = path_file
        self.recording_started = None
        self.recording_keys = []
        self.recorded_path = None

    async def serve_client(self, ws):
        if self.client is not None:
            await ws.close(code=1013, reason="One viewer client at a time")
            return
        self.client = ws
        wake = asyncio.Event()
        pending = {
            "render": True, "capture": False, "export": False, "video": False,
            "switch": None,
        }
        stopping = False
        cancelled = threading.Event()
        export_active = False
        export_cancel = None
        switch_active = False
        wake.set()

        async def worker():
            nonlocal export_active, export_cancel, switch_active
            while True:
                await wake.wait()
                wake.clear()
                if stopping:
                    return
                capture = pending["capture"]
                export = pending["export"]
                video = pending["video"]
                render = pending["render"]
                switch = pending["switch"]
                pending.update(
                    render=False, capture=False, export=False, video=False, switch=None,
                )
                state = dict(self.state)
                render_width, render_height = self.render_width, self.render_height
                if switch:
                    switch_active = True
                    try:
                        try:
                            await asyncio.to_thread(self.activate_scene, switch)
                        except Exception as exc:
                            if stopping:
                                return
                            await self.send(ws, dict(
                                type="error", action="switch_scene", message=str(exc),
                            ))
                        if stopping:
                            return
                        await self.send(ws, self.hello_message())
                        state = dict(self.state)
                        render_width, render_height = self.render_width, self.render_height
                        render = True
                    finally:
                        switch_active = False
                try:
                    if video:
                        export_state, plan, job_cancel = video
                        export_active = True
                        loop = asyncio.get_running_loop()
                        progress_pending = None

                        def progress(completed, total, elapsed):
                            nonlocal progress_pending
                            if stopping or (progress_pending is not None and not progress_pending.done()):
                                return
                            progress_pending = asyncio.run_coroutine_threadsafe(self.send(ws, dict(
                                type="export_progress", completed=completed,
                                total=total, elapsed=elapsed,
                            )), loop)

                        try:
                            result = await asyncio.to_thread(
                                self.export_video, export_state, plan,
                                lambda: cancelled.is_set() or job_cancel.is_set(), progress,
                            )
                            if stopping:
                                return
                            self.exports[result.name] = result
                            await self.send(ws, dict(
                                type="saved", action="video", path=str(result),
                                file=result.name, url=f"/exports/{result.name}",
                            ))
                        except VideoExportCancelled:
                            if not stopping:
                                await self.send(ws, dict(type="export_cancelled"))
                        except Exception as exc:
                            if not stopping:
                                await self.send(ws, dict(
                                    type="error", action="export_video", message=str(exc),
                                ))
                        finally:
                            export_active = False
                            export_cancel = None
                    if export:
                        export_active = True
                        try:
                            result = await asyncio.to_thread(self.export_path, *export, cancelled)
                            if stopping:
                                return
                            await self.send(ws, dict(type="saved", action="export", path=str(result)))
                        except Exception as exc:
                            if not stopping:
                                await self.send(ws, dict(
                                    type="error", action="export_path", message=str(exc),
                                ))
                        finally:
                            export_active = False
                    if capture:
                        capture_state, width, height = capture
                        width, height = fit_camera_extent(capture_state["camera"], width, height)
                        data, elapsed = await asyncio.to_thread(
                            self.render_image, capture_state, width, height, True
                        )
                        if stopping:
                            return
                        from .app import unique_path
                        from .asset import atomic_json
                        target = unique_path(self.output, "capture", ".png")
                        atomic_bytes(target, data)
                        atomic_json(target.with_suffix(".json"), dict(
                            schema="ftgspp-capture-v1", model_sha256=self.asset.sha256,
                            camera=dict(
                                scaled_camera(capture_state["camera"], width, height),
                                timestamp=capture_state["time"],
                            ),
                            display=display_options(capture_state), sh_degree=capture_state["sh_degree"],
                            marginal_threshold=capture_state["marginal_threshold"],
                        ), overwrite=False)
                        await self.send(ws, dict(type="capture", path=str(target), ms=elapsed))
                        await ws.send(b"\x02" + data)
                    if render:
                        width, height = fit_camera_extent(state["camera"], render_width, render_height)
                        data, elapsed = await asyncio.to_thread(
                            self.render_image, state, width, height
                        )
                        if stopping:
                            return
                        await self.send(ws, dict(
                            type="frame", width=width, height=height, ms=elapsed,
                            bytes=len(data), stats=self.renderer.stats(), pipeline=self.last_timings,
                        ))
                        await ws.send(b"\x01" + data)
                except Exception as exc:
                    if stopping:
                        return
                    await self.send(ws, dict(type="error", action="render", message=str(exc)))

        job = asyncio.create_task(worker())
        try:
            await self.send(ws, self.hello_message())
            async for raw in ws:
                action = None
                try:
                    if not isinstance(raw, str):
                        raise ValueError("Expected JSON control message")
                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        raise ValueError("Expected a JSON object")
                    action = message.get("type")
                    if message.get("model_sha256", self.asset.sha256) != self.asset.sha256:
                        if action == "update":
                            continue
                        raise ValueError("Command belongs to a previous scene")
                    if switch_active or pending["switch"] is not None:
                        # Browser updates describe the old scene and will be
                        # superseded by the new hello/state.  Other commands
                        # must be retried explicitly against the new scene.
                        if action == "update":
                            continue
                        raise ValueError("Wait for the scene switch to finish")
                    if action in (
                        "capture", "save_session", "save_path", "sample_path",
                        "export_path", "export_video", "start_recording", "stop_recording",
                    ) and "state" in message:
                        self.apply_update(message)
                    if action == "update":
                        self.apply_update(message)
                        if self.recording_started is not None:
                            self.record_interaction_sample()
                            if len(self.recording_keys) == MAX_RECORDED_KEYS:
                                await self.send(ws, dict(
                                    type="recording_stopped", **self.stop_interaction_recording(),
                                ))
                        pending["render"] = True
                        wake.set()
                    elif action == "capture":
                        pending["capture"] = (dict(self.state), self.capture_width, self.capture_height)
                        wake.set()
                    elif action == "save_session":
                        save_session(self.session, self.asset, self.state)
                        await self.send(ws, dict(type="saved", action="session", path=str(self.session)))
                    elif action == "load_session":
                        self.apply_update(dict(state=remote_state(load_session(self.session, self.asset))))
                        await self.send(ws, dict(type="state", state=self.state))
                        pending["render"] = True
                        wake.set()
                    elif action == "save_path":
                        save_path(self.path_file, from_state(self.state))
                        await self.send(ws, dict(type="saved", action="path", path=str(self.path_file)))
                    elif action == "load_path":
                        path = load_path(self.path_file)
                        self.apply_update(dict(state=dict(
                            bookmarks=[k["camera"] for k in path["keyframes"]],
                            path_times=[k["path_time"] for k in path["keyframes"]],
                        )))
                        await self.send(ws, dict(type="state", state=self.state))
                    elif action == "sample_path":
                        path = CameraPath(from_state(self.state))
                        seconds = float(message["seconds"])
                        c = path.sample(seconds)
                        scene_mode = message.get("scene_mode", "track")
                        if scene_mode == "freeze":
                            scene_time = float(message["scene_time"])
                            c = dict(c, timestamp=scene_time)
                        elif scene_mode == "range":
                            scene_start = float(message["time_start"])
                            scene_end = float(message["time_end"])
                            if scene_end <= scene_start:
                                raise ValueError("Scene end time must be greater than start time")
                            fraction = 0.0 if path.duration <= 0 else max(0.0, min(1.0, (
                                seconds - path.times[0]
                            ) / path.duration))
                            c = dict(c, timestamp=(
                                scene_start + fraction * (scene_end - scene_start)
                            ))
                        elif scene_mode != "track":
                            raise ValueError("Unknown camera-path scene mode")
                        timestamp = float(c["timestamp"])
                        lo = float(self.asset.metadata.get("time_start", timestamp))
                        hi = float(self.asset.metadata.get("time_end", timestamp))
                        if not math.isfinite(timestamp) or not lo <= timestamp <= hi:
                            raise ValueError(f"Scene time must stay within {lo:g}..{hi:g}")
                        self.state.update(
                            camera=c, time=c["timestamp"], playing=False,
                            selected_camera=-1, lock_camera=True,
                        )
                        await self.send(ws, dict(type="state", state=self.state))
                        pending["render"] = True
                        wake.set()
                    elif action == "start_recording":
                        if export_active or pending["export"] or pending["video"]:
                            raise ValueError("Cannot record interaction during export")
                        summary = self.start_interaction_recording()
                        await self.send(ws, dict(type="recording_started", **summary))
                    elif action == "stop_recording":
                        summary = self.stop_interaction_recording()
                        await self.send(ws, dict(type="recording_stopped", **summary))
                    elif action == "clear_recording":
                        if export_active or pending["export"] or pending["video"]:
                            raise ValueError("Cannot clear interaction recording during export")
                        summary = self.clear_interaction_recording()
                        await self.send(ws, dict(type="recording_cleared", **summary))
                    elif action == "export_path":
                        if export_active or pending["export"] or pending["video"]:
                            raise ValueError("An export is already in progress")
                        pending["export"] = (dict(self.state), self.capture_width, self.capture_height)
                        wake.set()
                    elif action == "export_video":
                        if export_active or pending["export"] or pending["video"]:
                            raise ValueError("An export is already in progress")
                        export_state = wire(self.state)
                        plan = self.prepare_video_export(export_state, message.get("export"))
                        export_cancel = threading.Event()
                        pending["video"] = (export_state, plan, export_cancel)
                        await self.send(ws, dict(type="export_started", **{
                            key: plan[key] for key in (
                                "mode", "fps", "width", "height", "duration", "frame_count",
                                "scene_mode",
                            )
                        }))
                        wake.set()
                    elif action == "switch_scene":
                        if not self.catalog:
                            raise ValueError("This viewer has no scene catalog")
                        if export_active or pending["video"] or pending["export"]:
                            raise ValueError(
                                "Wait for the current export to finish before switching scenes"
                            )
                        if pending["capture"]:
                            raise ValueError(
                                "Wait for the pending screenshot before switching scenes"
                            )
                        if self.recording_started is not None:
                            raise ValueError(
                                "Stop the interaction recording before switching scenes"
                            )
                        name = message.get("scene")
                        if not isinstance(name, str) or not name:
                            raise ValueError("Scene name required")
                        pending["switch"] = name
                        wake.set()
                    elif action == "cancel_export":
                        if export_cancel is None:
                            raise ValueError("No video export is in progress")
                        export_cancel.set()
                        await self.send(ws, dict(type="export_cancel_requested"))
                    else:
                        raise ValueError("Unknown viewer action")
                except (ValueError, KeyError, OSError, TypeError, OverflowError) as exc:
                    await self.send(ws, dict(type="error", action=action, message=str(exc)))
        finally:
            stopping = True
            cancelled.set()
            if export_cancel is not None:
                export_cancel.set()
            if self.recording_started is not None:
                try:
                    self.stop_interaction_recording()
                except ValueError:
                    self.recording_started = None
            wake.set()
            try:
                await job
            except Exception as exc:
                print(f"Remote viewer client ended: {exc}", flush=True)
            self.client = None
            if not self.args.no_session:
                try:
                    save_session(self.session, self.asset, self.state)
                except OSError as exc:
                    print(f"Remote viewer could not save session: {exc}", flush=True)

    def export_path(self, state, width, height, cancelled=None):
        from .app import unique_path
        from .asset import atomic_json
        path = CameraPath(from_state(state))
        width, height = fit_camera_extent(path.cameras[0], width, height)
        fps = self.args.fps or self.asset.metadata["fps"]
        count = _bounded_frame_count(path.duration, fps, label="Path export")
        target = unique_path(self.output, "path")
        temporary = Path(tempfile.mkdtemp(prefix=".path_", dir=self.output))
        try:
            frames = []
            for i in range(count):
                if cancelled is not None and cancelled.is_set():
                    raise RuntimeError("Path export cancelled after client disconnect")
                c = path.sample(path.times[0] + i / fps, width, height)
                current = dict(state, camera=c, time=c["timestamp"])
                data, _ = self.render_image(current, width, height, True)
                name = f"{i:06d}.png"
                (temporary / name).write_bytes(data)
                frames.append(dict(file=name, path_time=i / fps, camera=c))
            atomic_json(temporary / "render.json", dict(
                schema="ftgspp-render-v1", model_sha256=self.asset.sha256,
                fps=fps, frame_count=count, sh_degree=state["sh_degree"],
                marginal_threshold=state["marginal_threshold"],
                display=display_options(state), frames=frames,
            ))
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("Path export cancelled after client disconnect")
            temporary.rename(target)
            return target
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def prepare_video_export(self, state, options):
        if not isinstance(options, dict):
            raise ValueError("Missing video export options")
        mode = options.get("mode")
        if mode not in ("recording", "path", "view"):
            raise ValueError("Video mode must be recording, path, or view")
        fps = options.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise ValueError("Video FPS must be a number")
        fps = float(fps)
        if not math.isfinite(fps) or not 1 <= fps <= 240:
            raise ValueError("Video FPS must be between 1 and 240")
        width, height = options.get("width"), options.get("height")
        if type(width) is not int or type(height) is not int:
            raise ValueError("Video width and height must be integers")
        if not 1 <= width <= 3840 or not 1 <= height <= 3840:
            raise ValueError("Video width and height must be 1..3840")
        if width * height > 3840 * 2160:
            raise ValueError("Video area exceeds 3840x2160")

        scene_mode = "track"
        scene_start = scene_end = None
        if mode == "recording":
            if self.recording_started is not None:
                raise ValueError("Stop interaction recording before exporting it")
            if self.recorded_path is None:
                raise ValueError("Record viewer interaction before exporting it")
            path = CameraPath(wire(self.recorded_path))
            if path.duration <= 0:
                raise ValueError("Recorded interaction has no duration")
        elif mode == "path":
            if len(state.get("bookmarks", [])) < 2:
                raise ValueError("Camera-path video needs at least two timed keyframes")
            path = CameraPath(from_state(state))
            if len(path.cameras) < 2 or path.duration <= 0:
                raise ValueError("Camera-path video needs at least two timed keyframes")
            scene_mode = options.get("scene_mode")
            if scene_mode == "freeze":
                scene_start = scene_end = options.get("scene_time")
            elif scene_mode == "range":
                scene_start, scene_end = options.get("time_start"), options.get("time_end")
            else:
                raise ValueError("Camera path scene mode must be freeze or range")
        else:
            scene_start, scene_end = options.get("time_start"), options.get("time_end")
            scene_mode = "range"

        if scene_mode in ("freeze", "range"):
            if isinstance(scene_start, bool) or isinstance(scene_end, bool):
                raise ValueError("Scene time range must be numeric")
            try:
                scene_start, scene_end = float(scene_start), float(scene_end)
            except (TypeError, ValueError) as exc:
                raise ValueError("Scene time range must be numeric") from exc
            if not math.isfinite(scene_start) or not math.isfinite(scene_end):
                raise ValueError("Scene time range must be finite")
            if scene_mode == "range" and scene_end <= scene_start:
                raise ValueError("Scene end time must be greater than start time")
            lo = float(self.asset.metadata.get("time_start", scene_start))
            hi = float(self.asset.metadata.get("time_end", scene_end))
            if scene_start < lo or scene_end > hi:
                raise ValueError(f"Scene time range must stay within {lo:g}..{hi:g}")
        if mode == "view":
            first = wire(dict(state["camera"], timestamp=scene_start))
            last = wire(dict(state["camera"], timestamp=scene_end))
            path = CameraPath(dict(
                schema="ftgspp-camera-path-v1",
                keyframes=[
                    dict(path_time=0.0, camera=first),
                    dict(path_time=scene_end - scene_start, camera=last),
                ],
            ))

        width, height = fit_camera_extent(path.cameras[0], width, height)
        count = _bounded_frame_count(path.duration, fps, label="Video export")
        return dict(
            mode=mode, path=path, fps=fps, width=width, height=height,
            duration=path.duration, frame_count=count, scene_mode=scene_mode,
            scene_start=scene_start, scene_end=scene_end,
        )

    def export_video(self, state, plan, is_cancelled=lambda: False, progress=None):
        from .app import unique_path
        from .asset import atomic_json

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Video export requires ffmpeg on the server")
        target = unique_path(self.output, "video", ".mp4")
        temporary = target.with_name(target.stem + ".tmp.mp4")
        path = plan["path"]
        fps, width, height = plan["fps"], plan["width"], plan["height"]
        count = plan["frame_count"]
        command = [
            ffmpeg, "-nostdin", "-v", "error", "-n",
            "-f", "rawvideo", "-pixel_format", "rgba",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", VIDEO_ENCODER_PRESET,
            "-tune", "zerolatency", "-crf", "18",
            "-threads", str(VIDEO_ENCODER_THREADS),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
        ]
        # A PIPE that is read only after all frames have been written can fill
        # with encoder diagnostics and deadlock both processes. Spool stderr,
        # then retain only its tail in the error shown to the client.
        errors = tempfile.TemporaryFile()
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors,
            )
        except BaseException:
            errors.close()
            raise
        frames = []
        started = last_progress = time.monotonic()
        settings = display_options(state)
        settings["transparent"] = False
        self._forward_key = self._encoded_key = self._encoded = None

        def encoder_result():
            code = process.wait()
            errors.seek(0, os.SEEK_END)
            errors.seek(max(0, errors.tell() - 65536))
            return code, errors.read().decode(errors="replace").strip()

        def encoder_failure(default):
            code, message = encoder_result()
            detail = message or default
            return RuntimeError(f"ffmpeg failed ({code}): {detail}")

        try:
            assert process.stdin is not None
            for i in range(count):
                if is_cancelled():
                    raise VideoExportCancelled("Video export cancelled")
                path_time = path.times[0] + i / fps
                c = path.sample(path_time, width, height)
                if plan["scene_mode"] == "freeze":
                    c = dict(c, timestamp=plan["scene_start"])
                elif plan["scene_mode"] == "range":
                    fraction = (path_time - path.times[0]) / path.duration
                    c = dict(c, timestamp=(
                        plan["scene_start"]
                        + fraction * (plan["scene_end"] - plan["scene_start"])
                    ))
                self.renderer.render(
                    c, c["timestamp"], state["sh_degree"], state["marginal_threshold"]
                )
                shape = (height, width, 4)
                if self._rgba is None or self._rgba.shape != shape:
                    self._rgba = np.empty(shape, dtype=np.uint8)
                self.renderer.readback_rgba8_into(self._rgba, **settings)
                try:
                    process.stdin.write(memoryview(self._rgba).cast("B"))
                except BrokenPipeError as exc:
                    raise encoder_failure(
                        "encoder stopped while receiving raw video frames"
                    ) from exc
                frames.append(dict(index=i, path_time=i / fps, camera=c))
                now = time.monotonic()
                if progress is not None and (
                    i == 0 or i + 1 == count or now - last_progress >= 0.2
                ):
                    progress(i + 1, count, now - started)
                    last_progress = now
            try:
                process.stdin.close()
            except BrokenPipeError as exc:
                raise encoder_failure(
                    "encoder stopped while finalizing the video"
                ) from exc
            code, message = encoder_result()
            if code:
                raise RuntimeError(f"ffmpeg failed ({code}): {message or 'unknown error'}")
            if is_cancelled():
                raise VideoExportCancelled("Video export cancelled")
            os.replace(temporary, target)
            report = dict(
                schema="ftgspp-video-export-v1", model_sha256=self.asset.sha256,
                video=target.name, mode=plan["mode"], fps=fps,
                duration=plan["duration"], frame_count=count,
                width=width + width % 2, height=height + height % 2,
                sh_degree=state["sh_degree"],
                marginal_threshold=state["marginal_threshold"],
                export_seconds=time.monotonic() - started,
                scene_timeline=dict(
                    mode=plan["scene_mode"], start=plan["scene_start"],
                    end=plan["scene_end"],
                ),
                display=settings, encoding=dict(
                    codec="h264", crf=18, pixel_format="yuv420p",
                    preset=VIDEO_ENCODER_PRESET, tune="zerolatency",
                    threads=VIDEO_ENCODER_THREADS,
                ),
                frames=frames,
            )
            atomic_json(target.with_suffix(".json"), report)
            return target
        except BaseException:
            # Terminate before closing stdin: closing a buffered pipe may flush
            # pending data and otherwise block on an encoder that stopped reading.
            if process.poll() is None:
                process.terminate()
            if process.stdin is not None and not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        finally:
            errors.close()
            self._forward_key = self._encoded_key = self._encoded = None


async def async_run(args, asset):
    from websockets.asyncio.server import serve
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Remote viewer binds loopback only; use SSH port forwarding")
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be 1..65535")
    viewer = RemoteViewer(args, asset)
    catalog_root = getattr(args, "catalog", None)
    if catalog_root is not None:
        from .app import catalog_runs

        viewer.catalog = catalog_runs(catalog_root)
        if not viewer.catalog:
            raise ValueError(f"No 4DAnyone or DNA-Rendering runs in {catalog_root}")
        current = asset.model_path.parent.resolve()
        if current not in viewer.catalog:
            raise ValueError(f"Opened run is not in the scene catalog {catalog_root}: {current}")
        print(
            "Scenes: " + ", ".join(path.name for path in viewer.catalog),
            flush=True,
        )
    html = files("ftgspp.viewer").joinpath("remote.html").read_bytes()
    await asyncio.to_thread(viewer.start_renderer)

    def response(code, body, content_type="text/plain; charset=utf-8", extra_headers=None):
        headers = {
            "Content-Type": content_type, "Content-Length": str(len(body)),
            "Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src blob: data:; media-src 'self'; connect-src 'self'",
        }
        headers.update(extra_headers or {})
        return Response(code, {200: "OK", 403: "Forbidden", 404: "Not Found", 503: "Busy"}.get(code, "Error"),
                        Headers(headers), body)

    def process_request(connection, request):
        parsed = urlsplit(request.path)
        token = parse_qs(parsed.query).get("token", [None])[0]
        if token is None or not secrets.compare_digest(token, viewer.token):
            return response(403, b"Invalid viewer token\n")
        host = request.headers.get("Host", "")
        if host not in (f"127.0.0.1:{args.port}", f"localhost:{args.port}", f"[::1]:{args.port}"):
            return response(403, b"Invalid host\n")
        if parsed.path == "/ws":
            if request.headers.get("Origin") != f"http://{host}":
                return response(403, b"Invalid origin\n")
            if viewer.client is not None:
                return response(503, b"Viewer already connected\n")
            return None
        if parsed.path == "/":
            return response(200, html, "text/html; charset=utf-8")
        if parsed.path.startswith("/exports/"):
            name = parsed.path.removeprefix("/exports/")
            target = viewer.exports.get(name)
            if target is None or target.name != name or not target.is_file():
                return response(404, b"Export not found\n")
            return response(200, target.read_bytes(), "video/mp4", {
                "Content-Disposition": f'inline; filename="{name}"',
            })
        return response(404, b"Not found\n")

    try:
        async with serve(
            viewer.serve_client, args.host, args.port, process_request=process_request,
            compression=None, max_size=512 * 1024, max_queue=4, server_header=None,
        ):
            url_host = "[::1]" if args.host == "::1" else "127.0.0.1"
            tunnel_host = "localhost" if args.host == "::1" else "127.0.0.1"
            print(f"Remote viewer: http://{url_host}:{args.port}/?token={viewer.token}", flush=True)
            print(f"SSH tunnel: ssh -L {args.port}:{tunnel_host}:{args.port} USER@SERVER", flush=True)
            await asyncio.Future()
    finally:
        if viewer.renderer is not None:
            await asyncio.to_thread(viewer.renderer.close)


def run_remote(args, asset):
    try:
        asyncio.run(async_run(args, asset))
    except ImportError as exc:
        if exc.name == "websockets":
            raise RuntimeError(
                "Remote viewer needs websockets; install with scripts/setup.sh --with-remote"
            ) from exc
        raise
    except KeyboardInterrupt:
        pass
