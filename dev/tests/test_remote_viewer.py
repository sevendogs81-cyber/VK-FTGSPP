"""Cache correctness, session interoperability and bounded render scheduling."""

import asyncio
import io
import json
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ftgspp.viewer.app import catalog_runs, parser_for
from ftgspp.viewer.asset import camera, fit_camera_extent
from ftgspp.viewer.offline import compose
from ftgspp.viewer.remote import RemoteViewer, VideoExportCancelled, wire
from ftgspp.viewer.session import save_session


def test_browser_protocol():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional for browser protocol tests")
    subprocess.run([node, "--test", str(Path(__file__).with_name("viewer_browser.test.cjs"))],
                   check=True, capture_output=True, text=True, timeout=15)


class Renderer:
    def __init__(self):
        self.times = []
        self.destinations = []
        self.fail = False

    def render(self, c, t, sh, threshold):
        self.times.append(t)

    def readback_rgba8_into(self, out, **settings):
        self.destinations.append(id(out))
        if self.fail:
            self.fail = False
            raise RuntimeError("readback failed")
        assert out.dtype == np.uint8
        ft = np.full(out.shape, [.1 + self.times[-1] * .01, .2, .3, .4], dtype=np.float32)
        rgb = compose(ft, **settings)
        out[..., :3] = rgb[..., :3]
        out[..., 3] = rgb[..., 3] if rgb.shape[-1] == 4 else 255

    def stats(self):
        return dict(frames=len(self.times), readbacks=0, rgba8_readbacks=len(self.destinations))


@pytest.fixture
def viewer(tmp_path):
    c = camera(dict(width=48, height=32, K=[[40, 0, 24], [0, 40, 16], [0, 0, 1]],
                    w2c=np.eye(4), timestamp=.4))
    args = parser_for("viewer").parse_args([
        "--model", str(tmp_path / "gaussians.npz"), "--remote", "--no-session",
    ])
    asset = SimpleNamespace(model_path=args.model, sha256="test", metadata=dict(
        initial_camera=c, cameras=[c], sh_degree=3, marginal_threshold=.05,
        world_up=[0, -1, 0], fps=30,
    ))
    v = RemoteViewer(args, asset)
    v.renderer = Renderer()
    return v


def test_catalog_keeps_completed_4danyone_and_dna_runs(tmp_path):
    def touch(run, text):
        run.mkdir()
        (run / "gaussians.npz").write_bytes(b"")
        (run / "viewer_manifest.json").write_text("{}", encoding="utf-8")
        (run / "config.resolved.yaml").write_text(text, encoding="utf-8")

    touch(tmp_path / "karina", "root: /data/4danyone/douyin/karina\n")
    touch(tmp_path / "0025_11", "root: /data/dna_rendering/0025_11\n")
    touch(tmp_path / "other", "root: /data/somewhere/else\n")
    (tmp_path / "incomplete").mkdir()
    assert [path.name for path in catalog_runs(tmp_path)] == ["0025_11", "karina"]


def test_default_exports_are_beside_the_model_and_have_an_obvious_name(viewer):
    assert viewer.output == viewer.asset.model_path.parent / "exports"


def test_explicit_export_directory_still_wins(viewer, tmp_path):
    viewer.args.output = tmp_path / "custom-captures"
    overridden = RemoteViewer(viewer.args, viewer.asset)
    assert overridden.output == viewer.args.output


def test_remote_cache_tracks_render_and_display_dependencies(viewer):
    state = wire(viewer.state)
    original, _ = viewer.render_image(state, 48, 32, True)
    duplicate, _ = viewer.render_image(dict(state, speed=2), 48, 32, True)
    assert duplicate is original and len(viewer.renderer.times) == 1
    assert viewer.last_timings["encoded_cached"]
    exposed, _ = viewer.render_image(dict(state, exposure=1), 48, 32, True)
    assert exposed != original and len(viewer.renderer.times) == 1
    assert viewer.last_timings["forward_cached"]
    image = cv2.imdecode(np.frombuffer(exposed, np.uint8), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(image[0, 0], [153, 102, 53])
    for key, value in (("time", .9), ("sh_degree", 2), ("marginal_threshold", .1)):
        count = len(viewer.renderer.times)
        state[key] = value
        viewer.render_image(state, 48, 32, True)
        assert len(viewer.renderer.times) == count + 1
    for matrix, row, column in (("w2c", 0, 3), ("K", 0, 0)):
        count = len(viewer.renderer.times)
        state["camera"][matrix][row][column] += .1
        viewer.render_image(state, 48, 32, True)
        assert len(viewer.renderer.times) == count + 1
    assert len(set(viewer.renderer.destinations)) == 1  # reuse at fixed extent
    viewer.render_image(state, 24, 16, True)
    assert viewer._rgba.shape == (16, 24, 4)
    assert len(set(viewer.renderer.destinations)) == 2


def test_failed_readback_is_never_cached(viewer):
    state = wire(viewer.state)
    viewer.renderer.fail = True
    with pytest.raises(RuntimeError, match="readback failed"):
        viewer.render_image(state, 48, 32)
    viewer.render_image(state, 48, 32)
    assert len(viewer.renderer.times) == 2
    assert not viewer.last_timings["forward_cached"]


def test_remote_always_preserves_camera_aspect(viewer):
    # A non-matching browser/window box must not change the camera's projection.
    saved = dict(viewer.state, lock_camera=False)
    save_session(viewer.session, viewer.asset, saved)
    viewer.args.no_session = False
    restored = RemoteViewer(viewer.args, viewer.asset)
    assert restored.state["lock_camera"] is True
    viewer.apply_update(dict(state=dict(lock_camera=False), width=192, height=192))
    assert viewer.state["lock_camera"] is True
    cameras = []
    original = viewer.renderer.render
    def record_camera(c, *args):
        cameras.append(c)
        original(c, *args)
    viewer.renderer.render = record_camera
    for width, height, expected in ((192, 192, (128, 192)), (96, 32, (32, 48))):
        data, _ = viewer.render_image(viewer.state, width, height, True)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        assert image.shape[:2] == expected
        c = cameras[-1]
        np.testing.assert_allclose(c["K"][0] / c["width"], viewer.state["camera"]["K"][0] / 48)
        np.testing.assert_allclose(c["K"][1] / c["height"], viewer.state["camera"]["K"][1] / 32)


@pytest.mark.parametrize("source,box,expected", [
    ((1024, 1024), (1920, 1080), (1080, 1080)),
    ((1920, 1080), (800, 1200), (800, 450)),
    ((3000, 1000), (100, 80), (100, 33)),
    ((1000, 3000), (80, 100), (33, 100)),
])
def test_aspect_survives_pixel_rounding(source, box, expected):
    c = dict(width=source[0], height=source[1])
    extent = fit_camera_extent(c, *box)
    assert extent == expected
    assert fit_camera_extent(c, *extent) == extent


def test_remote_alpha_formats_and_gpu_readback_cache(viewer):
    state = wire(viewer.state)
    for mode in (0, 1, 2):
        data, _ = viewer.render_image(dict(state, mode=mode), 48, 32, True)
        rgb = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        assert rgb.shape == (32, 48, 3)  # opaque PNG retains its original format
        assert viewer.last_timings["readback_bytes"] == 48 * 32 * 4 + 4
    assert len(viewer.renderer.times) == 1
    data, _ = viewer.render_image(dict(state, transparent=True), 48, 32, True)
    rgba = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert rgba.shape == (32, 48, 4)
    np.testing.assert_array_equal(rgba[0, 0], [128, 85, 44, 153])
    jpeg, _ = viewer.render_image(dict(state, transparent=True), 48, 32)
    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_UNCHANGED).shape[-1] == 3
    transfers = len(viewer.renderer.destinations)
    duplicate, _ = viewer.render_image(dict(state, transparent=True), 48, 32)
    assert duplicate is jpeg and len(viewer.renderer.destinations) == transfers


def test_desktop_session_and_incremental_validation(viewer, monkeypatch):
    saved = dict(viewer.state, frames=500, render_scale=.75, axes=True)
    save_session(viewer.session, viewer.asset, saved)
    viewer.args.no_session = False
    restored = RemoteViewer(viewer.args, viewer.asset)
    assert "frames" not in restored.state and "axes" not in restored.state
    restored.apply_update(dict(state=wire(restored.state)))
    restored.apply_update(dict(state=dict(bookmarks=[restored.state["camera"]] * 100,
                                         path_times=list(range(100)))))
    import ftgspp.viewer.session as sessions
    def no_revalidation(*args):
        raise ValueError("revalidated path")
    monkeypatch.setattr(sessions, "from_state", no_revalidation)
    restored.apply_update(dict(state=dict(time=.8)))
    assert restored.state["time"] == .8
    with pytest.raises(ValueError, match="revalidated path"):
        restored.apply_update(dict(state=dict(path_times=list(range(100)))))
    before = dict(restored.state)
    with pytest.raises(ValueError):
        restored.apply_update(dict(state=dict(time=.2), width=0))
    assert restored.state == before


def test_video_export_plans_current_view_and_camera_path(viewer):
    state = wire(viewer.state)
    current = viewer.prepare_video_export(state, dict(
        mode="view", fps=24, width=48, height=48, time_start=.4, time_end=1.4,
    ))
    assert (current["width"], current["height"]) == (48, 32)
    assert current["frame_count"] == 24
    np.testing.assert_allclose(current["path"].cameras[0]["w2c"],
                               current["path"].cameras[1]["w2c"])
    assert [c["timestamp"] for c in current["path"].cameras] == [.4, 1.4]

    first, last = wire(state["camera"]), wire(state["camera"])
    first["timestamp"], last["timestamp"] = .4, .8
    state.update(bookmarks=[first, last], path_times=[1.0, 1.2])
    path = viewer.prepare_video_export(state, dict(
        mode="path", fps=10, width=48, height=32,
        scene_mode="freeze", scene_time=.5,
    ))
    assert path["duration"] == pytest.approx(.2)
    assert path["frame_count"] == 2
    assert path["scene_mode"] == "freeze"
    assert path["scene_start"] == path["scene_end"] == .5
    with pytest.raises(ValueError, match="at least two"):
        viewer.prepare_video_export(dict(state, bookmarks=[], path_times=[]), dict(
            mode="path", fps=30, width=48, height=32,
        ))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is optional")
def test_remote_video_export_streams_frames_directly_to_mp4(viewer):
    state = wire(viewer.state)
    first, last = wire(state["camera"]), wire(state["camera"])
    first["timestamp"], last["timestamp"] = .4, .8
    state.update(bookmarks=[first, last], path_times=[0.0, .2])
    plan = viewer.prepare_video_export(state, dict(
        mode="path", fps=10, width=48, height=32,
        scene_mode="freeze", scene_time=.5,
    ))
    reports = []
    target = viewer.export_video(
        state, plan, progress=lambda completed, total, elapsed:
        reports.append((completed, total, elapsed)),
    )
    assert target.read_bytes()[4:8] == b"ftyp"
    manifest = json.loads(target.with_suffix(".json").read_text())
    assert manifest["schema"] == "ftgspp-video-export-v1"
    assert manifest["video"] == target.name
    assert (manifest["width"], manifest["height"]) == (48, 32)
    assert manifest["frame_count"] == 2
    assert manifest["encoding"]["codec"] == "h264"
    assert manifest["encoding"]["preset"] == "veryfast"
    assert manifest["encoding"]["tune"] == "zerolatency"
    assert manifest["encoding"]["threads"] == 4
    assert reports[-1][:2] == (2, 2)
    assert viewer.renderer.times == pytest.approx([.5, .5])
    video = cv2.VideoCapture(str(target))
    decoded = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        decoded.append(frame)
    video.release()
    assert len(decoded) == 2
    assert decoded[0].shape == (32, 48, 3)
    assert not list(viewer.output.glob("*.tmp.mp4"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is optional")
def test_cancelled_video_export_removes_partial_file(viewer):
    state = wire(viewer.state)
    plan = viewer.prepare_video_export(state, dict(
        mode="view", fps=30, width=48, height=32, time_start=.4, time_end=1.4,
    ))
    with pytest.raises(VideoExportCancelled):
        viewer.export_video(state, plan, is_cancelled=lambda: True)
    assert not list(viewer.output.glob("*.mp4"))


def test_broken_ffmpeg_pipe_reports_the_encoder_error(viewer, monkeypatch):
    state = wire(viewer.state)
    plan = viewer.prepare_video_export(state, dict(
        mode="view", fps=10, width=48, height=32, time_start=.4, time_end=.5,
    ))

    class Input:
        closed = False

        def write(self, data):
            raise BrokenPipeError(32, "Broken pipe")

        def close(self):
            self.closed = True

    class Process:
        def __init__(self):
            self.stdin = Input()
            self.stderr = io.BytesIO(
                b"Error initializing output stream: encoder unavailable\n"
            )
            self.returncode = None

        def wait(self, timeout=None):
            self.returncode = 1
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    command = []

    def start(args, **kwargs):
        command.extend(args)
        return Process()

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(subprocess, "Popen", start)
    with pytest.raises(RuntimeError, match="encoder unavailable"):
        viewer.export_video(state, plan)
    assert command[command.index("-preset") + 1] == "veryfast"
    assert command[command.index("-tune") + 1] == "zerolatency"
    assert command[command.index("-threads") + 1] == "4"
    assert not list(viewer.output.glob("*.mp4"))


def test_interaction_recording_captures_camera_and_4d_time_independently(viewer, monkeypatch):
    ticks = iter([10.0, 10.5, 11.0])
    monkeypatch.setattr("ftgspp.viewer.remote.time.monotonic", lambda: next(ticks))
    started = viewer.start_interaction_recording()
    assert started == dict(active=True, sample_count=1, duration=0.0)

    viewer.state["camera"]["w2c"][0, 3] = .25
    viewer.state["time"] = .7
    viewer.record_interaction_sample()
    viewer.state["camera"]["w2c"][0, 3] = .5
    viewer.state["time"] = .9
    stopped = viewer.stop_interaction_recording()
    assert stopped == dict(active=False, sample_count=3, duration=1.0)

    plan = viewer.prepare_video_export(wire(viewer.state), dict(
        mode="recording", fps=30, width=48, height=32,
    ))
    assert plan["scene_mode"] == "track"
    assert [key["camera"]["timestamp"] for key in viewer.recorded_path["keyframes"]] == [
        .4, .7, .9,
    ]
    assert [key["path_time"] for key in viewer.recorded_path["keyframes"]] == [
        0.0, .5, 1.0,
    ]
    midpoint = plan["path"].sample(.75)
    assert midpoint["timestamp"] == pytest.approx(.8)
    assert midpoint["w2c"][0, 3] == pytest.approx(.375)


def test_latest_frame_and_capture_snapshot_are_bounded(viewer):
    started, release = threading.Event(), threading.Event()
    original_render = viewer.renderer.render
    def slow_first(*args):
        if not viewer.renderer.times:
            started.set()
            assert release.wait(5)
        original_render(*args)
    viewer.renderer.render = slow_first

    async def exercise():
        class Socket:
            def __init__(self):
                self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()
            def __aiter__(self):
                return self
            async def __anext__(self):
                item = await self.incoming.get()
                if item is None:
                    raise StopAsyncIteration
                return json.dumps(item)
            async def send(self, data):
                await self.outgoing.put(data)
        ws = Socket()
        task = asyncio.create_task(viewer.serve_client(ws))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            await ws.incoming.put(dict(type="capture", state=dict(time=.8), fullWidth=192, fullHeight=192))
            for i in range(1, 51):
                await ws.incoming.put(dict(type="update", state=dict(time=float(i))))
            for _ in range(1000):
                if viewer.state["time"] == 50:
                    break
                await asyncio.sleep(.001)
            assert viewer.state["time"] == 50
            release.set()
            frames = 0
            while frames < 2:
                output = await asyncio.wait_for(ws.outgoing.get(), 3)
                if isinstance(output, bytes) and output[0] == 1:
                    frames += 1
                elif isinstance(output, str):
                    assert json.loads(output)["type"] != "error", output
            await ws.incoming.put(None)
            await asyncio.wait_for(task, 3)
        finally:
            release.set()
            if not task.done():
                await ws.incoming.put(None)
                await task
        assert viewer.renderer.times == [.4, .8, 50.0]
        sidecar = next(viewer.output.glob("capture_*.json"))
        capture_camera = json.loads(sidecar.read_text())["camera"]
        assert capture_camera["timestamp"] == .8
        assert (capture_camera["width"], capture_camera["height"]) == (192, 128)
        assert cv2.imread(str(sidecar.with_suffix(".png"))).shape[:2] == (128, 192)
        assert viewer.client is None
    asyncio.run(exercise())
