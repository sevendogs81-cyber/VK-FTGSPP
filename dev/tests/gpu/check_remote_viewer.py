#!/usr/bin/env python3
"""End-to-end headless HTTP/WebSocket viewer acceptance with a supplied NPZ."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus


async def receive(ws, kind):
    while True:
        data = await asyncio.wait_for(ws.recv(), timeout=30)
        if kind == "binary" and isinstance(data, bytes):
            return data
        if isinstance(data, str):
            message = json.loads(data)
            if message["type"] == "error":
                raise AssertionError(message)
            if message["type"] == kind:
                return message


async def check(port, token, scene):
    base = f"http://127.0.0.1:{port}"
    try:
        urlopen(base + "/", timeout=5)
        raise AssertionError("HTTP page accepted a missing token")
    except HTTPError as exc:
        assert exc.code == 403
    html = urlopen(base + f"/?token={token}", timeout=5).read()
    assert b"FTGSPP Viewer" in html and b"Camera path" in html
    endpoint = f"ws://127.0.0.1:{port}/ws?token={token}"
    try:
        async with connect(endpoint, origin="http://evil.example"):
            raise AssertionError("WebSocket accepted a foreign Origin")
    except InvalidStatus as exc:
        assert exc.response.status_code == 403
    try:
        async with connect(f"ws://127.0.0.1:{port}/ws?token=wrong", origin=base):
            raise AssertionError("WebSocket accepted a wrong token")
    except InvalidStatus as exc:
        assert exc.response.status_code == 403

    async with connect(endpoint, origin=base) as ws:
        hello = await receive(ws, "hello")
        frame = await receive(ws, "frame")
        jpeg = await receive(ws, "binary")
        assert jpeg[:3] == b"\x01\xff\xd8", frame
        assert cv2.imdecode(np.frombuffer(jpeg[1:], np.uint8), cv2.IMREAD_COLOR).shape[:2] == (frame["height"], frame["width"])
        try:
            async with connect(endpoint, origin=base):
                raise AssertionError("Accepted second viewer")
        except InvalidStatus as exc:
            assert exc.response.status_code == 503

        state = hello["state"]
        c = state["camera"]
        modified = dict(state, camera=dict(c, w2c=[list(row) for row in c["w2c"]]), time=0.7)
        modified["camera"]["w2c"][0][3] = 0.15
        await ws.send(json.dumps(dict(type="update", state=modified, width=96, height=64,
                                      fullWidth=192, fullHeight=128)))
        frame = await receive(ws, "frame")
        jpeg = await receive(ws, "binary")
        assert (frame["width"], frame["height"]) == (96, 64)
        assert jpeg[0] == 1
        rendered = frame["stats"]["frames"]
        readbacks = frame["stats"]["readbacks"]
        rgba8_readbacks = frame["stats"]["rgba8_readbacks"]
        assert readbacks == 0  # the remote path never downloads FP32 F/T
        assert frame["pipeline"]["readback_bytes"] == 96 * 64 * 4 + 4
        for exposure in (0.25, 0.5, 0.5):
            await ws.send(json.dumps(dict(type="update", state=dict(exposure=exposure))))
            cached = await receive(ws, "frame")
            await receive(ws, "binary")
            assert cached["stats"]["frames"] == rendered
            assert cached["stats"]["readbacks"] == readbacks
            assert cached["pipeline"]["forward_cached"]
        assert cached["pipeline"]["encoded_cached"]
        assert cached["stats"]["rgba8_readbacks"] == rgba8_readbacks + 2
        assert cached["pipeline"]["readback_bytes"] == 0
        await ws.send("[]")
        assert json.loads(await ws.recv())["type"] == "error"
        modified["time"] = 0.83
        await ws.send(json.dumps(dict(type="capture", state=modified,
                                     fullWidth=192, fullHeight=128)))
        screenshot = await receive(ws, "capture")
        png = await receive(ws, "binary")
        image = cv2.imdecode(np.frombuffer(png[1:], np.uint8), cv2.IMREAD_UNCHANGED)
        assert png[:9] == b"\x02\x89PNG\r\n\x1a\n"
        assert image.shape[:2] == (128, 192)
        assert Path(screenshot["path"]).is_file()
        assert Path(screenshot["path"]).with_suffix(".json").is_file()
        assert json.loads(Path(screenshot["path"]).with_suffix(".json").read_text())["camera"]["timestamp"] == 0.83
        await ws.send(json.dumps(dict(type="capture", state=dict(mode=0, transparent=True))))
        await receive(ws, "capture")
        png = await receive(ws, "binary")
        transparent = cv2.imdecode(np.frombuffer(png[1:], np.uint8), cv2.IMREAD_UNCHANGED)
        assert transparent.shape == (128, 192, 4)
        assert transparent[..., 3].min() == 0 and transparent[..., 3].max() > 0

        modified["bookmarks"] = [c, dict(c, timestamp=0.8)]
        modified["path_times"] = [0.0, 0.2]
        await ws.send(json.dumps(dict(type="update", state=modified, width=96, height=64,
                                      fullWidth=192, fullHeight=128)))
        await receive(ws, "frame")
        await receive(ws, "binary")
        for action in ("save_path", "sample_path", "save_session", "export_path"):
            await ws.send(json.dumps(dict(type=action, **({"seconds": 0.1} if action == "sample_path" else {}))))
            await receive(ws, "state" if action == "sample_path" else "saved")
            if action == "sample_path":
                await receive(ws, "frame")
                await receive(ws, "binary")
        assert (scene / "viewer_path.json").is_file()
        assert (scene / "viewer_session.json").is_file()
        assert list((scene / "viewer_output").glob("path_*/000000.png"))
        await ws.send(json.dumps(dict(type="load_path")))
        restored = await receive(ws, "state")
        assert len(restored["state"]["bookmarks"]) == 2
        await ws.send(json.dumps(dict(type="load_session")))
        await receive(ws, "state")
        await receive(ws, "frame")
        await receive(ws, "binary")

    async with connect(endpoint, origin=base) as ws:
        hello = await receive(ws, "hello")
        await receive(ws, "frame")
        await receive(ws, "binary")
        state = hello["state"]
        state.update(bookmarks=[c, dict(c, timestamp=.8)], path_times=[0, 100])
        await ws.send(json.dumps(dict(type="export_path", state=state)))
        await asyncio.sleep(.05)
    # Disconnecting a long export must release the renderer before reconnect.
    deadline = time.monotonic() + 3
    while True:
        try:
            async with connect(endpoint, origin=base) as ws:
                await receive(ws, "hello")
                await receive(ws, "frame")
                await receive(ws, "binary")
            break
        except InvalidStatus as exc:
            assert exc.response.status_code == 503 and time.monotonic() < deadline
            await asyncio.sleep(.02)
    print("Remote protocol, cache, current-state PNG, path/session/export, cancellation and reconnect: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--installed", action="store_true", help="test the installed wheel")
    parser.add_argument("--native-dir", type=Path, help="use extensions from a CMake build directory")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ftgspp-remote-") as tmp:
        scene = Path(tmp)
        shutil.copy2(args.model, scene / "gaussians.npz")
        manifest = args.model.parent / "viewer_manifest.json"
        if manifest.exists():
            shutil.copy2(manifest, scene / manifest.name)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        code = (
            "from ftgspp.viewer.app import main; "
            f"main(['--remote','--model',{str(scene / 'gaussians.npz')!r},"
            f"'--device',{str(args.device)!r},'--port',{str(port)!r}])"
        )
        if args.native_dir:
            code = (f"import ftgspp; ftgspp.__path__.insert(0, {str(args.native_dir.resolve())!r}); "
                    + code)
        env = os.environ.copy()
        if args.installed:
            env.pop("PYTHONPATH", None)
        else:
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "src")
        if args.native_dir:
            env["FTGSPP_VK_SHADER_DIR"] = str(args.native_dir.resolve() / "shaders")
        process = subprocess.Popen([sys.executable, "-c", code], cwd=Path.cwd(), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            deadline = time.monotonic() + 30
            token = None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(f"Viewer exited with {process.returncode}")
                ready, _, _ = select.select([process.stdout], [], [], 0.2)
                if ready:
                    line = process.stdout.readline()
                    match = re.search(rf"Remote viewer: http://127\.0\.0\.1:{port}/\?token=(\S+)", line)
                    if match:
                        token = match.group(1)
                        break
            assert token, "Viewer did not publish a URL"
            asyncio.run(check(port, token, scene))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
