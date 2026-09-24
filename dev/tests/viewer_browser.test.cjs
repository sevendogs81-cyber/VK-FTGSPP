// Minimal browser protocol harness: no frontend runtime dependency is needed.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { test } = require("node:test");
const path = require("node:path");

test("browser preserves aspect, navigation playback, deltas and decode backpressure", async () => {
  const nodes = new Map(),
    timers = new Map(),
    animations = new Map(),
    windowListeners = new Map();
  let timerId = 0,
    now = 0,
    decodeDone,
    animationRequests = 0,
    ws,
    viewportBox = { width: 4000, height: 2250 };
  function element() {
    return {
      value: "",
      textContent: "",
      src: "",
      style: {},
      children: [],
      listeners: {},
      append(child) {
        this.children.push(child);
      },
      replaceChildren() {
        this.children = [];
      },
      addEventListener(type, callback) {
        this.listeners[type] = callback;
      },
      setPointerCapture() {},
      click() {
        this.onclick?.();
      },
      getBoundingClientRect() {
        return viewportBox;
      },
      decode() {
        return new Promise((resolve) => {
          decodeDone = resolve;
        });
      },
    };
  }
  const document = {
    hidden: false,
    activeElement: { tagName: "BODY" },
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, element());
      return nodes.get(id);
    },
    createElement: element,
    addEventListener() {},
  };
  const context = vm.createContext({
    document,
    window: { addEventListener(type, callback) { windowListeners.set(type, callback); } },
    URLSearchParams,
    URL,
    Blob,
    Uint8Array,
    structuredClone,
    performance: { now: () => now },
    console,
    devicePixelRatio: 1,
    location: {
      search: "?token=test",
      protocol: "http:",
      host: "localhost:8765",
    },
    WebSocket: class {
      constructor() {
        ws = this;
        this.readyState = 1;
        this.sent = [];
      }
      send(value) {
        this.sent.push(JSON.parse(value));
      }
    },
    ResizeObserver: class {
      observe() {}
    },
    requestAnimationFrame(callback) {
      animations.set(++animationRequests, callback);
      return animationRequests;
    },
    setTimeout(fn) {
      timers.set(++timerId, fn);
      return timerId;
    },
    clearTimeout(id) {
      timers.delete(id);
    },
  });
  const html = fs.readFileSync(
    path.join(__dirname, "../../src/ftgspp/viewer/remote.html"),
    "utf8",
  );
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  const camera = {
    width: 1280,
    height: 720,
    K: [
      [900, 0, 640],
      [0, 900, 360],
      [0, 0, 1],
    ],
    w2c: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    timestamp: 2,
  };
  function assertRigid(w2c, label) {
    const rows = w2c.slice(0, 3).map((row) => row.slice(0, 3)),
      dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0),
      cross = (a, b) => [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
      ];
    for (const row of rows)
      assert.ok(Math.abs(dot(row, row) - 1) < 1e-10, `${label}: unit row`);
    assert.ok(Math.abs(dot(rows[0], rows[1])) < 1e-10, `${label}: orthogonal 0/1`);
    assert.ok(Math.abs(dot(rows[0], rows[2])) < 1e-10, `${label}: orthogonal 0/2`);
    assert.ok(Math.abs(dot(rows[1], rows[2])) < 1e-10, `${label}: orthogonal 1/2`);
    assert.ok(Math.abs(dot(cross(rows[0], rows[1]), rows[2]) - 1) < 1e-10,
      `${label}: right handed`);
  }
  const state = {
    camera,
    time: 2,
    speed: 1,
    playing: false,
    loop: true,
    snap: false,
    lock_camera: false,
    selected_camera: -1,
    mode: 0,
    background: [0, 0, 0],
    checker: false,
    exposure: 0,
    sh_degree: 3,
    marginal_threshold: 0.05,
    interactive: false,
    target_fps: 30,
    transparent: false,
    bookmarks: Array(100).fill(camera),
    path_times: Array.from({ length: 100 }, (_, i) => i),
  };
  await ws.onmessage({
    data: JSON.stringify({
      type: "hello",
      state,
      output_path: "/runs/example/exports",
      metadata: {
        initial_camera: camera,
        cameras: [],
        center: [0, 0, 1],
        radius: 1,
        world_up: [0, -1, 0],
        sample_times: [2, 3, 4],
        time_start: 2,
        time_end: 4,
      },
    }),
  });
  assert.equal(nodes.get("camera").disabled, true);
  assert.equal(
    nodes.get("outputPath").textContent,
    "Server exports: /runs/example/exports",
  );
  assert.equal(nodes.get("time").min, 2);
  assert.equal(nodes.get("speed").value, 1);
  assert.equal(
    animationRequests,
    0,
    "paused viewer must not run an animation loop",
  );
  const image = new Blob([new Uint8Array([1, 255, 216, 255])]);
  const display = ws.onmessage({ data: image });
  for (let i = 0; i < 10 && !decodeDone; ++i)
    await new Promise((resolve) => setImmediate(resolve));
  assert.equal(ws.sent.length, 0, "wait for initial image decode");
  decodeDone();
  await display;
  const flushTimers = () => {
    const pending = [...timers.values()];
    timers.clear();
    pending.forEach((fn) => fn());
  };
  flushTimers();
  assert.equal(ws.sent.length, 1);
  assert.deepEqual(ws.sent[0].state, {});
  nodes.get("time").value = 3;
  nodes.get("time").oninput();
  flushTimers();
  assert.equal(ws.sent.length, 1, "only one frame request can be in flight");
  const next = ws.onmessage({ data: image });
  await new Promise((resolve) => setImmediate(resolve));
  decodeDone();
  await next;
  flushTimers();
  assert.deepEqual(ws.sent[1].state, { time: 3 });
  nodes.get("quality").value = "interactive";
  nodes.get("quality").onchange();
  nodes.get("play").onclick();
  nodes.get("shot").onclick();
  const capture = ws.sent.at(-1);
  assert.equal(capture.type, "capture");
  assert.equal(
    capture.width,
    1440,
    "scale must apply after the 1080p stream cap",
  );
  assert.equal(capture.fullWidth, 3840);
  assert.equal(capture.state.bookmarks, undefined);

  // A camera pose edit used to unlock the aspect and stretch the render into
  // the browser box. Exercise real UI handlers, including legacy session state.
  nodes.get("quality").value = "faithful";
  nodes.get("quality").onchange();
  viewportBox = { width: 800, height: 1200 };
  const captureDimensions = (width, height, label) => {
    nodes.get("shot").click();
    const message = ws.sent.at(-1);
    assert.equal(message.width, width, label);
    assert.equal(message.height, height, label);
    assert.equal(message.fullWidth, width, label);
    assert.equal(message.fullHeight, height, label);
    assert.notEqual(message.state.lock_camera, false, label);
  };
  const stage = nodes.get("stage");
  for (const button of [0, 1, 2]) {
    stage.listeners.pointerdown({
      button, clientX: 20, clientY: 20, pointerId: 1, preventDefault() {},
    });
    stage.listeners.pointermove({ clientX: 45, clientY: 35 });
    stage.listeners.pointerup();
    captureDimensions(800, 450, `drag button ${button}`);
  }
  stage.listeners.wheel({ deltaY: 50, preventDefault() {} });
  captureDimensions(800, 450, "dolly");
  nodes.get("fit").click();
  captureDimensions(800, 450, "fit");
  nodes.get("reset").click();
  captureDimensions(800, 450, "reset");
  nodes.get("camera").value = -1;
  nodes.get("camera").onchange();
  captureDimensions(800, 450, "free camera");

  // Roll is a real camera-pose edit. A 180-degree correction must survive
  // subsequent orbiting so upside-down captures remain usable in free view.
  nodes.get("reset").click();
  nodes.get("rollFlip").click();
  nodes.get("shot").click();
  let rolled = ws.sent.at(-1).state.camera.w2c;
  assert.ok(rolled[0][0] < -0.999, "180 roll reverses camera right");
  assert.ok(rolled[1][1] < -0.999, "180 roll reverses camera down");
  assert.ok(rolled[2][2] > 0.999, "roll preserves the viewing direction");
  stage.listeners.pointerdown({
    button: 0, clientX: 20, clientY: 20, pointerId: 1, preventDefault() {},
  });
  stage.listeners.pointermove({ clientX: 70, clientY: 20 });
  stage.listeners.pointerup();
  nodes.get("shot").click();
  rolled = ws.sent.at(-1).state.camera.w2c;
  assert.ok(rolled[1][1] < -0.999, "orbit preserves camera roll");
  assertRigid(rolled, "rolled orbit pose");
  nodes.get("fit").click();
  nodes.get("shot").click();
  rolled = ws.sent.at(-1).state.camera.w2c;
  assert.ok(rolled[1][1] < -0.999, "fit preserves camera roll");
  nodes.get("reset").click();

  // Ctrl+left drag provides continuous optical-axis rotation in addition to
  // the exact-angle buttons.
  stage.listeners.pointerdown({
    button: 0, ctrlKey: true, clientX: 20, clientY: 20, pointerId: 1,
    preventDefault() {},
  });
  stage.listeners.pointermove({ clientX: 120, clientY: 20 });
  stage.listeners.pointerup();
  nodes.get("shot").click();
  rolled = ws.sent.at(-1).state.camera.w2c;
  assert.ok(Math.abs(rolled[0][1]) > 0.4, "roll drag changes camera orientation");
  assert.ok(rolled[2][2] > 0.999, "roll drag preserves the viewing direction");
  assertRigid(rolled, "continuous roll pose");
  nodes.get("reset").click();

  // A square capture like DNA-Rendering must stay square, even if an older
  // session carried lock_camera=false, and after loading a bookmark.
  const square = { ...camera, width: 1024, height: 1024 };
  await ws.onmessage({ data: JSON.stringify({
    type: "state", state: { ...state, camera: square, bookmarks: [square], path_times: [0] },
  }) });
  nodes.get("keys").children[0].children[0].click();
  captureDimensions(800, 800, "square bookmark / legacy session");
  viewportBox = { width: 1600, height: 900 };
  captureDimensions(900, 900, "square after resize");

  const tick = () => {
    const pending = [...animations.values()];
    animations.clear();
    now += 50;
    pending.forEach((callback) => callback(now));
  };
  const keepsPlaying = (label) => {
    const before = Number(nodes.get("time").value);
    tick();
    assert.equal(nodes.get("play").textContent, "Pause", label);
    assert.ok(Number(nodes.get("time").value) > before, `${label}: time must advance`);
    assert.equal(animations.size, 1, `${label}: one continuing animation loop`);
    nodes.get("shot").click();
    const patch = ws.sent.at(-1).state;
    assert.notEqual(patch.playing, false, label);
    assert.ok(patch.time > before, `${label}: advancing time reaches the server`);
  };
  nodes.get("play").click();
  for (const button of [0, 1, 2]) {
    stage.listeners.pointerdown({
      button, clientX: 20, clientY: 20, pointerId: 1, preventDefault() {},
    });
    stage.listeners.pointermove({ clientX: 45, clientY: 35 });
    keepsPlaying(`playing during drag ${button}`);
    stage.listeners.pointerup();
    keepsPlaying(`playing after drag ${button}`);
  }
  stage.listeners.wheel({ deltaY: 50, preventDefault() {} });
  keepsPlaying("playing during dolly");
  nodes.get("fit").click();
  keepsPlaying("playing after fit");
  stage.listeners.pointerdown({
    button: 2, clientX: 20, clientY: 20, pointerId: 1, preventDefault() {},
  });
  windowListeners.get("keydown")({ key: "w" });
  keepsPlaying("playing during keyboard fly");
  windowListeners.get("keyup")({ key: "w" });
  stage.listeners.pointerup();

  nodes.get("play").click();
  const pausedTime = Number(nodes.get("time").value);
  stage.listeners.wheel({ deltaY: 50, preventDefault() {} });
  tick();
  assert.equal(nodes.get("play").textContent, "Play", "navigation keeps a paused scene paused");
  assert.equal(Number(nodes.get("time").value), pausedTime);
  assert.equal(animations.size, 0);

  // Export snapshots the current free camera and explicit video settings.
  nodes.get("exportMode").value = "view";
  nodes.get("exportStart").value = 2;
  nodes.get("exportEnd").value = 4;
  nodes.get("exportWidth").value = 1024;
  nodes.get("exportFps").value = 30;
  nodes.get("pathSceneMode").value = "freeze";
  nodes.get("videoExport").click();
  const exportRequest = ws.sent.at(-1);
  assert.equal(exportRequest.type, "export_video");
  assert.deepEqual(exportRequest.export, {
    mode: "view", fps: 30, width: 1024, height: 1024,
    scene_mode: "freeze", scene_time: 2, time_start: 2, time_end: 4,
  });
  assert.equal(nodes.get("videoExport").disabled, true);
  await ws.onmessage({ data: JSON.stringify({
    type: "export_started", mode: "view", fps: 30, width: 1024,
    height: 1024, duration: 2, frame_count: 60,
  }) });
  await ws.onmessage({ data: JSON.stringify({
    type: "export_progress", completed: 30, total: 60, elapsed: 1,
  }) });
  assert.equal(nodes.get("exportProgress").value, 30);
  assert.match(nodes.get("exportStatus").textContent, /30\/60/);
  await ws.onmessage({ data: JSON.stringify({
    type: "saved", action: "video", file: "video_test.mp4",
    path: "/tmp/video_test.mp4", url: "/exports/video_test.mp4",
  }) });
  assert.equal(nodes.get("videoExport").disabled, false);
  assert.equal(nodes.get("exportResult").hidden, false);
  assert.equal(
    nodes.get("exportDownload").href,
    "/exports/video_test.mp4?token=test",
  );

  // Interaction recording is a distinct camera-time track. It can be exported
  // without borrowing scene timestamps from manually authored camera keys.
  nodes.get("recordToggle").click();
  assert.equal(ws.sent.at(-1).type, "start_recording");
  await ws.onmessage({ data: JSON.stringify({
    type: "recording_started", active: true, sample_count: 1, duration: 0,
  }) });
  assert.equal(nodes.get("recordToggle").textContent, "Stop recording");
  nodes.get("recordToggle").click();
  assert.equal(ws.sent.at(-1).type, "stop_recording");
  await ws.onmessage({ data: JSON.stringify({
    type: "recording_stopped", active: false, sample_count: 12, duration: 1.5,
  }) });
  nodes.get("exportMode").value = "recording";
  nodes.get("exportMode").onchange();
  nodes.get("videoExport").click();
  const recordedRequest = ws.sent.at(-1);
  assert.equal(recordedRequest.type, "export_video");
  assert.equal(recordedRequest.export.mode, "recording");
});
