// Run the actual browser scheduler against a bounded, ordered virtual transport.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

async function simulation({ interactive = false, bandwidth = Infinity, sourceFps = 15, targetFps = 30, refreshHz = 60 } = {}) {
  let now = 0, taskId = 0, objectId = 0, ws, serverStream,
    commandArrival = 0, packetArrival = 0, jitterIndex = 0, maxOutstanding = 0;
  const tasks = new Map(), nodes = new Map(), presentations = [], commands = [];
  const sentFrames = new Map();
  const later = (fn, delay = 0) => {
    const id = ++taskId;
    tasks.set(id, { at: now + delay, fn });
    return id;
  };
  async function until(end) {
    let steps = 0;
    while (true) {
      const next = [...tasks.entries()].sort((a, b) => a[1].at - b[1].at)[0];
      if (!next || next[1].at > end) break;
      assert.ok(++steps < 40000, 'scheduler must stay bounded');
      tasks.delete(next[0]); now = next[1].at; next[1].fn();
      await new Promise((resolve) => setImmediate(resolve));
    }
    now = end;
  }
  function element() {
    return {
      value: '', src: '', textContent: '', style: {}, children: [], listeners: {},
      classList: { toggle() {} },
      append(child) { this.children.push(child); }, replaceChildren() { this.children = []; },
      addEventListener(type, fn) { this.listeners[type] = fn; },
      click() { this.onclick?.(); }, setPointerCapture() {},
      getBoundingClientRect() { return { width: 1280, height: 720 }; },
      decode() { return new Promise((resolve) => later(resolve, 4)); },
    };
  }
  const document = {
    hidden: false, activeElement: { tagName: 'BODY' }, listeners: {},
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id); },
    createElement: element, addEventListener(type, fn) { this.listeners[type] = fn; },
  };
  const oneWay = () => [100, 160, 110, 140, 100, 120][jitterIndex++ % 6];
  function downstream(fn, bytes = 0) {
    packetArrival = Math.max(packetArrival, now + oneWay()) + bytes * 8 / bandwidth / 1000;
    later(fn, packetArrival - now);
  }
  function sendFrame(info) {
    const bytes = Math.round(50000 * ((info.width || 1280) / 1280) ** 2);
    downstream(() => {
      ws.onmessage({ data: JSON.stringify({ type: 'frame', width: 1280, height: 720,
        ms: 5, bytes, stats: { render_ms: 5, gaussians: 1000, vram_bytes: 10000 },
        pipeline: {}, ...info }) });
      ws.onmessage({ data: new Blob([new Uint8Array([1, 255, 216, 255])]) });
    }, bytes);
  }
  function pump() {
    const stream = serverStream;
    if (!stream || stream.pending || stream.finished || stream.next - stream.ack - 1 >= stream.window) return;
    stream.pending = true;
    later(() => {
      if (serverStream !== stream) return;
      const index = stream.next++, time = stream.time;
      const info = { stream_id: stream.id, frame_index: index, revision: stream.revision,
        navigating: stream.navigating, view_time: stream.view_time,
        scene_time: Math.round((time % ((8 * sourceFps + 1) / sourceFps)) * sourceFps) / sourceFps,
        next_time: time + (stream.playing ? 1 / stream.fps : 0),
        playing: stream.playing, finished: !stream.playing && !stream.navigating,
        fps: stream.fps, width: stream.width, pose: stream.pose };
      stream.time = info.next_time; stream.finished = info.finished;
      sentFrames.set(`${stream.id}/${index}`, info);
      maxOutstanding = Math.max(maxOutstanding, stream.next - stream.ack - 1);
      sendFrame(info);
      // GPU 5 ms, navigation paced at target FPS. Playback fills the credit window.
      later(() => { stream.pending = false; pump(); }, stream.navigating ? 1000 / stream.fps - 5 : 0);
    }, 5);
  }
  const context = vm.createContext({
    document, window: { addEventListener() {} }, URLSearchParams, Blob, Uint8Array,
    URL: { createObjectURL() { return `blob:${++objectId}`; }, revokeObjectURL() {} },
    structuredClone, performance: { now: () => now }, console, devicePixelRatio: 1,
    location: { search: '?token=test', protocol: 'http:', host: 'localhost:8765' },
    ResizeObserver: class { observe() {} }, setTimeout: later, clearTimeout: (id) => tasks.delete(id),
    requestAnimationFrame: (fn) => later(() => fn(now), 1000 / refreshHz),
    WebSocket: class {
      constructor() { ws = this; this.readyState = 1; this.bufferedAmount = 0; }
      send(raw) {
        const msg = JSON.parse(raw); commands.push({ ...msg, at: now });
        if (msg.type === 'stream_ack' && msg.presented !== false)
          presentations.push({ at: now, index: msg.frame_index, ...sentFrames.get(`${msg.stream_id}/${msg.frame_index}`) });
        commandArrival = Math.max(commandArrival + .1, now + oneWay());
        later(() => {
          if (msg.type === 'stream') {
            serverStream = { id: msg.stream_id, time: msg.playback_time ?? msg.state.time ?? 0,
              playing: msg.state.playing ?? false, next: 0, ack: -1, width: msg.width,
              revision: 0, navigating: msg.navigating, fps: Math.min(msg.state.target_fps ?? targetFps, msg.navigating ? 60 : sourceFps), window: msg.window,
              view_time: msg.view_time, pose: msg.state.camera };
            pump();
          } else if (msg.type === 'stream_view' && serverStream?.id === msg.stream_id) {
            Object.assign(serverStream, { revision: msg.revision, navigating: msg.navigating,
              fps: Math.min(msg.state.target_fps ?? targetFps, msg.navigating ? 60 : sourceFps), width: msg.width, window: msg.window, view_time: msg.view_time,
              pose: msg.state.camera, finished: false });
            pump();
          } else if (msg.type === 'stream_ack' && serverStream?.id === msg.stream_id) {
            serverStream.ack = Math.max(serverStream.ack, msg.frame_index); pump();
          } else if (msg.type === 'stream_stop') {
            if (serverStream?.id === msg.stream_id) serverStream = null;
          } else if (msg.type === 'update') sendFrame({});
          else if (msg.type === 'ping') downstream(() => ws.onmessage({ data: JSON.stringify({ type: 'pong', client_time: msg.client_time }) }));
        }, commandArrival - now);
      }
    },
  });
  const html = fs.readFileSync(path.join(__dirname, '../../src/ftgspp/viewer/remote.html'), 'utf8');
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  const camera = { width: 1280, height: 720, timestamp: 0,
    K: [[900, 0, 640], [0, 900, 360], [0, 0, 1]],
    w2c: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] };
  await ws.onmessage({ data: JSON.stringify({ type: 'hello', model_sha256: 'test',
    stream_protocol: 'ftgspp-remote-stream-v2', stream_window: 64,
    state: { camera, time: 0, playing: false, speed: 1, loop: true, snap: false,
      selected_camera: -1, mode: 0, background: [0, 0, 0], checker: false, exposure: 0,
      sh_degree: 3, marginal_threshold: .05, interactive, target_fps: targetFps,
      transparent: false, bookmarks: [], path_times: [], world_up: [0, -1, 0] },
    metadata: { initial_camera: camera, cameras: [], center: [0, 0, 1], radius: 1,
      world_up: [0, -1, 0], sample_times: Array.from({ length: 8 * sourceFps + 1 }, (_, i) => i / sourceFps),
      time_start: 0, time_end: 8, fps: sourceFps },
  }) });
  sendFrame({}); await until(1000);
  return { until, nodes, presentations, commands, document, ws,
    setBandwidth(value) { bandwidth = value; },
    get maxOutstanding() { return maxOutstanding; } };
}

function cadence(frames) {
  const gaps = frames.slice(1).map((p, i) => p.at - frames[i].at);
  return { fps: (frames.length - 1) * 1000 / (frames.at(-1).at - frames[0].at), maxGap: Math.max(...gaps) };
}

test('buffered playback sustains 15 FPS across 200–320 ms RTT with jitter', async () => {
  const s = await simulation();
  s.nodes.get('play').click(); await s.until(9000);
  const steady = s.presentations.filter((p) => p.at >= 2000), result = cadence(steady);
  assert.ok(result.fps >= 14.5, JSON.stringify(result));
  assert.ok(result.maxGap < 85, JSON.stringify(result));
  assert.ok(s.maxOutstanding >= 8 && s.maxOutstanding <= 12);
  assert.equal(s.commands.filter((m) => m.type === 'stream').length, 1);
  assert.ok(steady.every((p, i) => !i || p.index === steady[i - 1].index + 1));
  s.nodes.get('play').click(); await s.until(9400);
  const count = s.presentations.length; await s.until(10000);
  assert.equal(s.presentations.length, count);
  console.log('Playback:', result);
});

for (const playing of [false, true]) test(`camera navigation pipelines over 200–320 ms RTT; playing=${playing}`, async () => {
  const s = await simulation();
  if (playing) { s.nodes.get('play').click(); await s.until(2000); }
  const stage = s.nodes.get('stage');
  stage.listeners.pointerdown({ button: 0, clientX: 0, clientY: 0, pointerId: 1, preventDefault() {} });
  for (let i = 0; i < 240; ++i) {
    stage.listeners.pointermove({ clientX: i, clientY: i / 10 });
    await s.until(2000 + i * 1000 / 60);
  }
  const frames = s.presentations.filter((p) => p.at > 3500 && p.navigating), result = cadence(frames);
  assert.ok(result.fps > 24, JSON.stringify(result));
  assert.ok(result.maxGap < 180, JSON.stringify(result));
  assert.equal(s.commands.filter((m) => m.type === 'stream').length, 1, 'camera edits must not restart playback');
  assert.ok(s.commands.filter((m) => m.type === 'stream_view').length > 80);
  assert.ok(frames.every((f) => Math.abs(f.scene_time * 15 - Math.round(f.scene_time * 15)) < 1e-7));
  if (!playing) assert.ok(frames.every((f) => f.scene_time === 0), 'paused time stays fixed during camera motion');
  assert.ok(s.maxOutstanding <= 64);
  stage.listeners.pointerup(); await s.until(7000);
  assert.equal(s.nodes.get('play').textContent, playing ? 'Pause' : 'Play');
  const count = s.presentations.length; await s.until(8000);
  if (!playing) assert.equal(s.presentations.length, count, 'idle camera finishes after final full-resolution frame');
  console.log('Navigation:', { playing, ...result });
});

test('temporary bandwidth shortage reduces resolution moderately then recovers without restarting', async () => {
  const s = await simulation({ interactive: true, bandwidth: 2 });
  s.nodes.get('play').click(); await s.until(14000);
  const requests = () => s.commands.filter((m) => ['stream', 'stream_view'].includes(m.type));
  assert.equal(requests()[0].width, 1280, 'no unconditional 75% downscale');
  const low = Math.min(...requests().map((m) => m.width));
  assert.ok(low < 1280 && low >= 1280 * .6 - 1, `resolution floor: ${low}`);
  assert.ok(requests().every((m) => m.fullWidth === 1280), 'capture resolution never adapts');
  s.setBandwidth(Infinity); await s.until(60000);
  assert.equal(requests().at(-1).width, 1280, 'quality recovers automatically');
  assert.equal(s.commands.filter((m) => m.type === 'stream').length, 1);
  const status = s.nodes.get('status').textContent;
  assert.ok(status.includes('display FPS') && status.includes('ms RTT') && status.includes('buffer'));
  console.log('Resolution recovery:', low, '->', requests().at(-1).width);
});

for (const sourceFps of [24, 30, 60, 120]) test(`source ${sourceFps} FPS keeps time and bounded prefetch under jitter`, async () => {
  const s = await simulation({ sourceFps, targetFps: sourceFps, refreshHz: Math.max(60, sourceFps) });
  s.nodes.get('play').click(); await s.until(10000);
  const frames = s.presentations.filter((p) => p.at > 4000), result = cadence(frames);
  assert.ok(result.fps >= sourceFps * .92, JSON.stringify({ sourceFps, ...result }));
  assert.ok(frames.every((p) => Math.abs(p.scene_time * sourceFps - Math.round(p.scene_time * sourceFps)) < 1e-7));
  assert.ok(s.maxOutstanding <= 64);
  assert.equal(s.commands.filter((m) => m.type === 'stream').length, 1);
  console.log('Source cadence:', { sourceFps, window: s.maxOutstanding, ...result });
});

test('120 FPS scene on 60 Hz display advances scene time at real speed using captured samples', async () => {
  const s = await simulation({ sourceFps: 120, targetFps: 120, refreshHz: 60 });
  s.nodes.get('play').click(); await s.until(7000);
  const frames = s.presentations.filter((p) => p.at > 4000), result = cadence(frames);
  assert.ok(result.fps > 55);
  const first = frames[0], last = frames.at(-1);
  assert.ok(Math.abs((last.scene_time - first.scene_time) - (last.at - first.at) / 1000) < .1);
  assert.ok(frames.every((p) => Math.abs(p.scene_time * 120 - Math.round(p.scene_time * 120)) < 1e-7));
});
