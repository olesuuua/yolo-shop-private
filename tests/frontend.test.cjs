// Controller tests with deterministic camera/network timing; no real camera needed.
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function session(version = 0, count = 0) {
  return { session_version: version, packed_counts: count ? { apple: count } : {}, last_event: null };
}

async function controller() {
  const elements = new Map();
  const captures = [];
  const sockets = [];
  const requests = [];
  const revoked = [];
  const mediaRequest = deferred();
  const track = { stopped: false, stop() { this.stopped = true; } };
  const media = { getTracks: () => [track] };
  function element() {
    return {
      disabled: false, textContent: "", children: [], readyState: 2,
      classList: { add() {}, remove() {} },
      addEventListener(name, handler) { this[name] = handler; },
      replaceChildren() { this.children = []; },
      appendChild(child) { this.children.push(child); },
      append(...children) { this.children.push(...children); },
      removeAttribute(name) { delete this[name]; },
      play: async () => {},
      getContext: () => ({ drawImage() {} }),
      toBlob(callback) { captures.push(callback); }
    };
  }
  class Socket {
    static OPEN = 1;
    constructor() { this.readyState = 0; this.sent = []; sockets.push(this); }
    open() { this.readyState = 1; this.onopen(); }
    receive(data) { this.onmessage({ data: JSON.stringify(data) }); }
    send(blob) { this.sent.push(blob); }
    close() { this.readyState = 3; this.onclose?.(); }
  }
  const context = vm.createContext({
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
      },
      createElement: element
    },
    window: { addEventListener() {} },
    navigator: { mediaDevices: { getUserMedia: () => mediaRequest.promise } },
    location: { protocol: "http:", host: "localhost:8000" },
    WebSocket: Socket, HTMLMediaElement: { HAVE_CURRENT_DATA: 2 },
    fetch(url) {
      if (url === "/api/session") return Promise.resolve({ ok: true, json: async () => session() });
      if (url === "/api/ident-readiness") return Promise.resolve({ ok: true, json: async () => ({ products: [], catalog_ok: true }) });
      const request = deferred();
      requests.push(request);
      return request.promise;
    },
    Blob, Uint8Array, atob, TextEncoder, TextDecoder, DataView, Date,
    setInterval() { return 1; },
    setTimeout() { return 1; },
    URL: { createObjectURL: () => "blob:test", revokeObjectURL: (url) => revoked.push(url) },
    requestAnimationFrame() { throw new Error("Unexpected frame retry"); }
  });
  vm.runInContext(readFileSync(path.join(__dirname, "../static/app.js"), "utf8"), context);
  await context.loadSession();
  return {
    context, elements, captures, sockets, requests, mediaRequest, media, track, revoked,
    async connect() {
      const started = context.start();
      mediaRequest.resolve(media);
      await started;
      const socket = sockets.at(-1);
      socket.open();
      return socket;
    },
    capture() { captures.shift()(new Blob(["jpeg"])); },
    respond(socket, version = 0, count = 0) {
      socket.receive({ ...session(version, count), image: "eA==", visible_counts: { apple: 1 } });
    }
  };
}

test("camera sends one frame at a time and renders packed and visible counts", async () => {
  const app = await controller();
  const socket = await app.connect();
  app.capture();
  app.context.sendNextFrame();
  assert.equal(socket.sent.length, 1);
  assert.equal(app.captures.length, 0);
  app.respond(socket, 0, 2);
  assert.equal(app.elements.get("packedTotal").textContent, 2);
  assert.equal(app.elements.get("totalCount").textContent, 1);
  assert.equal(app.captures.length, 1);
  app.context.stop();
  assert.equal(app.track.stopped, true);
  assert.equal(app.elements.get("packedTotal").textContent, 2);
  assert.equal(app.elements.get("totalCount").textContent, 0);
});

test("stop during camera permission releases the late stream", async () => {
  const app = await controller();
  const started = app.context.start();
  app.context.stop();
  app.mediaRequest.resolve(app.media);
  await started;
  assert.equal(app.track.stopped, true);
  assert.equal(app.sockets.length, 0);
  assert.equal(app.elements.get("startButton").disabled, false);
});

test("a capture finishing after stop cannot send on a new connection", async () => {
  const app = await controller();
  const oldSocket = await app.connect();
  app.context.stop();
  const newSocket = await app.connect();
  app.capture();
  assert.equal(oldSocket.sent.length, 0);
  assert.equal(newSocket.sent.length, 0);
  app.capture();
  assert.equal(newSocket.sent.length, 1);
});

test("late frame from before reset cannot restore packed counts", async () => {
  const app = await controller();
  const socket = await app.connect();
  app.capture();
  const reset = app.context.resetSession();
  app.requests[0].resolve({ ok: true, json: async () => session(1) });
  await reset;
  app.respond(socket, 0, 4);
  assert.equal(app.elements.get("packedTotal").textContent, 0);
  assert.equal(app.elements.get("result").src, undefined);
  assert.equal(app.captures.length, 1);
  app.capture();
  app.respond(socket, 1, 1);
  assert.equal(app.elements.get("packedTotal").textContent, 1);
});

test("reset during capture resumes sending after the request finishes", async () => {
  const app = await controller();
  const socket = await app.connect();
  const reset = app.context.resetSession();
  app.capture();
  assert.equal(socket.sent.length, 0);
  app.requests[0].resolve({ ok: true, json: async () => session(1) });
  await reset;
  assert.equal(app.captures.length, 1);
  app.capture();
  assert.equal(socket.sent.length, 1);
});

test("server error closes the connection, releases camera and keeps the error visible", async () => {
  const app = await controller();
  const socket = await app.connect();
  socket.receive({ error: "Another camera is already connected." });
  assert.equal(socket.readyState, 3);
  assert.equal(app.track.stopped, true);
  assert.match(app.elements.get("status").textContent, /Another camera/);
  assert.equal(app.elements.get("startButton").disabled, false);
});

function detectionResponse(overrides = {}) {
  return {
    type: "detection", frame_id: 1, session_version: 0, image: "eA==",
    visible_counts: { apple: 1 }, packed_counts: {}, last_event: null,
    identification: {}, events: [], crop_requests: [], ...overrides,
  };
}

async function parseEnvelope(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  assert.equal(String.fromCharCode(...bytes.slice(0, 4)), "YOLO");
  assert.equal(bytes[4], 1);
  assert.equal(bytes[5], 2);
  const length = new DataView(bytes.buffer, bytes.byteOffset).getUint32(6, false);
  const header = JSON.parse(new TextDecoder().decode(bytes.slice(10, 10 + length)));
  return { header, jpeg: bytes.slice(10 + length) };
}

test("mapCropRect scales detection coordinates to retained uploads", async () => {
  const app = await controller();
  const { mapCropRect } = app.context;
  const full = mapCropRect([100, 100, 300, 300], 1280, 960);
  const half = mapCropRect([100, 100, 300, 300], 640, 480);
  assert.ok(full && half);
  assert.equal(full.w, half.w * 2);
  assert.equal(full.h, half.h * 2);
  assert.equal(mapCropRect([0, 0, 640, 480], 640, 480).w, 640);
  assert.equal(mapCropRect([10, 10, 20, 20], 640, 480), null);
  assert.equal(mapCropRect([300, 300, 100, 100], 640, 480), null);
  assert.equal(mapCropRect("nope", 640, 480), null);
});

test("crop envelope carries header plus jpeg without base64", async () => {
  const app = await controller();
  const header = { request_id: "1:7:1", frame_id: 1, track_id: 7, session_version: 0 };
  const envelope = app.context.buildCropEnvelope(header, new Blob(["jpeg"]));
  const { header: parsed, jpeg } = await parseEnvelope(envelope);
  assert.deepEqual(parsed, header);
  assert.ok(jpeg.length > 0);
});

test("crop requests are cut from the matching retained frame", async () => {
  const app = await controller();
  const socket = await app.connect();
  app.capture();
  assert.equal(socket.sent.length, 1);
  socket.receive(detectionResponse({
    frame_id: 5, crop_requests: [{
      request_id: "5:7:1", frame_id: 5, track_id: 7, session_version: 0,
      bbox: [100, 100, 300, 300], coord_space: "detect_640x480",
      upload_width: 640, upload_height: 480, margin: 0.08,
    }],
  }));
  // Crop extraction is queued synchronously during the response, before
  // the next detection frame (OCR never gates detection).
  assert.equal(socket.sent.length, 1);
  assert.equal(app.captures.length, 2);
  app.capture();
  assert.equal(socket.sent.length, 2);
  const { header, jpeg } = await parseEnvelope(socket.sent[1]);
  assert.deepEqual(header, { request_id: "5:7:1", frame_id: 5, track_id: 7, session_version: 0 });
  assert.ok(jpeg.length > 0);
  app.capture();
  assert.equal(socket.sent.length, 3);
});

test("stale, mismatched and tiny crop requests send nothing", async () => {
  const app = await controller();
  const socket = await app.connect();
  app.capture();
  socket.receive(detectionResponse({
    frame_id: 9, session_version: 0, crop_requests: [
      { request_id: "9:1:1", frame_id: 9, track_id: 1, session_version: -1,
        bbox: [100, 100, 300, 300], upload_width: 640, upload_height: 480 },
      { request_id: "9:2:2", frame_id: 8, track_id: 2, session_version: 0,
        bbox: [100, 100, 300, 300], upload_width: 640, upload_height: 480 },
      { request_id: "9:3:3", frame_id: 9, track_id: 3, session_version: 0,
        bbox: [100, 100, 300, 300], upload_width: 999, upload_height: 999 },
      { request_id: "9:4:4", frame_id: 9, track_id: 4, session_version: 0,
        bbox: [10, 10, 20, 20], upload_width: 640, upload_height: 480 },
    ],
  }));
  const queued = app.captures.length;
  for (let i = 0; i < queued; i += 1) app.capture();
  // Only detection frames were sent; no crop envelopes.
  for (const sent of socket.sent) {
    const bytes = new Uint8Array(await sent.arrayBuffer());
    assert.notEqual(String.fromCharCode(...bytes.slice(0, 4)), "YOLO");
  }
});

test("crop acks never clear the detection flight flag or duplicate sends", async () => {
  const app = await controller();
  const socket = await app.connect();
  app.capture();
  assert.equal(socket.sent.length, 1);
  socket.receive({ type: "crop_ack", request_id: "1:7:1", ok: true, reason: "accepted" });
  // Still waiting for the detection response: no new frame may go out.
  assert.equal(socket.sent.length, 1);
  assert.equal(app.captures.length, 0);
  socket.receive(detectionResponse({ frame_id: 3 }));
  assert.equal(app.captures.length, 1);
  app.capture();
  assert.equal(socket.sent.length, 2);
});

test("stop releases retained frames so late crops cannot send", async () => {
  const app = await controller();
  const socket = await app.connect();
  app.capture();
  socket.receive(detectionResponse({
    frame_id: 11, crop_requests: [{
      request_id: "11:7:1", frame_id: 11, track_id: 7, session_version: 0,
      bbox: [100, 100, 300, 300], upload_width: 640, upload_height: 480,
    }],
  }));
  app.context.stop();
  const queued = app.captures.length;
  for (let i = 0; i < queued; i += 1) {
    try { app.capture(); } catch { /* crop on closed socket is fine */ }
  }
  // At most the already-queued detection frame; no crop after stop.
  for (const sent of socket.sent.slice(1)) {
    const bytes = new Uint8Array(await sent.arrayBuffer());
    assert.notEqual(String.fromCharCode(...bytes.slice(0, 4)), "YOLO");
  }
});
