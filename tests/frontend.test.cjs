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
    Blob, Uint8Array, atob,
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
