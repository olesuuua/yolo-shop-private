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

async function controller(url = "") {
  const elements = new Map();
  const captures = [];
  const created = [];
  const sockets = [];
  const requests = [];
  const revoked = [];
  const timeouts = [];
  const mediaCalls = [];
  const mediaNext = [];
  const mediaRequest = deferred();
  const track = { stopped: false, stop() { this.stopped = true; } };
  const media = { getTracks: () => [track] };
  function element() {
    const ctx = { calls: [], drawImage(...args) { this.calls.push(args); }, fillStyle: "", fillRect(...args) { this.calls.push(args); } };
    const el = {
      disabled: false, textContent: "", children: [], readyState: 2, style: {},
      classList: { add() {}, remove() {} },
      addEventListener(name, handler) { this[["ended", "error"].includes(name) ? "on" + name : name] = handler; },
      replaceChildren() { this.children = []; },
      appendChild(child) { this.children.push(child); },
      append(...children) { this.children.push(...children); },
      removeAttribute(name) { delete this[name]; },
      play: async () => {}, pause() {}, load() {},
      toDataURL: () => "data:image/png;base64,eA==",
      getContext: () => ctx,
      toBlob(callback, type, quality) {
        const fn = (blob) => callback(blob);
        fn.quality = quality; fn.type = type; fn.canvas = el;
        captures.push(fn);
      },
      __ctx: ctx,
    };
    let rvfcId = 0; let rvfcCb = null;
    let cancelled = 0;
    el.requestVideoFrameCallback = (cb) => { rvfcCb = cb; return ++rvfcId; };
    el.cancelVideoFrameCallback = () => { rvfcCb = null; cancelled += 1; };
    el.__deliverFrame = (metadata = {}) => {
      const cb = rvfcCb; rvfcCb = null;
      if (cb) cb(0, metadata);
    };
    el.__rvfcCancelled = () => cancelled;
    return el;
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
      createElement(...args) {
        const el = element(...args);
        created.push(el);
        return el;
      }
    },
    window: { addEventListener() {} },
    navigator: { mediaDevices: { getUserMedia: (constraints) => {
      mediaCalls.push(constraints);
      if (mediaNext.length) {
        const next = mediaNext.shift();
        return next && next.reject ? Promise.reject(next.reject) : Promise.resolve(media);
      }
      return mediaRequest.promise;
    } } },
    location: { protocol: "http:", host: "localhost:8000", search: url },
    WebSocket: Socket, HTMLMediaElement: { HAVE_CURRENT_DATA: 2 },
    fetch(url) {
      if (url === "/api/session") return Promise.resolve({ ok: true, json: async () => session() });
      if (url === "/api/ident-readiness") return Promise.resolve({ ok: true, json: async () => ({ products: [], catalog_ok: true }) });
      const request = deferred();
      requests.push(request);
      return request.promise;
    },
    Blob, Uint8Array, atob, btoa, TextEncoder, TextDecoder, DataView, Date, URLSearchParams,
    setInterval() { return 1; },
    setTimeout(fn, ms) { const timer = { fn, ms }; timeouts.push(timer); return timer; },
    clearTimeout(timer) {
      const index = timeouts.indexOf(timer);
      if (index >= 0) timeouts.splice(index, 1);
    },
    URL: { createObjectURL: () => "blob:test", revokeObjectURL: (url) => revoked.push(url) },
    requestAnimationFrame() { throw new Error("Unexpected frame retry"); }
  });
  vm.runInContext(readFileSync(path.join(__dirname, "../static/app.js"), "utf8"), context);
  await context.loadSession();
  return {
    context, elements, captures, created, sockets, requests, mediaRequest, media, track, revoked, timeouts,
    mediaCalls, mediaNext,
    fireTimeouts(ms = null) {
      for (const timer of timeouts.filter(t => ms == null || t.ms === ms)) {
        const index = timeouts.indexOf(timer);
        if (index >= 0) timeouts.splice(index, 1);
        timer.fn();
      }
    },
    camera() { return elements.get("camera"); },
    async tick() { await new Promise((done) => setImmediate(done)); },
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

// Switch to video source with a file and complete the ordered reset.
// The stale camera capture is invalidated by the epoch bump and drained.
async function switchToVideo(app, socket, file = new Blob(["video"])) {
  app.elements.get("sourceSelect").value = "video";
  const changed = app.context.changeSource(file);
  socket.receive(detectionResponse());
  socket.receive({ ...session(1), type: "reset_ack" });
  await changed;
  if (app.captures.length) app.capture();
}

// Start Play and let it register the decoded-frame validation.
function startPlay(app) {
  return app.context.playVideo();
}

// The validation registers in a microtask after play(); a macrotask tick
// guarantees the frame callback is registered before delivering a frame.
async function tick(app) { await new Promise((done) => setImmediate(done)); }

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
  socket.receive({...session(1), type: "reset_ack"});
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
  assert.equal(socket.sent.filter(x => typeof x !== "string").length, 0);
  socket.receive({...session(1), type: "reset_ack"});
  await reset;
  assert.equal(app.captures.length, 1);
  app.capture();
  assert.equal(socket.sent.filter(x => typeof x !== "string").length, 1);
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

test("detection upload is 640x480 q0.75 from the same video grab", async () => {
  const app = await controller();
  const socket = await app.connect();
  app.capture();
  const camera = app.elements.get("camera");
  camera.videoWidth = 1280;
  camera.videoHeight = 960;
  socket.receive(detectionResponse({ frame_id: 1 }));
  const detFn = app.captures[0];
  const canvas = app.elements.get("canvas");
  assert.equal(canvas.width, 640);
  assert.equal(canvas.height, 480);
  assert.equal(detFn.quality, 0.75);
  // Single live grab: detection canvas draws the full canvas, which alone
  // draws the camera. No second live draw feeds either version.
  const full = app.created.at(-1);
  assert.equal(full.width, 1280);
  assert.equal(full.height, 960);
  // 4:3 grab: letterbox is the identity, drawn into the full canvas.
  assert.equal(canvas.__ctx.calls.at(-1)[0], full);
  assert.deepEqual(canvas.__ctx.calls.at(-1).slice(1), [0, 0, 1280, 960, 0, 0, 640, 480]);
  assert.equal(full.__ctx.calls.length, 1);
  assert.equal(full.__ctx.calls[0][0], camera);
});

test("crops are cut at full resolution with q0.85", async () => {
  const app = await controller();
  const socket = await app.connect();
  app.capture();
  const camera = app.elements.get("camera");
  camera.videoWidth = 1280;
  camera.videoHeight = 960;
  socket.receive(detectionResponse({ frame_id: 2 }));
  app.capture(); // frame 2 goes out at full capture resolution
  socket.receive(detectionResponse({
    frame_id: 3, crop_requests: [{
      request_id: "3:7:1", frame_id: 3, track_id: 7, session_version: 0,
      bbox: [100, 100, 300, 300], coord_space: "detect_640x480",
      upload_width: 640, upload_height: 480, margin: 0.08,
    }],
  }));
  const cropFn = app.captures[0];
  assert.equal(cropFn.quality, 0.85);
  app.capture();
  const { header } = await parseEnvelope(socket.sent.at(-1));
  assert.equal(header.request_id, "3:7:1");
  // 2x capture scale maps the 200px box plus 8% margin to a 464px crop.
  const cropCanvas = app.created.find((el) => el.width === 464 && el.height === 464);
  assert.ok(cropCanvas, "crop canvas uses full-resolution mapping");
});

test("widescreen capture maps both axes from the same grab", async () => {
  const app = await controller();
  const socket = await app.connect();
  app.capture();
  const camera = app.elements.get("camera");
  camera.videoWidth = 1280;
  camera.videoHeight = 720;
  socket.receive(detectionResponse({ frame_id: 4 }));
  const full = app.created.at(-1);
  assert.equal(full.width, 1280);
  assert.equal(full.height, 720);
  // 16:9 grab: detection canvas draws the letterboxed content rect
  // (centered 640x360 + 60px bars), matching the server.
  const lb = app.context.letterboxRect(1280, 720);
  assert.equal(lb.dx, 0); assert.equal(lb.dy, 60);
  assert.equal(lb.w, 640); assert.equal(lb.h, 360);
  assert.deepEqual(app.elements.get("canvas").__ctx.calls.at(-1).slice(1), [0, 0, 1280, 720, 0, 60, 640, 360]);
  const rect = app.context.mapCropRect([100, 100, 300, 300], 1280, 720);
  // Inverse letterbox + 8% margin in upload pixels: (168,48,632,512)-ish.
  assert.ok(Math.abs(rect.x - 168) <= 2 && Math.abs(rect.y - 48) <= 2);
  assert.ok(Math.abs(rect.w - 464) <= 2 && Math.abs(rect.h - 464) <= 2);
});

test("diag overlay reports network-inclusive and backend timings", async () => {
  const app = await controller("?diag=1");
  const socket = await app.connect();
  app.capture();
  socket.receive(detectionResponse({ frame_id: 6, detect_ms: 12.5 }));
  const diag = app.elements.get("diag");
  assert.match(diag.textContent, /det .*KB/);
  assert.match(diag.textContent, /rtt .*ms \(net\+server\)/);
  assert.match(diag.textContent, /backend 12\.5ms/);
  assert.match(diag.textContent, /res\/s/);
  assert.match(diag.textContent, /crops 0/);
});

test("diag overlay stays hidden without the flag", async () => {
  const app = await controller();
  const socket = await app.connect();
  app.capture();
  socket.receive(detectionResponse({ frame_id: 7, detect_ms: 9.5 }));
  assert.equal(app.elements.has("diag"), false);
});

test("video pause and end stop submissions but finish same-frame crops", async () => {
  const app = await controller();
  const socket = await app.connect();
  app.elements.get("sourceSelect").value = "video";
  app.capture();
  app.context.pauseVideo();
  socket.receive(detectionResponse({crop_requests: [{request_id: "1:7:1", frame_id: 1,
    track_id: 7, session_version: 0, bbox: [100,100,300,300], upload_width:640, upload_height:480}]}));
  assert.equal(app.captures.length, 1);
  app.capture();
  assert.equal((await parseEnvelope(socket.sent.at(-1))).header.request_id, "1:7:1");
  app.elements.get("camera").onended();
  app.context.sendNextFrame();
  assert.equal(app.captures.length, 0);
  assert.equal(socket.readyState, 1);
});

test("source change releases camera, resets in order and uses one socket", async () => {
  const app = await controller(); const socket = await app.connect(); app.capture();
  app.elements.get("sourceSelect").value = "video";
  const changed = app.context.changeSource(new Blob(["video"]));
  assert.equal(app.track.stopped, true);
  socket.receive(detectionResponse()); // old detection during reset
  socket.receive({...session(1), type:"reset_ack"}); await changed;
  assert.equal(app.elements.get("camera").src, "blob:test");
  const camera = app.camera();
  camera.videoWidth = 1280; camera.videoHeight = 720;
  const playing = startPlay(app);
  await app.tick(); // validation registers
  app.camera().__deliverFrame({ width: 1280, height: 720 }); // decoded frame
  await playing;
  app.context.sendNextFrame(); app.context.sendNextFrame();
  assert.equal(app.captures.length, 1);
  assert.equal(app.sockets.length, 1);
  const restart = app.context.restartVideo();
  app.capture(); // encode from old session is invalidated
  socket.receive({...session(2),type:"reset_ack"});
  await app.tick(); // reset finishes and restart re-validates
  app.camera().__deliverFrame({ width: 1280, height: 720 });
  await restart;
  assert.equal(app.elements.get("camera").currentTime, 0);
  app.context.stop(); assert.ok(app.revoked.includes("blob:test"));
});

test("portrait native frame mapping and bounded diagnostics retain attribution", async () => {
  const app = await controller("?diag=1");
  const rect = app.context.mapCropRect([100,100,300,300], 1080,1920);
  assert.equal(rect.w,524); assert.equal(rect.h,928);
  const entry = {canvas:app.elements.get("canvas"),width:1080,height:1920,video_timestamp:2.5};
  for(let i=0;i<65;i++) app.context.recordCrop({request_id:String(i),frame_id:i,track_id:7,
    session_version:0,bbox:[100,100,300,300]},entry,rect);
  app.context.mergeEvidence({dropped:3,requests:[{request_id:"64",jev:{source_request_ids:["63","64"]}}]});
  assert.equal(vm.runInContext("evidenceRequests.length",app.context),60);
  assert.equal(vm.runInContext("evidenceDropped",app.context),5);
  assert.equal(vm.runInContext("evidenceRequests.at(-1).backend.jev.source_request_ids.length",app.context),2);
  app.context.clearEvidence();
  assert.equal(vm.runInContext("evidenceRequests.length",app.context),0);
});

test("late async crop encoding cannot cross restart epoch", async () => {
  const app = await controller(); const socket = await app.connect(); app.capture();
  socket.receive(detectionResponse({crop_requests:[{request_id:"old",frame_id:1,track_id:7,
    session_version:0,bbox:[100,100,300,300],upload_width:640,upload_height:480}]}));
  const reset = app.context.resetSession();
  socket.receive({...session(1),type:"reset_ack"}); await reset;
  app.capture(); // old crop
  const binary = socket.sent.filter(x => typeof x !== "string");
  assert.equal(binary.length,1);
});

test("exact exported crop bytes and asynchronous outcome keep request identity", async () => {
  const app = await controller("?diag=1");
  const request = app.context.recordCrop({request_id:"crop",frame_id:2,track_id:7,session_version:0,
    bbox:[100,100,300,300]}, {canvas:app.elements.get("canvas"),width:640,height:480,video_timestamp:1.25},
    {x:84,y:84,w:232,h:232});
  await app.context.saveCrop(request,new Blob(["exact jpeg bytes"]),0);
  assert.equal(Buffer.from(request.crop_jpeg.split(",")[1],"base64").toString(),"exact jpeg bytes");
  app.context.evidenceAck({request_id:"crop",ok:false,reason:"blurry"});
  assert.equal(request.reason,"blurry");
  assert.equal(request.video_timestamp,1.25);
});

test("unsupported video reports an error and status replies never unlock detection", async () => {
  const app = await controller(); const socket = await app.connect(); app.capture();
  socket.receive({type:"status",...session(0),identification:{}});
  app.context.sendNextFrame(); assert.equal(app.captures.length,0);
  app.elements.get("sourceSelect").value = "video";
  app.elements.get("camera").onerror();
  assert.match(app.elements.get("status").textContent,/could not decode the video/);
  app.context.sendNextFrame(); assert.equal(app.captures.length,0);
});


// --- Replay decoded-frame validation (Part A) ---

function sentDetectionFrames(socket) {
  return socket.sent.filter((blob) => typeof blob !== "string");
}

test("timeline and audio progress without a decoded frame send zero detection uploads", async () => {
  const app = await controller();
  const socket = await app.connect();
  await switchToVideo(app, socket);
  // HEVC-style failure: timeline advances (audio plays) but no video frames
  // decode and videoWidth/videoHeight stay zero; no media error is raised.
  const camera = app.camera();
  camera.currentTime = 3.5;
  const playing = startPlay(app);
  await app.tick(); // validation registers
  assert.equal(app.captures.length, 0);
  app.fireTimeouts(10000); // bounded startup timeout
  await playing;
  assert.match(app.elements.get("status").textContent, /could not decode the video/);
  assert.equal(app.captures.length, 0);
  assert.equal(sentDetectionFrames(socket).length, 0);
});

test("metadata-only fallback without requestVideoFrameCallback sends zero uploads", async () => {
  const app = await controller();
  const socket = await app.connect();
  await switchToVideo(app, socket);
  const camera = app.camera();
  delete camera.requestVideoFrameCallback; // documented fallback path
  camera.readyState = 1; // HAVE_METADATA only: metadata is not proof
  const playing = startPlay(app);
  await app.tick(); // validation registers
  app.fireTimeouts(100); // a few poll ticks that keep waiting
  app.fireTimeouts(10000);
  await playing;
  assert.match(app.elements.get("status").textContent, /could not decode the video/);
  assert.equal(app.captures.length, 0);
  assert.equal(sentDetectionFrames(socket).length, 0);
});

test("a genuinely decoded frame starts replay with normal uploads", async () => {
  const app = await controller();
  const socket = await app.connect();
  await switchToVideo(app, socket);
  const camera = app.camera();
  camera.videoWidth = 1920; camera.videoHeight = 1080;
  const playing = startPlay(app);
  await app.tick(); // validation registers
  assert.equal(app.captures.length, 0); // nothing before the decoded frame
  camera.__deliverFrame({ width: 1920, height: 1080 });
  await playing;
  assert.match(app.elements.get("status").textContent, /Playing video/);
  assert.equal(app.captures.length, 1); // detection capture queued
  app.capture();
  assert.equal(sentDetectionFrames(socket).length, 1);
  // Pause and resume through the existing pipeline.
  app.context.pauseVideo();
  camera.__deliverFrame({ width: 1920, height: 1080 }); // no callback pending
  const resuming = startPlay(app);
  await app.tick(); // validation registers
  camera.__deliverFrame({ width: 1920, height: 1080 });
  await resuming;
  assert.match(app.elements.get("status").textContent, /Playing video/);
});

test("fallback poll accepts a renderable frame and starts replay", async () => {
  const app = await controller();
  const socket = await app.connect();
  await switchToVideo(app, socket);
  const camera = app.camera();
  delete camera.requestVideoFrameCallback;
  camera.videoWidth = 1280; camera.videoHeight = 720;
  camera.readyState = 2; // HAVE_CURRENT_DATA: current position is renderable
  const playing = startPlay(app);
  await app.tick(); // validation registers
  app.fireTimeouts(100); // first poll tick settles
  await playing;
  assert.match(app.elements.get("status").textContent, /Playing video/);
  assert.equal(app.captures.length, 1);
});

test("decode timeout clears validation timers and shows an actionable message", async () => {
  const app = await controller();
  const socket = await app.connect();
  await switchToVideo(app, socket);
  const camera = app.camera();
  const playing = startPlay(app);
  await app.tick(); // validation registers
  const timersBefore = app.timeouts.length;
  app.fireTimeouts(10000);
  await playing;
  assert.match(app.elements.get("status").textContent, /could not decode the video/);
  assert.equal(app.captures.length, 0);
  assert.ok(camera.__rvfcCancelled() >= 1, "video frame callback cancelled");
  assert.equal(app.timeouts.length, 0);
  assert.equal(timersBefore >= 1, true);
});

test("decode error stops validation with the actionable message and cleanup", async () => {
  const app = await controller();
  const socket = await app.connect();
  await switchToVideo(app, socket);
  const camera = app.camera();
  const playing = startPlay(app);
  await app.tick(); // validation registers
  camera.onerror(); // media error during validation
  await playing;
  assert.match(app.elements.get("status").textContent, /could not decode the video/);
  assert.equal(app.captures.length, 0);
  assert.ok(camera.__rvfcCancelled() >= 1, "validation callbacks cancelled");
  // A late decoded frame after failure cannot restart submissions.
  camera.videoWidth = 1920; camera.videoHeight = 1080;
  camera.__deliverFrame({ width: 1920, height: 1080 });
  assert.equal(app.captures.length, 0);
});

test("stop during validation leaves no late capture or obsolete start", async () => {
  const app = await controller();
  const socket = await app.connect();
  await switchToVideo(app, socket);
  const camera = app.camera();
  camera.videoWidth = 1920; camera.videoHeight = 1080;
  const playing = startPlay(app);
  await app.tick(); // validation registers
  app.context.stop();
  camera.__deliverFrame({ width: 1920, height: 1080 });
  await playing;
  assert.equal(app.captures.length, 0);
  assert.equal(app.sockets.length, 1); // no obsolete socket was started
  assert.ok(!/Playing video/.test(app.elements.get("status").textContent));
});

test("file replacement during validation drops the late capture", async () => {
  const app = await controller();
  const socket = await app.connect();
  await switchToVideo(app, socket);
  const camera = app.camera();
  const playing = startPlay(app);
  await app.tick(); // validation registers
  const replaced = app.context.changeSource(new Blob(["other"]));
  socket.receive(detectionResponse());
  socket.receive({ ...session(2), type: "reset_ack" });
  await replaced;
  assert.equal(camera.src, "blob:test");
  camera.__deliverFrame({ width: 1920, height: 1080 }); // late callback
  await replaced;
  await playing;
  assert.equal(app.captures.length, 0);
  assert.equal(sentDetectionFrames(socket).length, 0);
  assert.match(app.elements.get("status").textContent, /Select a video, then Play/);
});

test("detection continues while a crop is pending", async () => {
  const app = await controller();
  const socket = await app.connect();
  await switchToVideo(app, socket);
  const camera = app.camera();
  camera.videoWidth = 1280; camera.videoHeight = 720;
  const playing = startPlay(app);
  await app.tick(); // validation registers
  camera.__deliverFrame({ width: 1280, height: 720 });
  await playing;
  app.capture(); // frame 1 upload
  socket.receive(detectionResponse({
    frame_id: 1, session_version: 1,
    crop_requests: [{ request_id: "1:7:1", frame_id: 1, track_id: 7,
      session_version: 1, bbox: [100, 100, 300, 300],
      coord_space: "detect_640x480", upload_width: 640, upload_height: 480 }],
  }));
  // Crop encoding is queued while the next detection capture is already
  // in flight: OCR/crop work never gates detection.
  assert.equal(app.captures.length, 2);
  const [cropEncode, nextDetection] = app.captures;
  nextDetection(new Blob(["jpeg"])); // detection frame 2 goes out first
  assert.equal(sentDetectionFrames(socket).length, 2);
  cropEncode(new Blob(["jpeg"])); // the crop finishes encoding after it
  assert.equal((await parseEnvelope(socket.sent.at(-1))).header.request_id, "1:7:1");
});

test("camera prefers 1920x1080 16:9 and reports the selected resolution", async () => {
  const app = await controller();
  app.camera().videoWidth = 1920; app.camera().videoHeight = 1080;
  const socket = await app.connect();
  assert.ok(app.mediaCalls.length >= 1);
  const first = app.mediaCalls[0];
  assert.equal(first.video.width && first.video.width.ideal, 1920);
  assert.equal(first.video.height && first.video.height.ideal, 1080);
  assert.equal(first.video.aspectRatio && first.video.aspectRatio.ideal, 16 / 9);
  assert.equal(app.elements.get("cameraInfo").textContent, "Camera 1920×1080");
  socket.close();
});

test("camera falls back through constraint sets on rejection", async () => {
  const app = await controller();
  const overconstrained = Object.assign(new Error("no match"), { name: "OverconstrainedError" });
  app.mediaNext.push({ reject: overconstrained }, { reject: overconstrained });
  app.camera().videoWidth = 1280; app.camera().videoHeight = 960;
  const socket = await app.connect();
  assert.ok(app.mediaCalls.length >= 3);
  assert.equal(app.mediaCalls[0].video.width.ideal, 1920);
  assert.equal(app.elements.get("cameraInfo").textContent, "Camera 1280×960");
  socket.close();
});

test("packed section persists unnamed items and upgrades their names", async () => {
  const app = await controller();
  const socket = await app.connect();
  socket.receive(detectionResponse({
    packed_counts: { Bottle: 1 },
    packed_display_counts: { "Unidentified bottle": 1 },
    packed_items: [{ display_name: "Unidentified bottle", count: 1, track_id: 7, has_crop: false }],
    last_event: { track_id: 7, class_name: "Bottle", display_name: "Unidentified bottle", timestamp: "2026-09-24T08:00:00.000Z" },
  }));
  const list = app.elements.get("packedList");
  assert.equal(list.children.length, 1);
  assert.match(list.children[0].children.at(-1).children[0].textContent, /Unidentified bottle/);
  assert.equal(app.elements.get("packedHeading").hidden, false);
  // Recognition clears when the box disappears, but the packed row stays,
  // then upgrades when a reliable name arrives — with no extra count.
  socket.receive(detectionResponse({
    packed_counts: { Bottle: 1 },
    packed_display_counts: { "Dobryi Cola 0,5": 1 },
    packed_items: [{ display_name: "Dobryi Cola 0,5", count: 1, track_id: 7, has_crop: true }],
    identification: {},
    last_event: { track_id: 7, class_name: "Bottle", display_name: "Dobryi Cola 0,5", timestamp: "2026-09-24T08:00:00.000Z" },
  }));
  assert.equal(list.children.length, 1);
  assert.match(list.children[0].children.at(-1).children[0].textContent, /Dobryi Cola/);
  assert.match(app.elements.get("lastEvent").textContent, /Dobryi Cola/);
  socket.close();
});
