const camera = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const viewer = document.getElementById("viewer");
const resultImage = document.getElementById("result");
const statusText = document.getElementById("status");
const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const resetButton = document.getElementById("resetButton");
const totalCount = document.getElementById("totalCount");
const classCounts = document.getElementById("classCounts");
const packedTotal = document.getElementById("packedTotal");
const packedCounts = document.getElementById("packedCounts");
const lastEvent = document.getElementById("lastEvent");
const modelInfo = document.getElementById("modelInfo");
const identList = document.getElementById("identList");
const identReadiness = document.getElementById("identReadiness");

let socket = null;
let stream = null;
let running = false;
let resetting = false;
let frameInFlight = false;
let resultUrl = null;
let sessionVersion = -1;
let connectionGeneration = 0;

// Stage 2: detection uploads are exactly the backend processing size
// (640x480, JPEG quality 0.75, plain stretch like the server's cv2.resize,
// so geometry and the bag zone stay aligned). The original camera frame is
// retained at capture resolution (<=1280px wide) for OCR. Both versions are
// drawn from a single grab of the live video: video -> full canvas, then
// full canvas -> 640x480 detection canvas. Crops are cut from the retained
// full-resolution frame, never from the smaller detection JPEG. See
// docs/detect-crop-protocol.md. Detection coordinate space stays 640x480
// (origin top-left); the browser scales to retained pixels, expands by
// CROP_MARGIN, clamps, drops crops smaller than CROP_MIN_W/H, and sends
// crop JPEGs at CROP_JPEG_QUALITY.
const DETECT_WIDTH = 640;
const DETECT_HEIGHT = 480;
const DETECT_JPEG_QUALITY = 0.75;
const CROP_MARGIN = 0.08;
const CROP_MIN_WIDTH = 60;
const CROP_MIN_HEIGHT = 80;
const CROP_JPEG_QUALITY = 0.85;
const MAX_RETAINED_FRAMES = 4;
const RETAINED_FRAME_TTL_MS = 15000;
// Upload state for the last sent detection frame, awaiting its response.
let pendingUpload = null;
// Server frame_id -> {canvas, width, height, detectWidth, detectHeight, at}.
const retainedFrames = new Map();

// Opt-in performance diagnostics (?diag=1): detection JPEG bytes,
// send-to-response time (network-inclusive), backend detect_ms
// (network-exclusive), received results per second, crop count/bytes.
let diagEnabled = false;
try {
  diagEnabled = new URLSearchParams(location.search || "").has("diag");
} catch (error) {
  diagEnabled = false;
}
const diagTimes = [];
const diag = { detBytes: 0, rttMs: null, backendMs: null, crops: 0, cropBytes: 0 };
let diagSentAt = 0;
const diagClock = () => ((typeof performance !== "undefined" && performance.now)
  ? performance.now() : Date.now());

function renderDiag() {
  if (!diagEnabled) return;
  const box = document.getElementById("diag");
  if (!box) return;
  const cutoff = diagClock() - 2000;
  while (diagTimes.length && diagTimes[0] < cutoff) diagTimes.shift();
  const rate = (diagTimes.length / 2).toFixed(1);
  const rtt = diag.rttMs == null ? "?" : `${Math.round(diag.rttMs)}ms`;
  const backend = diag.backendMs == null ? "?" : `${diag.backendMs}ms`;
  box.textContent = `det ${(diag.detBytes / 1024).toFixed(1)}KB · ` +
    `rtt ${rtt} (net+server) · backend ${backend} · ${rate} res/s · ` +
    `crops ${diag.crops} (${(diag.cropBytes / 1024).toFixed(1)}KB)`;
  box.style.display = "block";
}

function mapCropRect(bbox, retainedWidth, retainedHeight) {
  if (!Array.isArray(bbox) || bbox.length !== 4) return null;
  const [bx1, by1, bx2, by2] = bbox.map(Number);
  if (![bx1, by1, bx2, by2].every(Number.isFinite)) return null;
  if (!(bx2 > bx1) || !(by2 > by1)) return null;
  if (!(retainedWidth > 0) || !(retainedHeight > 0)) return null;
  const scaleX = retainedWidth / DETECT_WIDTH;
  const scaleY = retainedHeight / DETECT_HEIGHT;
  const width = bx2 - bx1;
  const height = by2 - by1;
  const x1 = Math.max(0, Math.round((bx1 - width * CROP_MARGIN) * scaleX));
  const y1 = Math.max(0, Math.round((by1 - height * CROP_MARGIN) * scaleY));
  const x2 = Math.min(retainedWidth, Math.round((bx2 + width * CROP_MARGIN) * scaleX));
  const y2 = Math.min(retainedHeight, Math.round((by2 + height * CROP_MARGIN) * scaleY));
  if (x2 - x1 < CROP_MIN_WIDTH || y2 - y1 < CROP_MIN_HEIGHT) return null;
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

function buildCropEnvelope(header, jpegBlob) {
  // Single binary message: b"YOLO" + 0x01 + 0x02 + u32BE(header_len) +
  // JSON header + JPEG. No base64 uploads.
  const json = JSON.stringify(header);
  const jsonBytes = new TextEncoder().encode(json);
  const prefix = new Uint8Array(10);
  prefix[0] = 0x59; prefix[1] = 0x4f; prefix[2] = 0x4c; prefix[3] = 0x4f;
  prefix[4] = 0x01; prefix[5] = 0x02;
  new DataView(prefix.buffer).setUint32(6, jsonBytes.length, false);
  return new Blob([prefix, jsonBytes, jpegBlob], { type: "application/octet-stream" });
}

function pruneRetainedFrames(now = Date.now()) {
  for (const [frameId, entry] of retainedFrames) {
    if (now - entry.at > RETAINED_FRAME_TTL_MS) retainedFrames.delete(frameId);
  }
  while (retainedFrames.size > MAX_RETAINED_FRAMES) {
    retainedFrames.delete(retainedFrames.keys().next().value);
  }
}

function clearRetainedFrames() {
  pendingUpload = null;
  retainedFrames.clear();
}

function handleCropRequests(cropRequests, frameId) {
  if (!Array.isArray(cropRequests) || !cropRequests.length) return;
  const connection = socket;
  if (!connection || connection.readyState !== WebSocket.OPEN) return;
  const generation = connectionGeneration;
  const entry = retainedFrames.get(frameId);
  if (!entry) return;
  for (const request of cropRequests) {
    if (!request || request.frame_id !== frameId) continue;
    if (typeof request.session_version === "number" && request.session_version < sessionVersion) continue;
    // The request's upload dimensions describe the small detection JPEG;
    // the retained frame is the full-resolution original from the same
    // capture, so validate against the stored detection size and map the
    // 640x480 box to the retained original.
    if (request.upload_width !== entry.detectWidth || request.upload_height !== entry.detectHeight) continue;
    const rect = mapCropRect(request.bbox, entry.width, entry.height);
    if (!rect) continue;
    const source = entry.canvas;
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = rect.w;
    cropCanvas.height = rect.h;
    try {
      cropCanvas.getContext("2d", { alpha: false, desynchronized: true })
        .drawImage(source, rect.x, rect.y, rect.w, rect.h, 0, 0, rect.w, rect.h);
    } catch (error) {
      continue;
    }
    const header = {
      request_id: request.request_id,
      frame_id: request.frame_id,
      track_id: request.track_id,
      session_version: request.session_version,
    };
    cropCanvas.toBlob((blob) => {
      if (socket !== connection || generation !== connectionGeneration) return;
      if (!blob) return;
      if (!running || connection.readyState !== WebSocket.OPEN) return;
      try {
        if (diagEnabled) { diag.crops += 1; diag.cropBytes += blob.size || 0; }
        connection.send(buildCropEnvelope(header, blob));
      } catch (error) {
        // Crop delivery is best-effort; detection continues regardless.
      }
      // Crop sends never touch frameInFlight and never trigger detection.
    }, "image/jpeg", CROP_JPEG_QUALITY);
  }
}

function setStatus(message) {
  statusText.textContent = message;
}

function renderCounts(counts, totalElement, listElement, emptyText) {
  const entries = Object.entries(counts).sort(([a], [b]) => a.localeCompare(b));
  totalElement.textContent = entries.reduce((sum, [, count]) => sum + count, 0);
  listElement.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("div");
    empty.className = "empty-count";
    empty.textContent = emptyText;
    listElement.appendChild(empty);
  }
  for (const [label, count] of entries) {
    const row = document.createElement("div");
    row.className = "class-row";
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("span");
    value.className = "count";
    value.textContent = count;
    row.append(name, value);
    listElement.appendChild(row);
  }
}

function renderIdentification(identification) {
  const entries = Object.entries(identification || {}).sort(([a], [b]) => Number(a) - Number(b));
  identList.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("div");
    empty.className = "empty-count";
    empty.textContent = "No tracked bottles yet";
    identList.appendChild(empty);
    return;
  }
  for (const [trackId, info] of entries) {
    const row = document.createElement("div");
    row.className = "class-row";
    const thumb = document.createElement("img");
    thumb.className = "crop-thumb";
    thumb.alt = "";
    if (info.submitted) {
      thumb.src = `/api/ident-crop/${trackId}?s=${info.submitted}`;
    }
    const name = document.createElement("span");
    name.textContent = `#${trackId} ${info.hint || ""}`.trim();
    const value = document.createElement("span");
    value.className = "count";
    if (info.status === "candidate" && info.choice) {
      value.textContent = info.confidence == null ? `${info.choice}?` : `${info.choice}? ${info.confidence.toFixed(2)}`;
      if (info.complete) value.textContent = `${info.choice} ${info.confidence.toFixed(2)} · OCR paused`;
    } else if (info.status === "unknown") {
      value.textContent = "unknown";
    } else if (info.lines) {
      value.textContent = `need evidence (${info.lines} lines)`;
    } else {
      value.textContent = `collecting… (sharp ${info.sharpness ?? "?"})`;
    }
    row.append(thumb, name, value);
    identList.appendChild(row);
  }
}

async function loadReadiness() {
  try {
    const response = await fetch("/api/ident-readiness");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const state = await response.json();
    const catalog = state.catalog_ok ? `catalog: ${state.products.length} products` : "catalog: MISSING";
    const ocr = state.ocr_available ? "OCR CPU: ready" : (state.ocr_error ? "OCR CPU: unavailable" : "OCR CPU: starting…");
    const jev = state.jev_key_present ? "Jev key: set" : "Jev key: MISSING (add TYPESAFE_API_KEY to .env)";
    identReadiness.textContent = `${catalog} · ${ocr} · ${jev}`;
  } catch (error) {
    identReadiness.textContent = `Cannot load identification status: ${error.message}`;
  }
}

function renderSession(response) {
  // A frame sent before reset may arrive after the reset HTTP response.
  if (response.session_version < sessionVersion) return false;
  sessionVersion = response.session_version;
  if (response.model_label) modelInfo.textContent = response.model_label;
  renderCounts(response.packed_counts || {}, packedTotal, packedCounts, "Nothing packed yet");
  const event = response.last_event;
  lastEvent.textContent = event
    ? `Last packed: ${event.class_name} #${event.track_id} · ${new Date(event.timestamp).toLocaleTimeString()}`
    : "Waiting for a transfer";
  return true;
}

function base64ToBlob(base64Text) {
  const binary = atob(base64Text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: "image/jpeg" });
}

function showImage(base64Text) {
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = URL.createObjectURL(base64ToBlob(base64Text));
  resultImage.src = resultUrl;
  viewer.classList.add("has-image");
}

async function start() {
  if (startButton.disabled) return;
  const generation = ++connectionGeneration;
  startButton.disabled = true;
  stopButton.disabled = false;
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Camera access requires localhost or HTTPS.");
    }
    setStatus("Requesting camera permission...");
    let acquiredStream = null;
    try {
      acquiredStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 960 }, facingMode: { ideal: "environment" } },
        audio: false
      });
    } catch (error) {
      // Single-camera laptops often reject the facing-mode constraint; retry plainly.
      if (error && (error.name === "OverconstrainedError" || error.name === "NotFoundError")) {
        acquiredStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } else {
        throw error;
      }
    }
    if (generation !== connectionGeneration) {
      acquiredStream.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = acquiredStream;
    camera.srcObject = stream;
    camera.muted = true;
    // play() can hang forever on a camera that yields no frames; fail loudly instead.
    await Promise.race([
      camera.play(),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Camera did not start producing frames.")), 10000)
      ),
    ]);
    if (generation !== connectionGeneration) return;
    setStatus("Connecting to packing server...");
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const connection = new WebSocket(`${protocol}://${location.host}/ws/detect`);
    socket = connection;
    sessionVersion = -1;
    clearRetainedFrames();
    connection.onopen = () => {
      if (socket !== connection) return;
      running = true;
      setStatus("Connected. Move products into the yellow bag zone.");
      sendNextFrame();
    };
    connection.onmessage = (event) => {
      if (socket !== connection) return;
      let response = null;
      try {
        response = JSON.parse(event.data);
      } catch (error) {
        stop(`Cannot process camera result: ${error.message}`);
        return;
      }
      // Crop acknowledgments must not clear the detection-in-flight flag
      // and must never trigger a duplicate detection send.
      if (response && response.type === "crop_ack") return;
      if (response && response.error) {
        stop(`Server: ${response.error}`);
        return;
      }
      frameInFlight = false;
      try {
        // Missing type means a legacy detection response (tests/back-compat).
        if (renderSession(response)) {
          showImage(response.image);
          renderCounts(response.visible_counts || {}, totalCount, classCounts, "No tracked products visible");
          renderIdentification(response.identification);
        }
        if (diagEnabled && typeof response.detect_ms === "number") {
          diag.rttMs = diagClock() - diagSentAt;
          diag.backendMs = response.detect_ms;
          diagTimes.push(diagClock());
          renderDiag();
        }
        if (typeof response.frame_id === "number" && pendingUpload) {
          retainedFrames.set(response.frame_id, pendingUpload);
          pendingUpload = null;
          pruneRetainedFrames();
        } else {
          pendingUpload = null;
        }
        // Stale sessions render nothing and request no crops, but the next
        // detection still goes out to keep the single-frame pipeline moving.
        if (response.session_version >= sessionVersion) {
          handleCropRequests(response.crop_requests, response.frame_id);
        }
        if (typeof response.frame_id === "number") {
          retainedFrames.delete(response.frame_id);
          pruneRetainedFrames();
        }
        sendNextFrame();
      } catch (error) {
        stop(`Cannot process camera result: ${error.message}`);
      }
    };
    connection.onerror = () => {
      if (socket === connection) stop("Connection failed. Check that the server is running.");
    };
    connection.onclose = () => {
      if (socket === connection) stop("Server connection closed.");
    };
  } catch (error) {
    const reason = error && error.name ? `${error.name}: ${error.message}` : error.message;
    if (generation === connectionGeneration) stop(`Cannot start camera: ${reason}`);
  }
}

function sendNextFrame() {
  if (!running || resetting || frameInFlight || !socket || socket.readyState !== WebSocket.OPEN) return;
  if (camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    const generation = connectionGeneration;
    requestAnimationFrame(() => {
      if (generation === connectionGeneration) sendNextFrame();
    });
    return;
  }
  // Stage 2: one grab from the live video at capture resolution
  // (<=1280px wide, unchanged). The 640x480 detection upload is derived
  // from that same grab with a plain stretch to exactly DETECT_WIDTH x
  // DETECT_HEIGHT, matching the server's cv2.resize with no letterbox or
  // crop, so detection boxes, ROI alignment and the 640x480 coordinate
  // space are preserved. The full-resolution grab is retained for OCR.
  const sourceWidth = camera.videoWidth || 640;
  const sourceHeight = camera.videoHeight || 480;
  const scale = Math.min(1, 1280 / sourceWidth);
  const fullWidth = Math.round(sourceWidth * scale);
  const fullHeight = Math.round(sourceHeight * scale);
  const full = document.createElement("canvas");
  full.width = fullWidth;
  full.height = fullHeight;
  full.getContext("2d", { alpha: false, desynchronized: true }).drawImage(camera, 0, 0, fullWidth, fullHeight);
  canvas.width = DETECT_WIDTH;
  canvas.height = DETECT_HEIGHT;
  canvas.getContext("2d", { alpha: false, desynchronized: true }).drawImage(full, 0, 0, DETECT_WIDTH, DETECT_HEIGHT);
  pendingUpload = {
    canvas: full, width: fullWidth, height: fullHeight,
    detectWidth: DETECT_WIDTH, detectHeight: DETECT_HEIGHT, at: Date.now(),
  };
  frameInFlight = true;
  const connection = socket;
  const sentAt = diagClock();
  canvas.toBlob((blob) => {
    if (socket !== connection) return;
    if (!blob) {
      pendingUpload = null;
      stop("Could not capture camera frame.");
      return;
    }
    if (resetting || !running || connection.readyState !== WebSocket.OPEN) {
      pendingUpload = null;
      frameInFlight = false;
      return;
    }
    if (diagEnabled) { diag.detBytes = blob.size || 0; diagSentAt = sentAt; }
    connection.send(blob);
  }, "image/jpeg", DETECT_JPEG_QUALITY);
}

function stop(message = "Stopped. Packed items are preserved.") {
  connectionGeneration += 1;
  running = false;
  frameInFlight = false;
  clearRetainedFrames();
  const connection = socket;
  socket = null;
  if (connection) connection.close();
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  camera.srcObject = null;
  if (resultUrl) URL.revokeObjectURL(resultUrl);
  resultUrl = null;
  resultImage.removeAttribute("src");
  viewer.classList.remove("has-image");
  startButton.disabled = false;
  stopButton.disabled = true;
  renderCounts({}, totalCount, classCounts, "Camera stopped");
  renderIdentification({});
  setStatus(message);
}

async function resetSession() {
  if (resetting) return;
  resetting = true;
  resetButton.disabled = true;
  // Drop retained uploads: pending crop requests are tied to the old
  // session and the server rejects them after the version bump.
  clearRetainedFrames();
  try {
    const response = await fetch("/api/reset", { method: "POST" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderSession(await response.json());
    renderIdentification({});
    loadReadiness();
    // Remove the old annotated frame, which still contains the previous counts.
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = null;
    resultImage.removeAttribute("src");
    viewer.classList.remove("has-image");
    setStatus("Session reset. Move products outside the bag zone before packing.");
  } catch (error) {
    setStatus(`Could not reset session: ${error.message}`);
  } finally {
    resetting = false;
    resetButton.disabled = false;
    sendNextFrame();
  }
}

async function loadSession() {
  const initialGeneration = connectionGeneration;
  try {
    const response = await fetch("/api/session");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const session = await response.json();
    if (initialGeneration === connectionGeneration) renderSession(session);
  } catch (error) {
    if (initialGeneration === connectionGeneration) setStatus(`Cannot load session: ${error.message}`);
  }
}

startButton.addEventListener("click", start);
stopButton.addEventListener("click", () => stop());
resetButton.addEventListener("click", resetSession);
window.addEventListener("beforeunload", () => stop());
loadSession();
loadReadiness();
setInterval(loadReadiness, 5000);
