const sourceSelect = document.getElementById("sourceSelect");
let videoUrl = null;
let paused = false;
let playToken = 0;
let inferenceDevice = "cpu";
let epoch = 0;
let resetResolve = null;
let statusPending = false;
let retryPending = false;
let sourceChange = Promise.resolve();
const isVideo = () => sourceSelect.value === "video";
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
// retained at capture resolution (at native video dimensions) for OCR. Both versions are
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
// Replay validation: Play starts detection only after one genuinely decoded
// video frame. requestVideoFrameCallback (when supported) fires only for a
// presented video frame; without it the fallback polls for
// readyState >= HAVE_CURRENT_DATA with nonzero video dimensions. Metadata,
// timeline progress or audio alone never count as proof.
const REPLAY_DECODE_TIMEOUT_MS = 10000;
const REPLAY_DECODE_FALLBACK_POLL_MS = 100;
const REPLAY_DECODE_MESSAGE =
  "This browser could not decode the video; try a compatible H.264 MP4.";
let replayValidation = null; // {settled, timers, rVFCId}
let replayDecoded = false;   // set once a decoded frame is verified per source
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

function clearReplayValidation() {
  const state = replayValidation;
  if (!state) return;
  replayValidation = null;
  // Cancel as obsolete: the caller (Stop, source change, pause) always
  // invalidates the validation context first, so this only cleans up.
  if (state.finish) state.finish("obsolete");
}

function failReplay(message) {
  clearReplayValidation();
  paused = true;
  running = false;
  camera.pause?.();
  setStatus(message);
}

// Wait for one genuinely decoded frame before replay submits detections.
// onReady runs only when a real decoded frame with nonzero dimensions has
// been presented (or is renderable via the documented fallback); onSettled
// always runs once, including obsolete/failed settles.
function startReplayValidation(onReady, onSettled) {
  const generation = connectionGeneration;
  const validationEpoch = epoch;
  const startedToken = playToken;
  clearReplayValidation();
  const state = { settled: false, timers: new Set(), rVFCId: null };
  replayValidation = state;
  const finish = (outcome, message) => {
    if (state.settled) return;
    const current = generation === connectionGeneration &&
      validationEpoch === epoch && startedToken === playToken;
    state.settled = true;
    if (replayValidation === state) replayValidation = null;
    for (const id of state.timers) clearTimeout(id);
    state.timers.clear();
    if (state.rVFCId != null && camera.cancelVideoFrameCallback) {
      camera.cancelVideoFrameCallback(state.rVFCId);
      state.rVFCId = null;
    }
    if (outcome !== "obsolete" && current) {
      if (outcome === "decoded") { replayDecoded = true; onReady(); }
      else failReplay(message);
    }
    onSettled();
  };
  state.finish = finish;
  const onFrame = (_now, metadata) => {
    if (state.settled || generation !== connectionGeneration ||
        validationEpoch !== epoch || startedToken !== playToken) return;
    const width = metadata && metadata.width ? metadata.width : camera.videoWidth;
    const height = metadata && metadata.height ? metadata.height : camera.videoHeight;
    if (width > 0 && height > 0) { finish("decoded"); return; }
    // A zero-size presentation cannot be trusted; wait for the next frame
    // or the bounded timeout.
    state.rVFCId = camera.requestVideoFrameCallback(onFrame);
  };
  if (typeof camera.requestVideoFrameCallback === "function") {
    state.rVFCId = camera.requestVideoFrameCallback(onFrame);
  } else {
    // Documented fallback without requestVideoFrameCallback: only a frame
    // renderable at the current position counts (readyState
    // HAVE_CURRENT_DATA) and video dimensions must be nonzero.
    const poll = () => {
      if (state.settled || generation !== connectionGeneration ||
          validationEpoch !== epoch || startedToken !== playToken) return;
      if (camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
          camera.videoWidth > 0 && camera.videoHeight > 0) {
        finish("decoded"); return;
      }
      state.timers.add(setTimeout(poll, REPLAY_DECODE_FALLBACK_POLL_MS));
    };
    poll();
  }
  state.timers.add(setTimeout(() => {
    if (state.settled || generation !== connectionGeneration) return;
    finish("timeout", `Replay stopped: no decoded video frame within ` +
      `${Math.round(REPLAY_DECODE_TIMEOUT_MS / 1000)} s — ${REPLAY_DECODE_MESSAGE}`);
  }, REPLAY_DECODE_TIMEOUT_MS));
}

function handleCropRequests(cropRequests, frameId) {
  if (!Array.isArray(cropRequests) || !cropRequests.length) return;
  const connection = socket;
  if (!connection || connection.readyState !== WebSocket.OPEN) return;
  const generation = connectionGeneration;
  const cropEpoch = epoch;
  const entry = retainedFrames.get(frameId);
  if (!entry) return;
  for (const request of cropRequests) {
    if (!request || request.frame_id !== frameId) continue;
    if (typeof request.session_version === "number" && request.session_version < sessionVersion) continue;
    // The request's upload dimensions describe the small detection JPEG;
    // the retained frame is the full-resolution original from the same
    // capture, so validate against the stored detection size and map the
    // 640x480 box to the retained original.
    if (request.upload_width !== entry.detectWidth || request.upload_height !== entry.detectHeight ||
        (request.coord_space && request.coord_space !== "detect_640x480")) {
      const rejected = recordCrop(request, entry, null);
      if (rejected) rejected.reason = "coordinate_mismatch";
      continue;
    }
    const rect = mapCropRect(request.bbox, entry.width, entry.height);
    const evidence = recordCrop(request, entry, rect);
    if (!rect) continue;
    const source = entry.canvas;
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = rect.w;
    cropCanvas.height = rect.h;
    try {
      cropCanvas.getContext("2d", { alpha: false, desynchronized: true })
        .drawImage(source, rect.x, rect.y, rect.w, rect.h, 0, 0, rect.w, rect.h);
    } catch (error) {
      if (evidence) { evidence.status = "rejected"; evidence.reason = "crop_draw_failed"; }
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
      if (!blob) {
        if (evidence) { evidence.status = "rejected"; evidence.reason = "jpeg_encoding_failed"; }
        return;
      }
      if (cropEpoch !== epoch || resetting || connection.readyState !== WebSocket.OPEN) return;
      try {
        if (diagEnabled) { diag.crops += 1; diag.cropBytes += blob.size || 0; }
        if (diagEnabled) header.diagnostics = true;
        saveCrop(evidence, blob, cropEpoch);
        connection.send(buildCropEnvelope(header, blob));
      } catch (error) {
        if (evidence) { evidence.status = "rejected"; evidence.reason = "send_failed"; }
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
    if (/apple|banana|orange|lemon|pear|peach|grape|mango|fruit/i.test(label)) {
      row.classList.add("fruit-row");
    }
    const name = document.createElement("span");
    // Fruit detections are first-class demo results, not background counts.
    name.textContent = (/apple/i.test(label) ? `🍎 ${label}` : label);
    const value = document.createElement("span");
    value.className = "count";
    value.textContent = count;
    row.append(name, value);
    listElement.appendChild(row);
  }
}

// MVP demo: prominent product cards beside the large video.
// States: "Reading label", "Candidate" (catalog suggestion, never verified),
// or "Needs more evidence". OCR evidence is shown verbatim with the
// volume tokens (0,33 / 0,75) highlighted so the water-bottle distinction
// is visible at a glance. Jev errors are shown as text; the key never
// leaves the backend.
let catalogBySku = {};
function highlightVolume(text) {
  const fragment = document.createDocumentFragment();
  const pattern = /0[.,]\s?33|0[.,]\s?75|0[.,]\s?5\b|330\s?мл|750\s?мл/gi;
  let last = 0;
  let match;
  const source = String(text);
  while ((match = pattern.exec(source)) !== null) {
    if (match.index > last) fragment.append(source.slice(last, match.index));
    const mark = document.createElement("mark");
    mark.textContent = match[0];
    fragment.append(mark);
    last = match.index + match[0].length;
  }
  fragment.append(source.slice(last));
  return fragment;
}

function isFruitHint(hint) {
  return /apple|banana|orange|lemon|pear|peach|grape|mango|pineapple|fruit|berries/i.test(hint || "");
}

function renderIdentification(identification) {
  const entries = Object.entries(identification || {}).sort(([a], [b]) => Number(a) - Number(b));
  identList.replaceChildren();
  const rawLines = [];
  if (!entries.length) {
    const empty = document.createElement("div");
    empty.className = "empty-count";
    empty.textContent = "No tracked products yet — start the camera and show a bottle.";
    identList.appendChild(empty);
  }
  for (const [trackId, info] of entries) {
    const card = document.createElement("article");
    card.className = "product-card";
    const thumb = document.createElement("img");
    thumb.className = "crop-thumb";
    thumb.alt = "";
    if (info.submitted) {
      thumb.src = `/api/ident-crop/${trackId}?s=${info.submitted}`;
    }
    const body = document.createElement("div");
    const title = document.createElement("h3");
    const catalog = (info.choice && catalogBySku[info.choice]) || null;
    if (info.choice && catalog) {
      title.textContent = `#${trackId} ${catalog.name}`;
    } else if (info.choice) {
      title.textContent = `#${trackId} ${info.choice}`;
    } else {
      title.textContent = `#${trackId} ${info.hint || "Unidentified item"}`;
    }
    body.append(title);
    // Prominent volume line: the demo succeeds only when 0,33 vs 0,75 is shown.
    if (catalog && catalog.size) {
      const sizeLine = document.createElement("div");
      sizeLine.className = "size-line";
      sizeLine.textContent = catalog.size;
      body.append(sizeLine);
    }
    const badge = document.createElement("span");
    if (info.status === "candidate" && info.choice) {
      badge.className = "badge candidate";
      const conf = info.confidence == null ? "" : ` · ${(info.confidence).toFixed(2)}`;
      badge.textContent = info.complete ? `Candidate${conf} · OCR paused` : `Candidate${conf}`;
    } else if ((info.lines || 0) === 0 && !info.choice) {
      badge.className = "badge reading";
      badge.textContent = "Reading label…";
    } else if (info.status === "unknown") {
      badge.className = "badge needs";
      badge.textContent = "Needs more evidence";
    } else if (!info.choice) {
      badge.className = "badge needs";
      badge.textContent = `Needs more evidence (${info.lines || 0} lines)`;
    } else {
      badge.className = "badge needs";
      badge.textContent = "Needs more evidence";
    }
    body.append(badge);
    if (isFruitHint(info.hint)) {
      const fruit = document.createElement("span");
      fruit.className = "badge fruit";
      fruit.textContent = `🍎 Fruit detected (${info.hint})`;
      body.append(fruit);
    }
    if (info.choice) {
      const note = document.createElement("div");
      note.className = "candidate-note";
      note.textContent = "Catalog candidate — not a verified SKU.";
      body.append(note);
    }
    const ocrBox = document.createElement("div");
    ocrBox.className = "ocr-evidence";
    const ocrTexts = Array.isArray(info.ocr_texts) ? info.ocr_texts.filter(Boolean) : [];
    if (ocrTexts.length) {
      ocrTexts.slice(0, 6).forEach((line, index) => {
        if (index > 0) ocrBox.append(document.createElement("br"));
        ocrBox.append(highlightVolume(line));
      });
    } else if ((info.lines || 0) > 0) {
      ocrBox.textContent = `${info.lines} OCR line(s) collected — text pending…`;
    } else {
      ocrBox.textContent = "Point the label at the camera — OCR text will appear here.";
    }
    body.append(ocrBox);
    const meta = document.createElement("div");
    meta.className = "card-meta";
    const retry = (info.jev_retry_in_s != null)
      ? ` · Jev retry in ${info.jev_retry_in_s}s` : "";
    meta.textContent = `Track #${trackId} · hint ${info.hint || "—"} · OCR runs ${info.ocr_runs ?? 0} · sharp ${info.sharpness ?? "?"}${retry}`;
    body.append(meta);
    if (info.jev_error) {
      const error = document.createElement("div");
      error.className = "card-error";
      error.textContent = `Jev: ${info.jev_error}`;
      body.append(error);
    }
    card.append(thumb, body);
    identList.appendChild(card);
    for (const line of ocrTexts.slice(0, 4)) {
      rawLines.push(`[#${trackId}] ${line}`);
    }
    if (info.jev_error) rawLines.push(`[#${trackId} Jev] ${info.jev_error}`);
  }
  const rawBox = document.getElementById("rawOcr");
  if (rawBox) rawBox.textContent = rawLines.length ? rawLines.join("\n") : "No OCR text yet.";
}

async function loadReadiness() {
  try {
    const response = await fetch("/api/ident-readiness");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const state = await response.json();
    catalogBySku = {};
    for (const product of (state.catalog_products || [])) {
      if (product && product.sku) catalogBySku[product.sku] = product;
    }
    const skus = state.products || [];
    const has033 = skus.includes("saint-spring-0-33l");
    const has075 = skus.includes("saint-spring-0-75l");
    const catalog = state.catalog_ok
      ? `catalog: ${skus.length} products${has033 && has075 ? " (0,33 л + 0,75 л ready)" : " (⚠ water sizes missing)"}`
      : "catalog: MISSING";
    const ocr = state.ocr_available ? "OCR CPU: ready" : (state.ocr_error ? "OCR CPU: unavailable" : "OCR CPU: starting…");
    const jev = state.jev_key_present ? "Jev key: set" : "Jev key: MISSING (add TYPESAFE_API_KEY to .env)";
    identReadiness.textContent = `${catalog} · ${ocr} · ${jev}`;
    const jevBox = document.getElementById("jevStatus");
    if (jevBox) {
      const budget = state.jev_budget_unlimited
        ? "attempts unlimited for this MVP test"
        : `budget ${state.jev_budget_remaining ?? "?"} left of ${state.jev_budget_max ?? "?"}`;
      jevBox.textContent = `Jev matching: ${state.jev_attempts ?? 0} attempts · ` +
        `${state.jev_successes ?? 0} ok · ${state.jev_failures ?? 0} failed · ` +
        `${budget}. Idle bottles never stream calls (12 s debounce + changed evidence).`;
    }
  } catch (error) {
    identReadiness.textContent = `Cannot load identification status: ${error.message}`;
  }
}

function renderSession(response) {
  // A frame sent before reset may arrive after the reset HTTP response.
  if (response.session_version < sessionVersion) return false;
  sessionVersion = response.session_version;
  if (response.inference_device) inferenceDevice = response.inference_device;
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
    if (!isVideo()) {
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
    } else {
      if (!videoUrl) throw new Error("Select a playable local video first.");
      camera.muted = true;
      paused = true;
    }
    if (generation !== connectionGeneration) return;
    if (socket?.readyState === WebSocket.OPEN) {
      running = true; sendNextFrame(); return;
    }
    setStatus("Connecting to packing server...");
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const connection = new WebSocket(`${protocol}://${location.host}/ws/detect`);
    socket = connection;
    sessionVersion = -1;
    clearRetainedFrames();
    connection.onopen = () => {
      if (socket !== connection) return;
      running = true;
      if (isVideo()) startButton.disabled = false;
      setStatus("Connected. Move products into the yellow bag zone.");
      if (isVideo()) resetSession().then(ok => { if (ok) playVideo(); }); else sendNextFrame();
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
      if (response.type === "reset_ack") {
        if (resetResolve) { const resolve = resetResolve; resetResolve = null; resolve(response); }
        return;
      }
      if (response.type === "crop_ack") {
        if (!resetting && response.session_version === sessionVersion) evidenceAck(response);
        return;
      }
      if (response.type === "status") {
        statusPending = false;
        if (!resetting && response.session_version === sessionVersion) {
          renderSession(response); renderIdentification(response.identification);
          mergeEvidence(response.diagnostics);
        }
        return;
      }
      if (response && response.error) {
        stop(`Server: ${response.error}`);
        return;
      }
      frameInFlight = false;
      if (resetting) { pendingUpload = null; return; }
      if (response.session_version < sessionVersion) return;
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
          recordFrame(response, pendingUpload);
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
  if (!running || (isVideo() && (paused || camera.ended)) || resetting || frameInFlight || !socket || socket.readyState !== WebSocket.OPEN) return;
  if (isVideo()) {
    // No detection upload before a genuinely decoded frame is verified,
    // and no fabricated 640x480 fallback when the video has no image.
    if (!replayDecoded || !(camera.videoWidth > 0) || !(camera.videoHeight > 0)) return;
  }
  if (camera.seeking || camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    if (retryPending) return;
    retryPending = true;
    const generation = connectionGeneration;
    requestAnimationFrame(() => {
      retryPending = false;
      if (generation === connectionGeneration) sendNextFrame();
    });
    return;
  }
  // Stage 2: one grab from the live video at capture resolution
  // (at native video dimensions). The 640x480 detection upload is derived
  // from that same grab with a plain stretch to exactly DETECT_WIDTH x
  // DETECT_HEIGHT, matching the server's cv2.resize with no letterbox or
  // crop, so detection boxes, ROI alignment and the 640x480 coordinate
  // space are preserved. The full-resolution grab is retained for OCR.
  const videoTimestamp = isVideo() ? camera.currentTime : null;
  // Camera keeps its legacy dimension fallback; video mode is validated
  // above, so the true video dimensions are always used there.
  const sourceWidth = isVideo() ? camera.videoWidth : (camera.videoWidth || 640);
  const sourceHeight = isVideo() ? camera.videoHeight : (camera.videoHeight || 480);
  const scale = 1;
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
    video_timestamp: videoTimestamp,
  };
  frameInFlight = true;
  const connection = socket;
  const captureEpoch = epoch;
  const sentAt = diagClock();
  canvas.toBlob((blob) => {
    if (socket !== connection || captureEpoch !== epoch) return;
    if (!blob) {
      pendingUpload = null;
      stop("Could not capture camera frame.");
      return;
    }
    if (resetting || !running || (isVideo() && (paused || camera.ended)) || connection.readyState !== WebSocket.OPEN) {
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
  statusPending = false;
  if (resetResolve) { resetResolve(null); resetResolve = null; }
  epoch += 1;
  clearReplayValidation();
  replayDecoded = false;
  camera.pause?.();
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = null;
  camera.removeAttribute("src");
  camera.load?.();
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
  if (resetting) return false;
  resetting = true;
  epoch += 1;
  clearEvidence();
  Object.assign(diag, {detBytes: 0, rttMs: null, backendMs: null, crops: 0, cropBytes: 0});
  diagTimes.length = 0;
  resetButton.disabled = true;
  // Drop retained uploads: pending crop requests are tied to the old
  // session and the server rejects them after the version bump.
  clearRetainedFrames();
  try {
    let state;
    if (socket?.readyState === WebSocket.OPEN) {
      state = await new Promise(resolve => { resetResolve = resolve; socket.send(JSON.stringify({type: "reset"})); });
    } else {
      const response = await fetch("/api/reset", { method: "POST" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state = await response.json();
    }
    if (!state) throw new Error("Connection closed during reset");
    frameInFlight = false;
    renderSession(state);
    renderCounts({}, totalCount, classCounts, "No tracked products visible");
    renderIdentification({});
    loadReadiness();
    // Remove the old annotated frame, which still contains the previous counts.
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = null;
    resultImage.removeAttribute("src");
    viewer.classList.remove("has-image");
    setStatus("Session reset. Move products outside the bag zone before packing.");
    return true;
  } catch (error) {
    running = false;
    setStatus(`Could not reset session: ${error.message}`);
    return false;
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

startButton.addEventListener("click", async () => { await sourceChange; return isVideo() ? playVideo() : start(); });
stopButton.addEventListener("click", () => stop());
resetButton.addEventListener("click", resetSession);
window.addEventListener("beforeunload", () => stop());
loadSession();
loadReadiness();
setInterval(loadReadiness, 5000);

async function changeSource(file = null) {
  paused = true;
  camera.pause?.();
  connectionGeneration += 1;
  clearReplayValidation();
  replayDecoded = false;
  const wasRunning = running;
  running = false;
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  camera.srcObject = null;
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = null;
  camera.removeAttribute("src");
  camera.load?.();
  if (!await resetSession()) return;
  if (file && isVideo()) {
    videoUrl = URL.createObjectURL(file);
    camera.src = videoUrl;
    camera.load?.();
  }
  running = wasRunning && !!socket;
  startButton.disabled = false;
  startButton.textContent = isVideo() ? "Play" : "Start camera";
  document.getElementById("videoControls").hidden = !isVideo();
  setStatus(isVideo() ? "Select a video, then Play." : "Camera selected.");
}
async function playVideo() {
  await sourceChange;
  if (!videoUrl) { setStatus("Select a local video first."); return; }
  if (resetting) return;
  if (!socket) { await start(); return; }
  if (inferenceDevice !== "cpu") {
    setStatus("Replay requires CPU. Restart the server with LIGHTSTORE_DEVICE=cpu."); return;
  }
  const playEpoch = epoch;
  const token = ++playToken;
  try {
    await camera.play();
  }
  catch { setStatus("Unsupported or unreadable video. Try MP4/H.264 or WebM."); return; }
  if (playEpoch !== epoch || token !== playToken || !isVideo()) return;
  // Submit detections only after one genuinely decoded frame is verified;
  // metadata, timeline progress or audio playback never count as proof.
  await new Promise((resolve) => startReplayValidation(() => {
    paused = false; running = true; startButton.disabled = false;
    setStatus("Playing video at normal speed. Pending frames are skipped.");
    sendNextFrame();
  }, () => resolve()));
}
function pauseVideo() {
  playToken++;
  clearReplayValidation();
  paused = true;
  camera.pause?.();
  setStatus("Paused. Pending identification continues.");
}
async function restartVideo() {
  if (!videoUrl || resetting) return;
  pauseVideo();
  if (!await resetSession()) return;
  camera.currentTime = 0;
  await playVideo();
}
sourceSelect.addEventListener("change", () => { sourceChange = sourceChange.then(() => changeSource()); });
document.getElementById("videoFile").addEventListener("change", event => {
  const file = event.target.files[0];
  sourceChange = sourceChange.then(() => changeSource(file));
});
document.getElementById("pauseVideo").addEventListener("click", pauseVideo);
document.getElementById("restartVideo").addEventListener("click", restartVideo);
camera.addEventListener("ended", () => { paused = true; setStatus("Video ended. Pending identification continues."); });
camera.addEventListener("error", () => {
  if (!isVideo()) return;
  // Never claim the codec is definitely at fault; point to a fix.
  failReplay(`Media error ${camera.error ? camera.error.code : ""} — ${REPLAY_DECODE_MESSAGE}`);
});
camera.addEventListener("timeupdate", () => {
  document.getElementById("videoTime").textContent = `${(camera.currentTime || 0).toFixed(2)} / ${Number.isFinite(camera.duration) ? camera.duration.toFixed(2) : "?"} s`;
});
setInterval(() => {
  if (socket?.readyState === WebSocket.OPEN && !resetting && !statusPending) {
    statusPending = true;
    socket.send(JSON.stringify({type: "status", diagnostics: diagEnabled}));
  }
}, 1000);

// JSON embeds lossless PNG originals and the exact uploaded JPEG bytes.
const evidenceFrames = [];
const evidenceRequests = [];
let evidenceDropped = 0;
let evidenceBytes = 0;
let backendDropped = 0;
const EVIDENCE_BYTES = 24 * 1024 * 1024;
function clearEvidence() {
  evidenceFrames.length = 0; evidenceRequests.length = 0;
  evidenceDropped = 0; evidenceBytes = 0; backendDropped = 0;
  if (diagEnabled) renderEvidence();
}
function boundEvidence() {
  while (evidenceFrames.length > 300) { evidenceFrames.shift(); evidenceDropped++; }
  while (evidenceRequests.length > 60 || evidenceBytes > EVIDENCE_BYTES) {
    const old = evidenceRequests.shift();
    evidenceBytes -= old.bytes || 0; evidenceDropped++;
  }
}
function recordFrame(response, entry) {
  if (!diagEnabled || response.session_version !== sessionVersion) return;
  evidenceFrames.push({frame_id: response.frame_id, session_version: response.session_version,
    video_timestamp: entry.video_timestamp, source_dimensions: [entry.width, entry.height],
    detection_dimensions: [entry.detectWidth, entry.detectHeight], tracks: response.tracks || []});
  boundEvidence();
}
function recordCrop(request, entry, rect) {
  if (!diagEnabled) return null;
  const record = {...request, coord_space: request.coord_space || "detect_640x480", video_timestamp: entry.video_timestamp, mapped_rect: rect,
    status: rect ? "encoding" : "rejected", reason: rect ? null : "invalid_or_small_rectangle",
    source_dimensions: [entry.width, entry.height], bytes: 0};
  // Refuse a large original before encoding; canvas RGBA size is a conservative bound.
  if (entry.width * entry.height * 4 > EVIDENCE_BYTES / 2) {
    record.image_dropped = "original exceeds per-image limit"; evidenceDropped++;
  } else {
    try {
      record.original_png = entry.canvas.toDataURL("image/png");
      record.bytes = record.original_png.length * 2;
    } catch {
      record.image_dropped = "original encoding failed"; evidenceDropped++;
    }
  }
  evidenceRequests.push(record); evidenceBytes += record.bytes;
  boundEvidence(); renderEvidence(); return record;
}
async function saveCrop(record, blob, captureEpoch) {
  if (!record) return;
  const data = new Uint8Array(await blob.arrayBuffer());
  if (captureEpoch !== epoch || !evidenceRequests.includes(record)) return;
  let binary = "";
  for (const byte of data) binary += String.fromCharCode(byte);
  record.crop_jpeg = "data:image/jpeg;base64," + btoa(binary);
  if (record.status === "encoding") record.status = "uploaded";
  const bytes = record.crop_jpeg.length * 2;
  record.bytes += bytes; evidenceBytes += bytes;
  boundEvidence(); renderEvidence();
}
function evidenceAck(ack) {
  const record = evidenceRequests.find(r => r.request_id === ack.request_id);
  if (record) Object.assign(record, {ack, status: ack.ok ? "accepted" : "rejected", reason: ack.reason});
  renderEvidence();
}
function mergeEvidence(data) {
  if (!diagEnabled || !data) return;
  backendDropped = data.dropped;
  for (const update of data.requests) {
    const record = evidenceRequests.find(r => r.request_id === update.request_id);
    if (record) record.backend = update;
  }
  renderEvidence();
}
function renderEvidence() {
  if (!diagEnabled) return;
  const panel = document.getElementById("evidencePanel"); panel.hidden = false;
  const select = document.getElementById("evidenceSelect");
  const selected = select.value;
  select.replaceChildren();
  for (const record of evidenceRequests) {
    const option = document.createElement("option"); option.value = record.request_id;
    option.textContent = `${record.request_id} · ${record.video_timestamp ?? "camera"}s · ${record.status}`;
    select.appendChild(option);
  }
  if (evidenceRequests.some(r => r.request_id === selected)) select.value = selected;
  document.getElementById("evidenceLoss").textContent = `Evicted/dropped: browser ${evidenceDropped}, server ${backendDropped}. Limits: 300 frames, 60 crops, 24 MiB image strings; server 200 requests.`;
  showEvidence();
}
function showEvidence() {
  const id = document.getElementById("evidenceSelect").value;
  const record = evidenceRequests.find(r => r.request_id === id);
  const original = document.getElementById("evidenceOriginal");
  const crop = document.getElementById("evidenceCrop");
  original.removeAttribute("src"); crop.removeAttribute("src");
  const box = document.getElementById("evidenceBox"); box.hidden = true;
  if (!record) return;
  if (record.original_png) original.src = record.original_png;
  if (record.crop_jpeg) crop.src = record.crop_jpeg;
  const [x1,y1,x2,y2] = record.bbox;
  Object.assign(box.style, {left: `${x1/640*100}%`, top: `${y1/480*100}%`, width: `${(x2-x1)/640*100}%`, height: `${(y2-y1)/480*100}%`});
  box.hidden = false;
  const {original_png, crop_jpeg, ...metadata} = record;
  document.getElementById("evidenceText").textContent = JSON.stringify(metadata, null, 2);
}
function exportEvidence() {
  const payload = {schema: "lightstore-replay-1", session_version: sessionVersion,
    sampling: "normal-speed live sampling; no frame backlog", dropped: evidenceDropped,
    backend_dropped: backendDropped, frames: evidenceFrames, requests: evidenceRequests};
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload)], {type: "application/json"}));
  const link = document.createElement("a"); link.href = url; link.download = "replay-evidence.json";
  link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
document.getElementById("evidenceSelect").addEventListener("change", showEvidence);
document.getElementById("exportEvidence").addEventListener("click", exportEvidence);
