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
    connection.onopen = () => {
      if (socket !== connection) return;
      running = true;
      setStatus("Connected. Move products into the yellow bag zone.");
      sendNextFrame();
    };
    connection.onmessage = (event) => {
      if (socket !== connection) return;
      frameInFlight = false;
      try {
        const response = JSON.parse(event.data);
        if (response.error) {
          stop(`Server: ${response.error}`);
          return;
        }
        if (renderSession(response)) {
          showImage(response.image);
          renderCounts(response.visible_counts || {}, totalCount, classCounts, "No tracked products visible");
          renderIdentification(response.identification);
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
  // Detection runs on 640x480 server-side; uploads keep the higher camera
  // resolution so OCR crops stay legible. One frame in flight at a time.
  const sourceWidth = camera.videoWidth || 640;
  const sourceHeight = camera.videoHeight || 480;
  const scale = Math.min(1, 1280 / sourceWidth);
  canvas.width = Math.round(sourceWidth * scale);
  canvas.height = Math.round(sourceHeight * scale);
  canvas.getContext("2d", { alpha: false, desynchronized: true }).drawImage(camera, 0, 0, canvas.width, canvas.height);
  frameInFlight = true;
  const connection = socket;
  canvas.toBlob((blob) => {
    if (socket !== connection) return;
    if (!blob) {
      stop("Could not capture camera frame.");
      return;
    }
    if (resetting || !running || connection.readyState !== WebSocket.OPEN) {
      frameInFlight = false;
      return;
    }
    connection.send(blob);
  }, "image/jpeg", 0.85);
}

function stop(message = "Stopped. Packed items are preserved.") {
  connectionGeneration += 1;
  running = false;
  frameInFlight = false;
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
