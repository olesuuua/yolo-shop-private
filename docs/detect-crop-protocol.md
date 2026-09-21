# Stage 1: detection frames vs OCR crops protocol

Goal: the browser supplies high-resolution OCR crops from the exact upload
frame used for detection. CPU inference, detection upload size/quality,
tracking, packing and annotated display are unchanged.

## Messages

### Client → server (binary WebSocket frames)

One binary message per upload. The first four bytes decide routing:

- **Detection frame**: raw JPEG bytes (unchanged stage-1 upload: canvas
  scaled to ≤1280 px wide, `image/jpeg` quality `0.85`, one frame in
  flight). JPEG magic (`FF D8 …`) never equals the crop magic below.
- **OCR crop response**: enveloped binary, no base64:
  `b"YOLO" (4B) + version u8 (0x01) + kind u8 (0x02) + header_len u32BE +
  JSON header (utf-8) + JPEG bytes`.

  Crop JSON header:
  `{"request_id": str, "frame_id": int>0, "track_id": int,
  "session_version": int}`.

Text frames from the client are rejected with a fatal
`{"error": ...}` (use binary).

### Server → client (JSON text)

- **Detection response**: `{"type": "detection", "frame_id": int,
  "session_version": int, "image": base64-annotated-640x480 JPEG,
  "visible_counts": {...}, "packed_counts": {...}, "packed_total": int,
  "last_event": ..., "events": [...], "identification": {...},
  "crop_requests": [...]}`.
- **Crop request** (inside a detection response):
  `{"request_id": str, "frame_id": int, "track_id": int,
  "session_version": int, "bbox": [x1,y1,x2,y2],
  "coord_space": "detect_640x480", "upload_width": int,
  "upload_height": int, "margin": 0.08}`.

  `bbox` uses **detection pixels**: 640×480 (`FRAME_WIDTH`×`FRAME_HEIGHT`),
  origin top-left, `x1 < x2`, `y1 < y2`. `upload_width/height` are the
  server-decoded upload dimensions for that `frame_id`.
- **Crop ack**: `{"type": "crop_ack", "request_id": str, "ok": bool,
  "reason": str}`. Reasons: `accepted`, `blurry`, `throttled`,
  `complete`, `too_small`, `oversized`, `empty`, `undecodable`,
  `mismatched`, `unknown_request` (stale/duplicate/after reset or
  disconnect), `expired` (folded into `unknown_request` after lazy sweep),
  `session_mismatch` (after Reset), `connection_mismatch`,
  `track_expired` (pruned or ID reused), `ocr_unavailable`, `internal`.
- **Fatal error**: `{"error": str}` (malformed frame, second camera, …).

## Browser rules

- Assign nothing: the server assigns `frame_id` per processed detection
  frame. The browser retains the exact uploaded pixels (offscreen canvas
  copy at upload resolution) as `pendingUpload` until the detection
  response arrives, then moves it into `retainedFrames[frame_id]`.
- Cut each requested crop from `retainedFrames[frame_id]` only. Never use
  a newer live frame. Skip when the entry is missing, the session is
  stale (`request.session_version < sessionVersion`), or retained
  dimensions differ from `upload_width/height`.
- Mapping (mirrors `upload_crop_rect` in `vision.py`):
  `scale_x = retainedWidth / 640`, `scale_y = retainedHeight / 480`;
  expand the box by `margin` (8 % of box w/h per side), round, clamp to
  retained bounds, drop when `< 60×80` px in upload space.
- Crop JPEG quality `0.85`. Send one enveloped binary message per crop.
  Crop sends never set/clear `frameInFlight` and never trigger the next
  detection send.
- Detection flow is unchanged: one frame in flight; `frameInFlight` is
  cleared only by detection responses (or legacy responses with `image`
  and no `type`); `crop_ack` returns early. `sendNextFrame()` runs right
  after a detection response so OCR never gates detection.
- Bound memory: at most 4 retained frames + 15 s TTL, pruned on every
  response. `clearRetainedFrames()` runs on Start (new socket), Stop,
  Reset, and capture errors/disconnects.

## Backend rules

- `FrameProcessor.process_detect()` decodes the upload, runs detection +
  `PackingTracker.update()`, touches/prunes identification liveness, then
  issues ≤ `MAX_CROP_REQUESTS_PER_FRAME` (4) requests from eligible
  tracks: in-tracker, not `is_complete()`, `request_due()` throttle peek
  (same intervals as `submit()` without consuming quota), expected padded
  size ≥ minimum. Global cap `MAX_PENDING_CROP_REQUESTS` (16); per-track
  latest replaces an unanswered request, preserving accumulated OCR lines.
- No server-side crop extraction in stage 1: the old
  `hires_crop → note → submit` path inside detection is removed, so old
  and new paths cannot double-submit. Image-dependent checks (minimum
  size, Laplacian sharpness ≥ `OCR_SHARPNESS_MIN`, JPEG validity) run on
  crop receipt; `note()` records collection state and `submit()` re-checks
  completion/throttling before queueing background OCR. Duplicate-image
  `dhash` dedup, Jev debounce and per-track slots are reused untouched.
- `process_crop_response()` never touches the tracker, frame timers,
  session counters or annotated image. It consumes the pending entry
  first (duplicates find nothing), then validates connection/session/
  frame/track, decodes, checks `track_expired` via identifier object
  identity (prune/reset/reused ID ⇒ reject), sharpness and submit gates.
- Requests are tied to `(connection_id, session_version, evidence object)`.
  `reset()` and disconnect/connect clear all pending; expiry
  (`CROP_REQUEST_TTL_S` = 8 s) is swept on every detection and crop.
- `app.py` routes binary by magic (`parse_client_message`), passes a
  per-connection UUID, and sends `crop_ack` for crops vs full detection
  JSON for frames. Single-camera, shared session, CPU-only inference and
  the annotated display are preserved.

## Stage 2: small detection upload, full-resolution retained original

- **Detection upload**: exactly 640×480 (`DETECT_WIDTH`×`DETECT_HEIGHT`),
  JPEG quality `0.75`, plain stretch with no letterbox or crop — the same
  transform as the server's `cv2.resize(full_frame, (640, 480))`, so
  detection boxes, `BAG_ROI` alignment and the `detect_640x480` coordinate
  space are unchanged. `upload_width/height` in crop requests is therefore
  always 640×480.
- **Retained original**: one grab from the live video at capture
  resolution (≤1280 px wide, JPEG source quality path unchanged); the
  640×480 detection canvas is drawn from that same grab
  (video → full canvas → detection canvas), never a second live draw.
  OCR crops are cut from the retained full-resolution frame at crop JPEG
  quality `0.85` with the unchanged 8 % margin and 60×80 px minimum.
- **Validation change**: the browser checks `request.upload_width/height`
  against the stored *detection* size (640×480) and maps the 640×480 box
  to the *retained* size (`scale = retained / 640,480`). Session, frame,
  clamping, TTL and memory bounds are unchanged.
- **Diagnostics**: detection responses carry `detect_ms` (pure server
  processing for that frame, network-exclusive) and `upload_bytes`
  (received detection JPEG size). The browser adds opt-in (`?diag=1`)
  overlay stats: detection bytes, send-to-response RTT (network-inclusive),
  backend `detect_ms`, received results/s, crop count and bytes.
