"""YOLO adapter, drawing and serialized access to one packing session."""

import base64
import logging
import time
from collections import Counter, deque
from threading import Lock

import cv2
import numpy as np

from config import (
    AGNOSTIC_NMS, CONF_THRESHOLD, CROP_REQUEST_TTL_S, FOOD_CLASSES,
    FRAME_HEIGHT, FRAME_WIDTH,
    MAX_CROP_REQUESTS_PER_FRAME, MAX_CROP_RESPONSE_BYTES, MAX_FRAME_BYTES,
    MAX_PACKED_OVERLAY_ROWS, MAX_PENDING_CROP_REQUESTS, NMS_IOU_THRESHOLD,
    MODEL_LABEL, MODEL_PROFILE, PACKED_BANNER_FRAMES, TRACKER_CONFIG,
    INFERENCE_DEVICE, INFERENCE_SIZE, OCR_CROP_JPEG_QUALITY,
    OCR_CROP_MARGIN, OCR_MIN_CROP_HEIGHT, OCR_MIN_CROP_WIDTH,
    OCR_SHARPNESS_MIN,
)
from tracking import Detection, PackingTracker


logger = logging.getLogger(__name__)

# Stage 1 WebSocket envelope for browser-supplied OCR crops.
# Detection frames stay raw JPEG binary (unchanged). Crop responses are a
# single binary message so no base64 upload is needed:
#   magic b"YOLO" (4B) + version u8 (0x01) + kind u8 (0x02=crop)
#   + header_len u32BE + JSON header (utf8) + JPEG bytes.
# Crop JSON header: {request_id, frame_id, track_id, session_version}.
# JPEG magic (FF D8) never collides with b"YOLO", so raw uploads route to
# detection and enveloped messages route to the OCR pipeline.
WS_MAGIC = b"YOLO"
WS_VERSION = 1
WS_KIND_CROP = 2
WS_HEADER_MAX = 4096

# Bounding boxes in crop requests use detection pixels.
CROP_COORD_SPACE = "detect_640x480"


def extract_food_detections(result) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return []
    detections = []
    for bbox, class_id, track_id in zip(
        boxes.xyxy.cpu().tolist(), boxes.cls.int().cpu().tolist(),
        boxes.id.int().cpu().tolist(),
    ):
        class_name = result.names[class_id]
        if class_name in FOOD_CLASSES:
            detections.append(Detection(track_id, class_name, tuple(bbox)))
    return detections


def draw_label(frame, text, origin, color=(255, 255, 255), scale=0.6):
    # Solid pill background: legible on light bottles and dark scenes alike.
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = origin
    pad = 5
    top = max(0, y - height - pad * 2)
    cv2.rectangle(frame, (max(0, x - pad), top),
                  (x + width + pad, y + baseline // 2), (15, 20, 25), -1)
    cv2.putText(frame, text, (max(0, x), top + height + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def crop_sharpness(crop) -> float:
    """Laplacian variance; blurry or flat crops score low and are skipped."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def upload_crop_rect(bbox_640x480, upload_width, upload_height,
                      margin=OCR_CROP_MARGIN):
    """Padded crop rectangle in upload pixels, or None when too small.

    Geometry mirrors hires_crop without needing pixels, so the backend can
    filter ineligible tracks before requesting and the browser can apply
    the identical mapping. Input bbox uses detection pixels
    (FRAME_WIDTH x FRAME_HEIGHT, origin top-left). Output is clamped
    (x1, y1, x2, y2) ints in upload pixels.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox_640x480)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    if upload_width <= 0 or upload_height <= 0:
        return None
    scale_x, scale_y = upload_width / FRAME_WIDTH, upload_height / FRAME_HEIGHT
    width, height = x2 - x1, y2 - y1
    margin_x, margin_y = width * margin, height * margin
    out_x1 = max(0, int((x1 - margin_x) * scale_x))
    out_y1 = max(0, int((y1 - margin_y) * scale_y))
    out_x2 = min(upload_width, int((x2 + margin_x) * scale_x))
    out_y2 = min(upload_height, int((y2 + margin_y) * scale_y))
    if out_x2 - out_x1 < OCR_MIN_CROP_WIDTH or out_y2 - out_y1 < OCR_MIN_CROP_HEIGHT:
        return None
    return (out_x1, out_y1, out_x2, out_y2)


def hires_crop(full_frame, bbox_640x480):
    """Map a detection box to the high-resolution frame with a small margin."""
    full_height, full_width = full_frame.shape[:2]
    rect = upload_crop_rect(bbox_640x480, full_width, full_height)
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    return full_frame[y1:y2, x1:x2]


def build_crop_envelope(header: dict, jpeg_bytes: bytes) -> bytes:
    """Pack a crop response for tests/clients (browser uses Blob parts)."""
    import json
    import struct
    header_raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return (WS_MAGIC + bytes((WS_VERSION, WS_KIND_CROP))
            + struct.pack("!I", len(header_raw)) + header_raw + bytes(jpeg_bytes))


def parse_client_message(data: bytes):
    """Route one client binary message.

    Returns ("detection", {}, jpeg_bytes) for raw uploads or
    ("crop", header_dict, jpeg_bytes) for enveloped crop responses.
    Raises ValueError on malformed enveloped messages.
    """
    import json
    import struct
    if data is None:
        raise ValueError("Camera frame is empty or too large.")
    raw = bytes(data)
    if not raw.startswith(WS_MAGIC):
        return ("detection", {}, raw)
    if len(raw) < 10:
        raise ValueError("Crop message is too short.")
    version, kind = raw[4], raw[5]
    if version != WS_VERSION:
        raise ValueError("Unsupported crop message version.")
    if kind != WS_KIND_CROP:
        raise ValueError("Unsupported message kind.")
    (header_len,) = struct.unpack("!I", raw[6:10])
    if not 0 < header_len <= WS_HEADER_MAX:
        raise ValueError("Crop header has an invalid size.")
    if len(raw) < 10 + header_len + 1:
        raise ValueError("Crop message is truncated.")
    try:
        header = json.loads(raw[10:10 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("Crop header is not valid JSON.")
    if not isinstance(header, dict):
        raise ValueError("Crop header must be a JSON object.")
    return ("crop", header, raw[10 + header_len:])


def ident_label(status, choice, confidence) -> str:
    if status == "candidate" and choice:
        if confidence is None:
            return f"ID: {choice}?"
        return f"ID: {choice}? {confidence:.2f}"
    if status == "unknown":
        return "ID: unknown"
    return "ID: need evidence"


def annotate_frame(frame, detections, tracker, identification=None):
    identification = identification or {}
    for detection in detections:
        state = tracker.tracks.get(detection.track_id)
        if state is None:
            continue
        color = (90, 220, 100) if state.packed else (255, 185, 70)
        x1, y1, x2, y2 = map(int, detection.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{state.class_name} #{state.track_id}"
        if state.packed:
            label += " PACKED"
        elif state.inside_frames:
            label += f" entering {state.inside_frames}/{tracker.min_inside_frames}"
        draw_label(frame, label, (max(2, x1), max(44, y1 - 34)), color)
        info = identification.get(detection.track_id)
        if info is not None:
            # Identification sits directly above the box, under the class
            # label, so it is never hidden behind the bottle or the frame edge.
            draw_label(frame, ident_label(info.get("status"), info.get("choice"),
                                          info.get("confidence")),
                       (max(2, x1), max(18, y1 - 8)), (140, 220, 255), scale=0.55)

    x1, y1, x2, y2 = map(int, tracker.roi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 3)
    draw_label(frame, "BAG / PACKING ZONE", (x1, max(18, y1 - 12)), (0, 220, 255))

    rows = [f"Packed: {sum(tracker.packed_counts.values())}"]
    counts = sorted(tracker.packed_counts.items())
    rows.extend(f"{name}: {count}" for name, count in counts[:MAX_PACKED_OVERLAY_ROWS])
    if len(counts) > MAX_PACKED_OVERLAY_ROWS:
        rows.append(f"+{len(counts) - MAX_PACKED_OVERLAY_ROWS} more in sidebar")
    top = FRAME_HEIGHT - 16 - 22 * len(rows)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, top - 16), (214, FRAME_HEIGHT - 8), (15, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    for index, text in enumerate(rows):
        draw_label(frame, text, (16, top + index * 22), (150, 245, 160))

    if tracker.packed_events:
        event = tracker.packed_events[-1]
        if tracker.frame_number - event["frame_number"] < PACKED_BANNER_FRAMES:
            draw_label(frame, f"PACKED: {event['class_name']} #{event['track_id']}",
                       (16, 32), (120, 255, 140), scale=0.8)
    return frame


class FrameProcessor:
    """One camera/table per server process. The lock also serializes reset.

    Stage 1: detection frames (raw JPEG) advance the packing tracker and
    return OCR crop *requests*; browser-supplied crop responses are routed
    to the background OCR pipeline and never advance detection/tracking.
    """

    def __init__(self, model, identifier=None):
        self.model = model
        self.identifier = identifier
        self.tracker = PackingTracker()
        self.lock = Lock()
        self.session_version = 0
        self.frame_times = deque(maxlen=30)
        self.detect_ms_ema = 0.0
        # Server-issued detection frame IDs (stage 1 protocol).
        self.frame_seq = 0
        # Outstanding browser crop requests: request_id -> record. One slot
        # per track keeps the latest view; accumulated OCR evidence in the
        # identifier is never cleared by replacement.
        self.crop_pending = {}
        self.crop_by_track = {}
        self.crop_counter = 0
        self.connection_id = None
        missing_classes = FOOD_CLASSES - set(model.names.values())
        if missing_classes:
            raise ValueError(
                "The model is missing configured FOOD_CLASSES: "
                + ", ".join(sorted(missing_classes))
                + ". Use the weights matching LIGHTSTORE_MODEL or update FOOD_CLASSES."
            )
        self.food_class_ids = [
            class_id for class_id, name in model.names.items() if name in FOOD_CLASSES
        ]
        if not self.food_class_ids:
            raise ValueError("The model has no classes matching FOOD_CLASSES.")

    def reset(self) -> dict:
        with self.lock:
            self.tracker.reset()
            self.session_version += 1
            self.crop_pending = {}
            self.crop_by_track = {}
            if self.identifier is not None:
                self.identifier.reset()
            # Keep ByteTrack IDs continuous; the packing state starts fresh.
            return self._snapshot()

    def set_connection(self, connection_id) -> None:
        with self.lock:
            self.connection_id = connection_id
            self.crop_pending = {}
            self.crop_by_track = {}

    def clear_connection(self, connection_id) -> None:
        with self.lock:
            if self.connection_id == connection_id:
                self.connection_id = None
            self.crop_pending = {}
            self.crop_by_track = {}

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot()

    def _snapshot(self) -> dict:
        return {
            **self.tracker.snapshot(), "session_version": self.session_version,
            "model_profile": MODEL_PROFILE, "model_label": MODEL_LABEL,
            "inference_device": INFERENCE_DEVICE, "inference_size": INFERENCE_SIZE,
            "enabled_classes": [self.model.names[index] for index in self.food_class_ids],
        }

    def _expire_crops(self, now: float) -> None:
        for request_id, record in list(self.crop_pending.items()):
            if now - record["issued_at"] > CROP_REQUEST_TTL_S:
                del self.crop_pending[request_id]
                if self.crop_by_track.get(record["track_id"]) == request_id:
                    del self.crop_by_track[record["track_id"]]

    def _issue_crop_requests(self, detections, upload_width, upload_height,
                             frame_id, connection_id) -> list:
        """Eligible tracks for this frame; bounded, latest-per-track."""
        identifier = self.identifier
        if identifier is None:
            return []
        now = time.monotonic()
        self._expire_crops(now)
        requests = []
        for detection in detections:
            if len(requests) >= MAX_CROP_REQUESTS_PER_FRAME:
                break
            if detection.track_id not in self.tracker.tracks:
                continue
            if identifier.is_complete(detection.track_id):
                continue
            if not identifier.request_due(detection.track_id, detection.class_name):
                continue
            if upload_crop_rect(detection.bbox, upload_width, upload_height) is None:
                continue
            if (len(self.crop_pending) >= MAX_PENDING_CROP_REQUESTS
                    and detection.track_id not in self.crop_by_track):
                continue
            # Prefer the latest pending crop per track: drop the unanswered
            # request, keep already accumulated OCR evidence untouched.
            old = self.crop_by_track.pop(detection.track_id, None)
            if old is not None:
                self.crop_pending.pop(old, None)
            self.crop_counter += 1
            request_id = f"{frame_id}:{detection.track_id}:{self.crop_counter}"
            with identifier.lock:
                evidence_ref = identifier.tracks.get(detection.track_id)
            record = {
                "request_id": request_id, "frame_id": frame_id,
                "track_id": detection.track_id,
                "session_version": self.session_version,
                "connection_id": connection_id,
                "bbox": [float(v) for v in detection.bbox],
                "hint": detection.class_name,
                "issued_at": now, "evidence_ref": evidence_ref,
            }
            self.crop_pending[request_id] = record
            self.crop_by_track[detection.track_id] = request_id
            requests.append({
                "request_id": request_id, "frame_id": frame_id,
                "track_id": detection.track_id,
                "session_version": self.session_version,
                "bbox": [float(v) for v in detection.bbox],
                "coord_space": CROP_COORD_SPACE,
                "upload_width": upload_width, "upload_height": upload_height,
                "margin": OCR_CROP_MARGIN,
            })
        return requests

    def process(self, image_bytes: bytes, connection_id=None) -> dict:
        return self.process_detect(image_bytes, connection_id=connection_id)

    def process_detect(self, image_bytes: bytes, connection_id=None) -> dict:
        if not image_bytes or len(image_bytes) > MAX_FRAME_BYTES:
            raise ValueError("Camera frame is empty or too large.")
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        full_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if full_frame is None:
            raise ValueError("Could not decode camera frame.")
        # Detection keeps the small fixed input; OCR crops are supplied by
        # the browser from the exact retained upload (stage 1), never
        # extracted here, so old and new OCR paths cannot double-submit.
        frame = cv2.resize(full_frame, (FRAME_WIDTH, FRAME_HEIGHT))

        with self.lock:
            if connection_id is not None and self.connection_id is None:
                self.connection_id = connection_id
            detect_start = time.monotonic()
            result = self.model.track(
                frame, persist=True, tracker=TRACKER_CONFIG,
                classes=self.food_class_ids, conf=CONF_THRESHOLD,
                imgsz=INFERENCE_SIZE, device=INFERENCE_DEVICE, verbose=False,
                agnostic_nms=AGNOSTIC_NMS, iou=NMS_IOU_THRESHOLD,
                # Select the NMS head on YOLO26 so the same duplicate-box filter
                # used by the YOLO11 profiles remains active before ByteTrack.
                nms=True,
            )[0]
            detect_ms = (time.monotonic() - detect_start) * 1000
            self.detect_ms_ema = (detect_ms if not self.detect_ms_ema
                                  else 0.2 * detect_ms + 0.8 * self.detect_ms_ema)
            now = time.monotonic()
            self.frame_times.append(now)
            fps = (len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
                   if len(self.frame_times) >= 2 and self.frame_times[-1] > self.frame_times[0] else 0.0)
            detections = extract_food_detections(result)
            events = self.tracker.update(detections)
            full_height, full_width = full_frame.shape[:2]
            identification = {}
            crop_requests = []
            if self.identifier is not None:
                try:
                    active_ids = set()
                    for detection in detections:
                        active_ids.add(detection.track_id)
                        self.identifier.touch(detection.track_id, detection.class_name)
                    self.identifier.prune(active_ids)
                    crop_requests = self._issue_crop_requests(
                        detections, full_width, full_height,
                        self.frame_seq + 1, connection_id if connection_id is not None
                        else self.connection_id)
                    identification = self.identifier.snapshot()
                except Exception:
                    logger.exception("Identification update failed; detection continues.")
                    identification = self.identifier.snapshot()
                self.identifier.note_capture((full_width, full_height), fps,
                                             self.detect_ms_ema)
            self.frame_seq += 1
            frame_id = self.frame_seq
            for request in crop_requests:
                request["frame_id"] = frame_id
                record = self.crop_pending.get(request["request_id"])
                if record is not None:
                    record["frame_id"] = frame_id
            visible_counts = Counter(
                self.tracker.tracks[d.track_id].class_name for d in detections
                if d.track_id in self.tracker.tracks
            )
            frame = annotate_frame(frame, detections, self.tracker, identification)
            success, image = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80],
            )
            if not success:
                raise ValueError("Could not encode annotated frame.")
            return {
                "type": "detection",
                "frame_id": frame_id,
                "image": base64.b64encode(image.tobytes()).decode("ascii"),
                "counts": dict(visible_counts),  # Original frontend compatibility.
                "visible_counts": dict(visible_counts),
                **self._snapshot(),
                "identification": identification,
                "crop_requests": crop_requests,
                "events": events,  # Only newly registered events, in frame order.
            }

    def process_crop_response(self, header: dict, jpeg_bytes: bytes,
                              connection_id=None) -> dict:
        """Validate a browser crop and queue it for background OCR.

        Never advances the packing tracker or detection timing. Returns a
        crop_ack payload ({type, request_id, ok, reason}).
        """
        def ack(request_id, ok, reason):
            return {"type": "crop_ack", "request_id": request_id or "",
                    "ok": bool(ok), "reason": reason}

        request_id = header.get("request_id") if isinstance(header, dict) else None
        frame_id = header.get("frame_id") if isinstance(header, dict) else None
        track_id = header.get("track_id") if isinstance(header, dict) else None
        session_version = header.get("session_version") if isinstance(header, dict) else None
        if (not isinstance(request_id, str) or not request_id
                or not isinstance(frame_id, int) or isinstance(frame_id, bool)
                or frame_id <= 0
                or not isinstance(track_id, int) or isinstance(track_id, bool)
                or not isinstance(session_version, int)
                or isinstance(session_version, bool)):
            return ack(request_id if isinstance(request_id, str) else "",
                       False, "mismatched")
        now = time.monotonic()
        with self.lock:
            self._expire_crops(now)
            record = self.crop_pending.get(request_id)
            if record is None:
                return ack(request_id, False, "unknown_request")
            if record["frame_id"] != frame_id or record["track_id"] != track_id:
                return ack(request_id, False, "mismatched")
            if record["session_version"] != self.session_version:
                self.crop_pending.pop(request_id, None)
                if self.crop_by_track.get(track_id) == request_id:
                    del self.crop_by_track[track_id]
                return ack(request_id, False, "session_mismatch")
            if session_version != self.session_version:
                self.crop_pending.pop(request_id, None)
                if self.crop_by_track.get(track_id) == request_id:
                    del self.crop_by_track[track_id]
                return ack(request_id, False, "session_mismatch")
            if (record["connection_id"] is not None and connection_id is not None
                    and record["connection_id"] != connection_id):
                return ack(request_id, False, "connection_mismatch")
            # Consume immediately: duplicate deliveries find no pending entry.
            self.crop_pending.pop(request_id, None)
            if self.crop_by_track.get(track_id) == request_id:
                del self.crop_by_track[track_id]
            hint = record["hint"]
            evidence_ref = record["evidence_ref"]
        identifier = self.identifier
        if identifier is None:
            return ack(request_id, False, "ocr_unavailable")
        if not jpeg_bytes or len(jpeg_bytes) > MAX_CROP_RESPONSE_BYTES:
            return ack(request_id, False,
                       "empty" if not jpeg_bytes else "oversized")
        crop_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        crop = cv2.imdecode(crop_array, cv2.IMREAD_COLOR)
        if crop is None:
            return ack(request_id, False, "undecodable")
        crop_height, crop_width = crop.shape[:2]
        if crop_width < OCR_MIN_CROP_WIDTH or crop_height < OCR_MIN_CROP_HEIGHT:
            return ack(request_id, False, "too_small")
        with identifier.lock:
            current = identifier.tracks.get(track_id)
            if current is not evidence_ref:
                # Pruned, reset, or ID reused by a new bottle: old evidence
                # must never attach to the new track.
                return ack(request_id, False, "track_expired")
            if current.complete:
                return ack(request_id, False, "complete")
        sharpness = crop_sharpness(crop)
        try:
            identifier.note(track_id, hint, sharpness, (crop_width, crop_height))
        except Exception:
            logger.exception("Identification note failed for crop %s.", request_id)
            return ack(request_id, False, "internal")
        if sharpness < OCR_SHARPNESS_MIN:
            return ack(request_id, False, "blurry")
        submitted = identifier.submit(track_id, hint, bytes(jpeg_bytes),
                                      (crop_width, crop_height))
        if not submitted:
            if identifier.is_complete(track_id):
                return ack(request_id, False, "complete")
            if identifier.ocr_gave_up:
                return ack(request_id, False, "ocr_unavailable")
            return ack(request_id, False, "throttled")
        return ack(request_id, True, "accepted")
