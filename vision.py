"""YOLO adapter, drawing and serialized access to one packing session."""

import base64
import logging
import time
from collections import Counter, deque
from threading import Lock

import cv2
import numpy as np

from config import (
    AGNOSTIC_NMS, BAG_ZONE_MODE, CONF_THRESHOLD, CROP_REQUEST_TTL_S, FOOD_CLASSES,
    FRAME_HEIGHT, FRAME_WIDTH,
    MAX_CROP_REQUESTS_PER_FRAME, MAX_CROP_RESPONSE_BYTES, MAX_FRAME_BYTES,
    MAX_CROP_TOMBSTONES, MAX_PACKED_OVERLAY_ROWS, MAX_PENDING_CROP_REQUESTS,
    MAX_OUTSTANDING_CROP_REQUESTS_PER_TRACK, NMS_IOU_THRESHOLD,
    MODEL_LABEL, MODEL_PROFILE, PACKED_BANNER_FRAMES, TRACKER_CONFIG,
    INFERENCE_DEVICE, INFERENCE_SIZE, OCR_CROP_JPEG_QUALITY,
    OCR_CROP_MARGIN, OCR_MIN_CROP_HEIGHT, OCR_MIN_CROP_WIDTH,
    OCR_SHARPNESS_MIN, OCR_SHARPNESS_NORM_HEIGHT, OCR_SHARPNESS_NORM_MIN,
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
    # Solid charcoal pill background: legible on light bottles and dark scenes alike.
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = origin
    pad = 5
    top = max(0, y - height - pad * 2)
    cv2.rectangle(frame, (max(0, x - pad), top),
                  (x + width + pad, y + baseline // 2), (19, 23, 18), -1)
    cv2.putText(frame, text, (max(0, x), top + height + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


# Audience overlay palette (BGR), matching the landing page.
CHARCOAL = (19, 23, 18)
PAPER = (244, 248, 244)
LIME = (102, 245, 198)
MUTED = (165, 173, 162)
PEACH = (150, 180, 255)

TAB_SCALE = 0.55
TAB_PAD_X = 7
TAB_GAP = 4


def track_tab_text(class_name: str, track_id: int, identified: bool) -> str:
    """Short tab text: "Bottle #3", plus " ✓" only when identified.

    The checkmark means identification is confirmed — never packed state.
    Packed boxes read lime instead. Same pattern for cans, boxes, apples.
    """
    text = f"{class_name} #{track_id}"
    return text + " ✓" if identified else text


def tab_rect(text, box, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Tab rectangle attached to a box: above it, else below it, in-frame."""
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, TAB_SCALE, 2)
    tab_w, tab_h = text_w + TAB_PAD_X * 2, text_h + baseline + 8
    x1, y1, x2, y2 = (int(v) for v in box)
    x = min(max(x1, 2), max(2, width - tab_w - 2))
    y = y1 - tab_h - 2
    if y < 2:
        y = y2 + 2
    y = min(y, height - tab_h - 2)
    return (x, max(2, y), tab_w, tab_h)


def layout_tabs(specs, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Place tab rects so neighbors never overlap (greedy row stagger).

    ``specs``: list of (text, box). Returns list of (x, y, w, h) in the
    same order. Each tab first tries its attached position, then steps
    down until free; always clamped inside the frame.
    """
    placed = []
    for text, box in specs:
        x, y, tab_w, tab_h = tab_rect(text, box, width, height)
        while any(x < px + pw + TAB_GAP and px < x + tab_w + TAB_GAP
                  and y < py + ph + TAB_GAP and py < y + tab_h + TAB_GAP
                  for px, py, pw, ph in placed):
            y += tab_h + TAB_GAP
            if y + tab_h > height - 2:
                y = 2
                break
        placed.append((x, y, tab_w, tab_h))
    return placed


def draw_tab(frame, text, rect, fill, foreground):
    x, y, tab_w, tab_h = (int(v) for v in rect)
    cv2.rectangle(frame, (x, y), (x + tab_w, y + tab_h), fill, -1)
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, TAB_SCALE, 2)
    cv2.putText(frame, text, (x + TAB_PAD_X, y + tab_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, TAB_SCALE, foreground, 2,
                cv2.LINE_AA)


def crop_sharpness(crop) -> float:
    """Laplacian variance; blurry or flat crops score low and are skipped."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_sharpness_at_height(crop, height: int) -> float:
    """Laplacian variance after normalizing to a canonical height.

    Large native crops (up to ~950x1780 here) dilute full-resolution
    variance with smooth background, so readable close labels can score
    below the full-resolution gate. INTER_AREA downscale to a common text
    scale recovers them without re-running OCR. Returns 0.0 (fail-closed)
    on malformed input instead of raising.
    """
    try:
        if crop is None or getattr(crop, "size", 0) == 0:
            return 0.0
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0 or height <= 0:
            return 0.0
        if h == height:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            resized = cv2.resize(crop, (max(1, round(w * height / h)), height),
                                 interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except (cv2.error, ValueError, TypeError):
        return 0.0


def quality_gate_accepts(sharpness_full: float, sharpness_norm: float) -> bool:
    """Dual pre-OCR gate: keep every current accept, rescue large readable.

    Tuned on group-deduped video crops, confirmed held-out
    (reports/quality-gate/eval-table.md): union recovers +7 tuning / +4
    validation target-brand hits (incl. 388:2:31 Saint Spring) with zero
    extra empty OCRs and zero extra neighbor-brand cases on both splits.
    """
    from config import OCR_SHARPNESS_MIN, OCR_SHARPNESS_NORM_MIN
    try:
        full = float(sharpness_full)
    except (TypeError, ValueError):
        full = 0.0
    try:
        normed = float(sharpness_norm)
    except (TypeError, ValueError):
        normed = 0.0
    return full >= OCR_SHARPNESS_MIN or normed >= OCR_SHARPNESS_NORM_MIN


def letterbox_rect(upload_width, upload_height,
                   canvas_width=FRAME_WIDTH, canvas_height=FRAME_HEIGHT):
    """Content rectangle mapping an upload frame into the detection canvas.

    Aspect-preserving letterbox: 16:9 uploads land on a centered 640x360
    content area (60 px black bars top/bottom); 4:3 uploads are the identity.
    The browser applies the identical math when drawing its detection
    canvas, so detection boxes, bag geometry, and OCR coordinates share one
    undistorted 640x480 space. Rounding matches the JS mirror
    (round-half-up on exact .5 aside, irrelevant at real dimensions).
    """
    try:
        upload_width, upload_height = float(upload_width), float(upload_height)
    except (TypeError, ValueError):
        return (0, 0, canvas_width, canvas_height)
    if upload_width <= 0 or upload_height <= 0:
        return (0, 0, canvas_width, canvas_height)
    scale = min(canvas_width / upload_width, canvas_height / upload_height)
    width = int(round(upload_width * scale))
    height = int(round(upload_height * scale))
    return ((canvas_width - width) // 2, (canvas_height - height) // 2,
            width, height)


def letterbox_frame(full_frame):
    """Resize an upload frame into detection pixels without distortion."""
    height, width = full_frame.shape[:2]
    dx, dy, content_width, content_height = letterbox_rect(width, height)
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    resized = cv2.resize(full_frame, (content_width, content_height))
    canvas[dy:dy + content_height, dx:dx + content_width] = resized
    return canvas


def detect_to_upload(bbox_640x480, upload_width, upload_height):
    """Inverse of the letterbox: detection pixels -> upload pixels."""
    x1, y1, x2, y2 = (float(v) for v in bbox_640x480)
    dx, dy, content_width, content_height = letterbox_rect(
        upload_width, upload_height)
    scale_x = upload_width / content_width
    scale_y = upload_height / content_height
    return ((x1 - dx) * scale_x, (y1 - dy) * scale_y,
            (x2 - dx) * scale_x, (y2 - dy) * scale_y)


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
    # Margin in detection pixels, then inverse letterbox into the upload.
    width, height = x2 - x1, y2 - y1
    margin_x, margin_y = width * margin, height * margin
    ux1, uy1, ux2, uy2 = detect_to_upload(
        (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y),
        upload_width, upload_height)
    out_x1 = max(0, int(ux1))
    out_y1 = max(0, int(uy1))
    out_x2 = min(upload_width, int(ux2))
    out_y2 = min(upload_height, int(uy2))
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


def generic_packed_name(class_name: str) -> str:
    """Audience display name for a packed item with no catalog match.

    Packing never requires identification: the detector class alone counts.
    Apples need only their name; everything else reads as unidentified
    until OCR evidence establishes the product name.
    """
    return {
        "Bottle": "Unidentified bottle",
        "Canned": "Unidentified can",
        "Storage box": "Unidentified box",
        "Apple": "Apple",
    }.get(class_name, f"Unidentified {class_name.lower()}")


def ident_label(info) -> tuple:
    """Audience overlay labels anchored to the tracked box.

    Returns (main, sub): the human-readable product name + size and a
    plain qualifier ("Likely match · checking label" or "Recognized").
    Never a SKU slug, a numeric confidence, or the word "Candidate"."""
    if not isinstance(info, dict):
        return ("Reading label…", "")
    main = str(info.get("label_main") or "").strip()
    sub = str(info.get("label_sub") or "").strip()
    if main:
        return (main, sub)
    # Back-compat for snapshots without display labels: still human words.
    if info.get("choice"):
        return ("Likely match", "checking label")
    if info.get("status") == "unknown":
        return ("Need a clearer view", "")
    return ("Reading label…", "")


def pending_tracks(tracker) -> list:
    """Tracks currently in the "Packing..." watch (hidden at the boundary)."""
    pendings = []
    for track_id, state in tracker.tracks.items():
        if state.pending_deadline is None or state.packed:
            continue
        pendings.append({
            "track_id": track_id, "class_name": state.class_name,
            "bbox": [float(v) for v in state.last_bbox],
        })
    return pendings


def annotate_frame(frame, detections, tracker, identification=None,
                   bag_zone=None, bag_mode="fixed"):
    """Audience overlay: short tabs on boxes, contour on the bag, one
    bottom line for packed items and one bottom-right zone notice.

    Long product names and stacked status lines never reach the video;
    they live in the side panel. A "✓" tab suffix means the item is
    identified (not packed); packed boxes read lime.
    """
    identification = identification or {}
    tab_specs = []  # (text, box, fill, foreground)
    draw_boxes = []  # (box, halo)
    for detection in detections:
        state = tracker.tracks.get(detection.track_id)
        if state is None:
            continue
        info = identification.get(detection.track_id)
        identified = bool(isinstance(info, dict) and info.get("complete"))
        text = f"{state.class_name} #{state.track_id}"
        if identified:
            text += " ✓"
        if state.packed:
            fill, foreground, halo = LIME, CHARCOAL, LIME
        else:
            fill, foreground, halo = CHARCOAL, PAPER, PAPER
        tab_specs.append((text, tuple(detection.bbox), fill, foreground))
        draw_boxes.append((tuple(detection.bbox), halo))
    for pending in pending_tracks(tracker):
        text = f"{pending['class_name']} Packing..."
        tab_specs.append((text, tuple(pending["bbox"]), PEACH, CHARCOAL))
        draw_boxes.append((tuple(pending["bbox"]), PEACH))
    for (x1, y1, x2, y2), halo in draw_boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), halo, 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), CHARCOAL, 1)
    for (text, _box, fill, foreground), rect in zip(
            tab_specs, layout_tabs([(t, b) for t, b, _f, _g in tab_specs])):
        draw_tab(frame, text, rect, fill, foreground)

    notice = None  # (text, color) for the reserved bottom-right slot.
    x1, y1, x2, y2 = map(int, tracker.roi)
    if bag_mode == "dynamic" and bag_zone is not None:
        # Dynamic whole-bag zone: fitted contour, never the fixed rectangle.
        # Charcoal underlay keeps the lime contour legible on white tables
        # and colorful plastic alike; product boxes stay thin by contrast.
        status = bag_zone.status
        poly = bag_zone.footprint
        if poly is not None and len(poly) >= 3 and status in ("stable", "moving", "grace"):
            points = poly.reshape(-1, 1, 2).astype(int)
            if status == "stable":
                cv2.polylines(frame, [points], True, CHARCOAL, 5)
                cv2.polylines(frame, [points], True, LIME, 3)
            elif status == "grace":
                cv2.polylines(frame, [points], True, MUTED, 2)
            else:
                cv2.polylines(frame, [points], True, CHARCOAL, 5)
                cv2.polylines(frame, [points], True, PEACH, 3)
        if status == "lost":
            notice = ("Bag lost - packing paused", PEACH)
        elif status == "moving":
            notice = ("Bag moving - packing paused", PEACH)
        elif status == "grace":
            notice = ("Reacquiring bag...", MUTED)
        elif status == "locating":
            notice = ("Locating bag...", MUTED)
        elif status == "unavailable":
            notice = ("Bag unavailable - packing paused", PEACH)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), PAPER, 3)
        cv2.rectangle(frame, (x1, y1), (x2, y2), CHARCOAL, 1)

    # Reserved bottom strip: packed line bottom-left, zone notice
    # bottom-right. Fixed short strings on opposite sides never overlap.
    total = sum(tracker.packed_counts.values())
    draw_label(frame, f"Items packed: {total}", (16, 464), LIME, scale=0.6)
    if notice is not None:
        text, color = notice
        (text_w, _text_h), _baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        draw_label(frame, text, (FRAME_WIDTH - 8 - text_w - 10, 464),
                   color, scale=0.55)
    return frame


class FrameProcessor:
    """One camera/table per server process. The lock also serializes reset.

    Stage 1: detection frames (raw JPEG) advance the packing tracker and
    return OCR crop *requests*; browser-supplied crop responses are routed
    to the background OCR pipeline and never advance detection/tracking.
    """

    def __init__(self, model, identifier=None, bag_zone="auto"):
        self.model = model
        self.identifier = identifier
        self.tracker = PackingTracker()
        # Dynamic whole-bag zone. The bag localizer is a separate
        # segmentation model: the bag class never enters the product
        # detector, product tracks, OCR crops, or packing counts.
        self.bag_mode = BAG_ZONE_MODE
        self.bag_zone = None
        if bag_zone == "auto" and self.bag_mode == "dynamic":
            self.bag_zone = self._build_bag_zone()
        elif bag_zone is not None and bag_zone != "auto":
            self.bag_zone = bag_zone
        if self.bag_zone is not None:
            self.tracker.zone_test = self.bag_zone.zone_test
            self.tracker.set_packing_paused(self.bag_zone.packing_paused)
        # Recognized names for packed tracks (track_id -> display string).
        # Counts live in the tracker; this cache only renames their rows.
        self.packed_labels = {}
        self.lock = Lock()
        self.session_version = 0
        self.frame_times = deque(maxlen=30)
        self.detect_ms_ema = 0.0
        # Server-issued detection frame IDs (stage 1 protocol).
        self.frame_seq = 0
        # Outstanding browser crop requests: request_id -> record. At most
        # one outstanding request per track (see
        # MAX_OUTSTANDING_CROP_REQUESTS_PER_TRACK): a newer detection never
        # invalidates a crop that is still travelling; accumulated OCR
        # evidence in the identifier is never cleared either.
        self.crop_pending = {}
        self.crop_by_track = {}
        self.crop_counter = 0
        # Bounded tombstones for answered/expired/invalidated request ids,
        # so late redeliveries report a specific rejection reason instead
        # of a generic unknown_request.
        self.crop_tombstones = {}
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

    def _build_bag_zone(self):
        """Construct the bag zone tracker; never raises.

        On failure the zone reports ``unavailable`` (packing paused, banner
        shown) instead of silently falling back to the fixed rectangle.
        """
        from pathlib import Path

        from bag_zone import BagLocalizer, BagZoneTracker, load_bag_model
        from config import (
            BAG_ACQUIRE_STABLE, BAG_CONF_THRESHOLD,
            BAG_GRACE_PERIOD_S, BAG_REFRESH_PERIOD_S,
            BAG_IMPLAUSIBLE_CONF, BAG_IMPLAUSIBLE_FRAC,
            BAG_MAX_FOOTPRINT_FRAC, BAG_MIN_FRAC,
            BAG_MOTION_THRESHOLD, BAG_OVERLAP_THRESHOLD, BAG_RELOCK_IOU,
            BAG_RIM_MARGIN_PX,
        )
        try:
            root = Path(__file__).resolve().parent
            model = load_bag_model(root)
            localizer = BagLocalizer(model, BAG_CONF_THRESHOLD, BAG_MIN_FRAC)
        except Exception:
            logger.exception("Bag localizer failed to load; packing stays paused.")
            localizer = None
        return BagZoneTracker(
            localizer, on_relocation=self.tracker.note_zone_relocation,
            overlap_threshold=BAG_OVERLAP_THRESHOLD,
            refresh_period_s=BAG_REFRESH_PERIOD_S,
            acquire_stable=BAG_ACQUIRE_STABLE,
            relock_iou=BAG_RELOCK_IOU, motion_threshold=BAG_MOTION_THRESHOLD,
            rim_margin=BAG_RIM_MARGIN_PX,
            grace_period_s=BAG_GRACE_PERIOD_S,
            max_footprint_frac=BAG_MAX_FOOTPRINT_FRAC,
            implausible_conf=BAG_IMPLAUSIBLE_CONF,
            implausible_frac=BAG_IMPLAUSIBLE_FRAC)

    def reset(self) -> dict:
        with self.lock:
            self.tracker.reset()
            self.packed_labels = {}
            if self.bag_zone is not None:
                self.bag_zone.reset()
                self.tracker.zone_test = self.bag_zone.zone_test
                self.tracker.set_packing_paused(self.bag_zone.packing_paused)
            self.session_version += 1
            for request_id in self.crop_pending:
                self._tombstone(request_id, "session_mismatch")
            self.crop_pending = {}
            self.crop_by_track = {}
            if self.identifier is not None:
                self.identifier.reset()
            self.frame_times.clear()
            self.detect_ms_ema = 0
            trackers = getattr(getattr(self.model, "predictor", None), "trackers", [])
            direct = getattr(self.model, "tracker", None)
            for tracker in list(trackers) + ([direct] if direct is not None else []):
                tracker.reset()
            return self._snapshot()

    def _tombstone(self, request_id: str, reason: str) -> None:
        """Remember why a request id is no longer valid (bounded memory)."""
        if not request_id:
            return
        self.crop_tombstones[request_id] = reason
        while len(self.crop_tombstones) > MAX_CROP_TOMBSTONES:
            del self.crop_tombstones[next(iter(self.crop_tombstones))]

    def _retire_request(self, request_id: str, reason: str) -> None:
        record = self.crop_pending.pop(request_id, None)
        if record is not None and self.crop_by_track.get(record["track_id"]) == request_id:
            del self.crop_by_track[record["track_id"]]
        self._tombstone(request_id, reason)

    def set_connection(self, connection_id) -> None:
        with self.lock:
            self.connection_id = connection_id
            for request_id in self.crop_pending:
                self._tombstone(request_id, "connection_mismatch")
            self.crop_pending = {}
            self.crop_by_track = {}

    def clear_connection(self, connection_id) -> None:
        with self.lock:
            if self.connection_id == connection_id:
                self.connection_id = None
                for request_id in self.crop_pending:
                    self._tombstone(request_id, "connection_mismatch")
                self.crop_pending = {}
                self.crop_by_track = {}

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot()

    def _snapshot(self) -> dict:
        snapshot = {
            **self.tracker.snapshot(), "session_version": self.session_version,
            "model_profile": MODEL_PROFILE, "model_label": MODEL_LABEL,
            "inference_device": INFERENCE_DEVICE, "inference_size": INFERENCE_SIZE,
            "enabled_classes": [self.model.names[index] for index in self.food_class_ids],
            "bag_zone_mode": self.bag_mode,
        }
        if self.bag_zone is not None:
            snapshot["bag_zone"] = self.bag_zone.snapshot()
        display_counts, last_display, display_items = self._packed_display()
        snapshot["packed_display_counts"] = display_counts
        snapshot["packed_items"] = display_items
        if last_display is not None:
            snapshot["last_event"] = last_display
        return snapshot

    def _refresh_packed_labels(self, identification: dict) -> None:
        """Cache recognized names for packed tracks (display only).

        The cache upgrades a packed row from its generic name when OCR
        evidence completes and survives track pruning, so a late
        recognition renames the row instead of adding a count.
        """
        for event in self.tracker.packed_events:
            track_id = event["track_id"]
            info = identification.get(track_id)
            if (isinstance(info, dict) and info.get("complete")
                    and str(info.get("label_main") or "").strip()):
                self.packed_labels[track_id] = str(info["label_main"]).strip()

    def _packed_display(self) -> tuple:
        """Audience names for packed items, without touching their counts.

        Each packed event counts exactly once under its detector class. The
        displayed name starts generic ("Unidentified bottle") and upgrades
        to the recognized product name when OCR evidence completes — the
        cached label survives track pruning so a late recognition still
        renames the packed row instead of adding a new count.
        """
        identification = {}
        if self.identifier is not None:
            try:
                identification = self.identifier.snapshot()
            except Exception:
                logger.exception("Packed display labels unavailable.")
        self._refresh_packed_labels(identification)
        counts: dict[str, int] = {}
        items: dict[str, dict] = {}
        for event in self.tracker.packed_events:
            name = self.packed_labels.get(
                event["track_id"],
                generic_packed_name(event["class_name"]))
            counts[name] = counts.get(name, 0) + 1
            entry = items.get(name)
            if entry is None:
                items[name] = {
                    "display_name": name,
                    "count": 1,
                    "track_id": event["track_id"],
                    "has_crop": self._packed_has_crop(event["track_id"]),
                }
            else:
                entry["count"] += 1
        last = None
        if self.tracker.packed_events:
            last = self.tracker.packed_events[-1].copy()
            last["display_name"] = self.packed_labels.get(
                last["track_id"], generic_packed_name(last["class_name"]))
        return counts, last, list(items.values())

    def _packed_has_crop(self, track_id: int) -> bool:
        """Whether a packed track still has an OCR crop preview available."""
        identifier = self.identifier
        if identifier is None:
            return False
        try:
            with identifier.lock:
                evidence = identifier.tracks.get(track_id)
                return bool(evidence is not None
                            and getattr(evidence, "last_crop_jpeg", b""))
        except Exception:
            return False

    def _expire_crops(self, now: float) -> None:
        for request_id, record in list(self.crop_pending.items()):
            if now - record["issued_at"] > CROP_REQUEST_TTL_S:
                self._retire_request(request_id, "expired")

    def _issue_crop_requests(self, detections, upload_width, upload_height,
                             frame_id, connection_id) -> list:
        """Eligible tracks for this frame, bounded per frame and globally.

        A track with a valid outstanding request gets no duplicate: the
        earlier request stays valid until it is answered or expires, so a
        crop still travelling from the browser is never invalidated by a
        newer detection for the same track.
        """
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
            with identifier.lock:
                evidence_ref = identifier.tracks.get(detection.track_id)
            # Keep a valid outstanding request travelling instead of
            # replacing it: the browser may still be encoding its crop, and
            # invalidating it produced unknown_request rejections.
            old_id = self.crop_by_track.get(detection.track_id)
            old = self.crop_pending.get(old_id) if old_id else None
            if old is not None:
                if old["evidence_ref"] is evidence_ref:
                    continue
                # The track's evidence object changed (pruned, reset or the
                # numeric ID was reused), so the old crop could never be
                # validly answered; retire it and allow a fresh request.
                self._retire_request(old_id, "track_expired")
            self.crop_counter += 1
            request_id = f"{frame_id}:{detection.track_id}:{self.crop_counter}"
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
        # The resize is an aspect-preserving letterbox (black bars for 16:9
        # uploads, identity for 4:3), so proportions stay truthful while all
        # geometry shares the 640x480 detection space.
        frame = letterbox_frame(full_frame)

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
            bag_zone_state = None
            bag_ms = 0.0
            suppressed_bag_self = 0
            if self.bag_zone is not None:
                bag_start = time.monotonic()
                bag_zone_state = self.bag_zone.update(frame, self.frame_seq + 1)
                bag_ms = (time.monotonic() - bag_start) * 1000
                self.tracker.zone_test = self.bag_zone.zone_test
                self.tracker.zone_version = self.bag_zone.zone_test.version
                self.tracker.set_packing_paused(self.bag_zone.packing_paused)
                # The bag itself can score as packaging (live: Storage box /
                # Canned boxes covering the footprint). Drop bag-sized boxes
                # before tracking so the bag never becomes a product track,
                # gets OCR crops, or counts as packed. Genuine items beside
                # or inside the bag cover far less of the footprint.
                if (self.bag_zone.status in ("stable", "grace")
                        and self.bag_zone.footprint is not None):
                    from bag_zone import is_bag_self
                    from config import BAG_SELF_BOX_FRAC, BAG_SELF_FOOT_FRAC
                    grid = self.bag_zone.grid
                    kept = []
                    for detection in detections:
                        if is_bag_self(detection.bbox, grid,
                                       BAG_SELF_FOOT_FRAC, BAG_SELF_BOX_FRAC):
                            suppressed_bag_self += 1
                        else:
                            kept.append(detection)
                    detections = kept
            events = self.tracker.update(detections, now=now)
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
            self._refresh_packed_labels(identification)
            frame = annotate_frame(frame, detections, self.tracker, identification,
                                   self.bag_zone, self.bag_mode)
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
                "tracks": [{"track_id": d.track_id, "bbox": list(map(float, d.bbox))} for d in detections],
                "events": events,  # Only newly registered events, in frame order.
                "pending": pending_tracks(self.tracker),
                # Diagnostics (stage 2): network-exclusive backend timings.
                # detect_ms is pure server processing for this frame;
                # upload_bytes is the received detection JPEG size.
                "detect_ms": round(detect_ms, 1),
                "bag_ms": round(bag_ms, 1),
                "bag_zone": bag_zone_state,
                "suppressed_bag_self": suppressed_bag_self,
                "upload_bytes": len(image_bytes),
            }

    def process_crop_response(self, header: dict, jpeg_bytes: bytes,
                              connection_id=None) -> dict:
        """Validate a browser crop and queue it for background OCR.

        Never advances the packing tracker or detection timing. Returns a
        crop_ack payload ({type, request_id, ok, reason}).
        """
        def ack(request_id, ok, reason):
            return {"type": "crop_ack", "request_id": request_id or "",
                    "ok": bool(ok), "reason": reason,
                    "session_version": self.session_version,
                    "request_to_receipt_ms": round((now-record["issued_at"])*1000, 2) if record else None}

        record = None
        now = time.monotonic()
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
                # Bounded tombstones give late duplicates, expired or
                # invalidated requests a specific reason; anything else is
                # genuinely unknown (stale beyond the tombstone window).
                tombstone = self.crop_tombstones.get(request_id)
                return ack(request_id, False, tombstone or "unknown_request")
            if record["frame_id"] != frame_id or record["track_id"] != track_id:
                return ack(request_id, False, "mismatched")
            if record["session_version"] != self.session_version:
                self._retire_request(request_id, "session_mismatch")
                return ack(request_id, False, "session_mismatch")
            if session_version != self.session_version:
                self._retire_request(request_id, "session_mismatch")
                return ack(request_id, False, "session_mismatch")
            if (record["connection_id"] is not None and connection_id is not None
                    and record["connection_id"] != connection_id):
                return ack(request_id, False, "connection_mismatch")
            # Consume immediately: duplicate deliveries find no pending
            # entry and are answered from the tombstone instead.
            self.crop_pending.pop(request_id, None)
            if self.crop_by_track.get(track_id) == request_id:
                del self.crop_by_track[track_id]
            self._tombstone(request_id, "duplicate")
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
        sharpness_norm = crop_sharpness_at_height(crop, OCR_SHARPNESS_NORM_HEIGHT)
        try:
            identifier.note(track_id, hint, sharpness, (crop_width, crop_height),
                            **({"expected": evidence_ref} if hasattr(identifier, "replay_debug") else {}))
        except Exception:
            logger.exception("Identification note failed for crop %s.", request_id)
            return ack(request_id, False, "internal")
        if not quality_gate_accepts(sharpness, sharpness_norm):
            return ack(request_id, False, "blurry")
        with self.lock:
            if session_version != self.session_version:
                return ack(request_id, False, "session_mismatch")
            kwargs = {}
            if hasattr(identifier, "replay_debug"):
                kwargs = {"expected": evidence_ref, "diagnostic":
                          {"request_id": request_id, "frame_id": frame_id, "track_id": track_id,
                           "session_version": session_version, "ocr_status": "queued",
                           "sharpness_full": round(sharpness, 2),
                           "sharpness_norm": round(sharpness_norm, 2)}
                          if header.get("diagnostics") is True else None}
            submitted = identifier.submit(track_id, hint, bytes(jpeg_bytes),
                                          (crop_width, crop_height),
                                          frame_id=frame_id, **kwargs)
        if submitted == "stale":
            # A newer crop for this track is already pending OCR; this
            # deliberate stale drop is distinct from an unknown request.
            return ack(request_id, False, "stale")
        if not submitted:
            if identifier.is_complete(track_id):
                return ack(request_id, False, "complete")
            if identifier.ocr_gave_up:
                return ack(request_id, False, "ocr_unavailable")
            return ack(request_id, False, "throttled")
        return ack(request_id, True, "accepted")
