"""YOLO adapter, drawing and serialized access to one packing session."""

import base64
import logging
import time
from collections import Counter, deque
from threading import Lock

import cv2
import numpy as np

from config import (
    AGNOSTIC_NMS, CONF_THRESHOLD, FOOD_CLASSES, FRAME_HEIGHT, FRAME_WIDTH,
    MAX_FRAME_BYTES, MAX_PACKED_OVERLAY_ROWS, NMS_IOU_THRESHOLD,
    MODEL_LABEL, MODEL_PROFILE, PACKED_BANNER_FRAMES, TRACKER_CONFIG,
    INFERENCE_DEVICE, INFERENCE_SIZE, OCR_CROP_JPEG_QUALITY,
    OCR_CROP_MARGIN, OCR_MIN_CROP_HEIGHT, OCR_MIN_CROP_WIDTH,
    OCR_SHARPNESS_MIN,
)
from tracking import Detection, PackingTracker


logger = logging.getLogger(__name__)


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


def hires_crop(full_frame, bbox_640x480):
    """Map a detection box to the high-resolution frame with a small margin."""
    full_height, full_width = full_frame.shape[:2]
    scale_x, scale_y = full_width / FRAME_WIDTH, full_height / FRAME_HEIGHT
    x1, y1, x2, y2 = bbox_640x480
    width, height = x2 - x1, y2 - y1
    margin_x, margin_y = width * OCR_CROP_MARGIN, height * OCR_CROP_MARGIN
    x1 = max(0, int((x1 - margin_x) * scale_x))
    y1 = max(0, int((y1 - margin_y) * scale_y))
    x2 = min(full_width, int((x2 + margin_x) * scale_x))
    y2 = min(full_height, int((y2 + margin_y) * scale_y))
    if x2 - x1 < OCR_MIN_CROP_WIDTH or y2 - y1 < OCR_MIN_CROP_HEIGHT:
        return None
    return full_frame[y1:y2, x1:x2]


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
    """One camera/table per server process. The lock also serializes reset."""

    def __init__(self, model, identifier=None):
        self.model = model
        self.identifier = identifier
        self.tracker = PackingTracker()
        self.lock = Lock()
        self.session_version = 0
        self.frame_times = deque(maxlen=30)
        self.detect_ms_ema = 0.0
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
            if self.identifier is not None:
                self.identifier.reset()
            # Keep ByteTrack IDs continuous; the packing state starts fresh.
            return self._snapshot()

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

    def process(self, image_bytes: bytes) -> dict:
        if not image_bytes or len(image_bytes) > MAX_FRAME_BYTES:
            raise ValueError("Camera frame is empty or too large.")
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        full_frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if full_frame is None:
            raise ValueError("Could not decode camera frame.")
        # Detection keeps the small fixed input; OCR crops use the upload.
        frame = cv2.resize(full_frame, (FRAME_WIDTH, FRAME_HEIGHT))

        with self.lock:
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
            identification = self._identify(detections, full_frame)
            if self.identifier is not None:
                self.identifier.note_capture((full_width, full_height), fps,
                                             self.detect_ms_ema)
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
                "image": base64.b64encode(image.tobytes()).decode("ascii"),
                "counts": dict(visible_counts),  # Original frontend compatibility.
                "visible_counts": dict(visible_counts),
                **self._snapshot(),
                "identification": identification,
                "events": events,  # Only newly registered events, in frame order.
            }

    def _identify(self, detections, full_frame) -> dict:
        """Queue sharp hires crops per track; never block detection."""
        identifier = self.identifier
        if identifier is None:
            return {}
        try:
            active_ids = set()
            for detection in detections:
                active_ids.add(detection.track_id)
                if identifier.is_complete(detection.track_id):
                    continue
                crop = hires_crop(full_frame, detection.bbox)
                if crop is None:
                    continue
                crop_height, crop_width = crop.shape[:2]
                sharpness = crop_sharpness(crop)
                # Record every seen crop so the overlay/sidebar can show
                # collection state; only sharp crops are queued for OCR.
                identifier.note(detection.track_id, detection.class_name, sharpness,
                                (crop_width, crop_height))
                if sharpness < OCR_SHARPNESS_MIN:
                    continue
                success, encoded = cv2.imencode(
                    ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), OCR_CROP_JPEG_QUALITY],
                )
                if success:
                    # The detector label is an unreliable hint, not a filter:
                    # every tracked product crop may carry label evidence.
                    identifier.submit(detection.track_id, detection.class_name,
                                      encoded.tobytes(), (crop_width, crop_height))
            identifier.prune(active_ids)
            return identifier.snapshot()
        except Exception:
            logger.exception("Identification update failed; detection continues.")
            return identifier.snapshot() if identifier is not None else {}
