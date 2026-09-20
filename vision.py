"""YOLO adapter, drawing and serialized access to one packing session."""

import base64
from collections import Counter
from threading import Lock

import cv2
import numpy as np

from config import (
    AGNOSTIC_NMS, CONF_THRESHOLD, FOOD_CLASSES, FRAME_HEIGHT, FRAME_WIDTH,
    MAX_FRAME_BYTES, MAX_PACKED_OVERLAY_ROWS, NMS_IOU_THRESHOLD,
    MODEL_LABEL, MODEL_PROFILE, PACKED_BANNER_FRAMES, TRACKER_CONFIG,
    INFERENCE_DEVICE, INFERENCE_SIZE,
)
from tracking import Detection, PackingTracker


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


def draw_label(frame, text, origin, color=(255, 255, 255), scale=0.5):
    # A dark outline keeps small labels legible on light and dark products.
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (15, 20, 25), 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def annotate_frame(frame, detections, tracker):
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
        draw_label(frame, label, (max(2, x1), max(18, y1 - 8)), color)

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

    def __init__(self, model):
        self.model = model
        self.tracker = PackingTracker()
        self.lock = Lock()
        self.session_version = 0
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
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode camera frame.")
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        with self.lock:
            result = self.model.track(
                frame, persist=True, tracker=TRACKER_CONFIG,
                classes=self.food_class_ids, conf=CONF_THRESHOLD,
                imgsz=INFERENCE_SIZE, device=INFERENCE_DEVICE, verbose=False,
                agnostic_nms=AGNOSTIC_NMS, iou=NMS_IOU_THRESHOLD,
                # Select the NMS head on YOLO26 so the same duplicate-box filter
                # used by the YOLO11 profiles remains active before ByteTrack.
                nms=True,
            )[0]
            detections = extract_food_detections(result)
            events = self.tracker.update(detections)
            visible_counts = Counter(
                self.tracker.tracks[d.track_id].class_name for d in detections
                if d.track_id in self.tracker.tracks
            )
            frame = annotate_frame(frame, detections, self.tracker)
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
                "events": events,  # Only newly registered events, in frame order.
            }
