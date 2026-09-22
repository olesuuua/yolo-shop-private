"""Stage 1: detection frames vs browser-supplied OCR crops (no weights)."""

import time
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from config import (
    FOOD_CLASSES, MAX_CROP_REQUESTS_PER_FRAME, MAX_CROP_TOMBSTONES,
    MAX_OUTSTANDING_CROP_REQUESTS_PER_TRACK, MAX_PENDING_CROP_REQUESTS,
)
from identification import IdentificationService, IdentConfig
from vision import (
    CROP_COORD_SPACE, FrameProcessor, build_crop_envelope,
    parse_client_message, upload_crop_rect,
)


class Array:
    def __init__(self, values):
        self.values = values

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    names = {index: name for index, name in enumerate(sorted(FOOD_CLASSES))}

    def __init__(self):
        self.frames = []
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        rows = self.frames.pop(0) if self.frames else []
        boxes = SimpleNamespace(
            xyxy=Array([r[0] for r in rows]),
            cls=Array([r[1] for r in rows]),
            id=Array([r[2] for r in rows]) if rows else None,
        )
        return [SimpleNamespace(boxes=boxes, names=self.names)]


CLASS_IDS = {name: class_id for class_id, name in FakeModel.names.items()}
PRIMARY = sorted(FOOD_CLASSES)[0]
BOX = (100, 100, 300, 300)  # Comfortably large in 640x480 space.


def jpeg_bytes(width=640, height=480, value=0):
    return cv2.imencode(
        ".jpg", np.full((height, width, 3), value, dtype=np.uint8))[1].tobytes()


def sharp_crop_jpeg(width=200, height=200):
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(frame, "SENEZHSKAYA WATER 0.5L", (5, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    for y in range(0, height, 10):
        cv2.line(frame, (0, y), (width, y), (0, 0, 0), 1)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def processor_with(rows_per_frame, **kwargs):
    model = FakeModel()
    model.frames = list(rows_per_frame)
    service = IdentificationService(jev_fn=lambda *a: {"answer": {}, "status": "x"})
    proc = FrameProcessor(model, identifier=service, **kwargs) if kwargs else \
        FrameProcessor(model, identifier=service)
    return proc, model, service


def detect(proc, model, rows, connection="conn-1"):
    model.frames.append(rows)
    return proc.process_detect(jpeg_bytes(), connection_id=connection)


def issue_request(proc, model, track=7, connection="conn-1"):
    response = detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], track)],
                      connection=connection)
    assert len(response["crop_requests"]) == 1, response["crop_requests"]
    return response["crop_requests"][0], response["frame_id"]


class RoutingTests(unittest.TestCase):
    def test_raw_jpeg_routes_to_detection(self):
        kind, header, payload = parse_client_message(jpeg_bytes())
        self.assertEqual(kind, "detection")
        self.assertEqual(header, {})
        self.assertTrue(payload.startswith(b"\xff\xd8"))

    def test_crop_envelope_round_trips(self):
        header = {"request_id": "1:2:3", "frame_id": 1, "track_id": 2,
                  "session_version": 0}
        blob = build_crop_envelope(header, b"\xff\xd8fake")
        kind, parsed, payload = parse_client_message(blob)
        self.assertEqual(kind, "crop")
        self.assertEqual(parsed, header)
        self.assertEqual(payload, b"\xff\xd8fake")

    def test_malformed_envelope_rejected(self):
        with self.assertRaises(ValueError):
            parse_client_message(b"YOLO\x01")
        with self.assertRaises(ValueError):
            parse_client_message(b"YOLO\x01\x09" + b"\x00" * 20)
        with self.assertRaises(ValueError):
            parse_client_message(build_crop_envelope({"request_id": 1}, b"x")[:12])


class MappingTests(unittest.TestCase):
    def test_scales_with_upload_resolution(self):
        rect_small = upload_crop_rect(BOX, 640, 480)
        rect_large = upload_crop_rect(BOX, 1280, 960)
        self.assertIsNotNone(rect_small)
        self.assertIsNotNone(rect_large)
        # 2x upload -> ~2x crop origin and size (margin included).
        self.assertAlmostEqual(rect_large[0], rect_small[0] * 2, delta=2)
        self.assertAlmostEqual(rect_large[2] - rect_large[0],
                               (rect_small[2] - rect_small[0]) * 2, delta=2)

    def test_widescreen_upload_scales_axes_independently(self):
        rect = upload_crop_rect(BOX, 1280, 720)
        self.assertIsNotNone(rect)
        x1, y1, x2, y2 = rect
        self.assertTrue(0 <= x1 < x2 <= 1280)
        self.assertTrue(0 <= y1 < y2 <= 720)

    def test_clamps_at_frame_edges(self):
        rect = upload_crop_rect((0, 0, 640, 480), 640, 480)
        self.assertEqual(rect, (0, 0, 640, 480))
        corner = upload_crop_rect((520, 360, 639, 479), 640, 480)
        self.assertIsNotNone(corner)
        self.assertLessEqual(corner[2], 640)
        self.assertLessEqual(corner[3], 480)

    def test_too_small_and_invalid_rejected(self):
        self.assertIsNone(upload_crop_rect((10, 10, 20, 20), 640, 480))
        self.assertIsNone(upload_crop_rect((300, 300, 100, 100), 640, 480))
        self.assertIsNone(upload_crop_rect(("a", 1, 2, 3), 640, 480))
        self.assertIsNone(upload_crop_rect(BOX, 0, 480))


class DetectionRequestTests(unittest.TestCase):
    def test_detection_returns_typed_requests_with_required_fields(self):
        proc, model, _ = processor_with([])
        response = detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        self.assertEqual(response["type"], "detection")
        self.assertEqual(response["frame_id"], 1)
        self.assertEqual(response["session_version"], 0)
        self.assertIn("image", response)
        self.assertEqual(len(response["crop_requests"]), 1)
        request = response["crop_requests"][0]
        self.assertEqual(request["frame_id"], 1)
        self.assertEqual(request["track_id"], 7)
        self.assertEqual(request["session_version"], 0)
        self.assertTrue(request["request_id"])
        self.assertEqual(request["coord_space"], CROP_COORD_SPACE)
        self.assertEqual(request["bbox"], [float(v) for v in BOX])
        self.assertEqual((request["upload_width"], request["upload_height"]), (640, 480))
        second = detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        self.assertEqual(second["frame_id"], 2)

    def test_complete_and_throttled_tracks_get_no_request(self):
        proc, model, service = processor_with([])
        detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        service.tracks[7].complete = True
        response = detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        self.assertEqual(response["crop_requests"], [])

    def test_per_frame_and_global_bounds(self):
        proc, model, _ = processor_with([])
        rows = [(BOX, CLASS_IDS[PRIMARY], track) for track in range(1, 20)]
        response = detect(proc, model, rows)
        self.assertLessEqual(len(response["crop_requests"]),
                             MAX_CROP_REQUESTS_PER_FRAME)
        # Fill globally across frames with distinct tracks (throttle is
        # per-track, so new tracks stay eligible).
        proc2, model2, _ = processor_with([])
        track = 100
        for _ in range(10):
            rows = [(BOX, CLASS_IDS[PRIMARY], track + i) for i in range(4)]
            detect(proc2, model2, rows)
            track += 4
        self.assertLessEqual(len(proc2.crop_pending), MAX_PENDING_CROP_REQUESTS)

    def test_outstanding_request_survives_newer_detections(self):
        proc, model, service = processor_with([])
        first = detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        request = first["crop_requests"][0]
        service._merge(7, PRIMARY, [{"text": "AQUA", "score": 0.9}])
        # Force re-eligibility: pretend the throttle interval has passed.
        service.tracks[7].last_submit_time -= 1000
        service.tracks[7].last_ocr_hash = None
        # Several newer detections must not invalidate the travelling crop
        # or duplicate the outstanding request for this track.
        for _ in range(5):
            detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        self.assertEqual(len(proc.crop_pending), 1)
        self.assertIn(request["request_id"], proc.crop_pending)
        header = {"request_id": request["request_id"], "frame_id": request["frame_id"],
                  "track_id": request["track_id"],
                  "session_version": request["session_version"]}
        ack = proc.process_crop_response(header, sharp_crop_jpeg(),
                                         connection_id="conn-1")
        self.assertTrue(ack["ok"], ack)
        self.assertEqual(ack["reason"], "accepted")
        with service.slot_lock:
            self.assertIn(request["track_id"], service.slots)
        self.assertIn("AQUA", service.tracks[7].lines)


class CropResponseTests(unittest.TestCase):
    def request(self, proc, model, track=7, connection="conn-1"):
        return issue_request(proc, model, track=track, connection=connection)

    def test_accepted_crop_reaches_ocr_without_advancing_tracker(self):
        proc, model, service = processor_with([])
        request, _ = self.request(proc, model)
        frames_before = proc.tracker.frame_number
        packed_before = dict(proc.tracker.packed_counts)
        ack = proc.process_crop_response(
            {"request_id": request["request_id"], "frame_id": request["frame_id"],
             "track_id": request["track_id"],
             "session_version": request["session_version"]},
            sharp_crop_jpeg(), connection_id="conn-1")
        self.assertEqual(ack["type"], "crop_ack")
        self.assertTrue(ack["ok"], ack)
        self.assertEqual(ack["reason"], "accepted")
        self.assertEqual(proc.tracker.frame_number, frames_before)
        self.assertEqual(dict(proc.tracker.packed_counts), packed_before)
        self.assertEqual(len(proc.frame_times), 1)  # No detection timing touched.
        with service.slot_lock:
            self.assertIn(request["track_id"], service.slots)

    def test_duplicate_delivery_rejected(self):
        proc, model, _ = processor_with([])
        request, _ = self.request(proc, model)
        header = {"request_id": request["request_id"], "frame_id": request["frame_id"],
                  "track_id": request["track_id"],
                  "session_version": request["session_version"]}
        first = proc.process_crop_response(header, sharp_crop_jpeg(),
                                           connection_id="conn-1")
        self.assertTrue(first["ok"])
        second = proc.process_crop_response(header, sharp_crop_jpeg(),
                                            connection_id="conn-1")
        self.assertFalse(second["ok"])
        self.assertEqual(second["reason"], "duplicate")

    def test_mismatched_frame_track_and_bad_header_rejected(self):
        proc, model, _ = processor_with([])
        request, _ = self.request(proc, model)
        base = {"request_id": request["request_id"], "frame_id": request["frame_id"],
                "track_id": request["track_id"],
                "session_version": request["session_version"]}
        for mutate in ({"frame_id": 999}, {"track_id": 999}, {"session_version": 999}):
            header = dict(base)
            header.update(mutate)
            ack = proc.process_crop_response(header, sharp_crop_jpeg(),
                                             connection_id="conn-1")
            self.assertFalse(ack["ok"], mutate)
        # Pending entry is consumed by the first terminal validation; a
        # session mismatch consumes it as well (re-submit needs a request).
        ack = proc.process_crop_response({"request_id": "", "frame_id": 1,
                                          "track_id": 7, "session_version": 0},
                                         sharp_crop_jpeg(), connection_id="conn-1")
        self.assertFalse(ack["ok"])

    def test_oversized_undecodable_small_and_blurry_rejected(self):
        proc, model, _ = processor_with([])
        request, _ = self.request(proc, model)
        header = {"request_id": request["request_id"], "frame_id": request["frame_id"],
                  "track_id": request["track_id"],
                  "session_version": request["session_version"]}
        self.assertEqual(
            proc.process_crop_response(header, b"", connection_id="conn-1")["reason"],
            "empty")
        request2, _ = self.request(proc, model, track=8)
        header2 = {"request_id": request2["request_id"], "frame_id": request2["frame_id"],
                   "track_id": request2["track_id"],
                   "session_version": request2["session_version"]}
        self.assertFalse(proc.process_crop_response(
            header2, b"x" * (3_000_000), connection_id="conn-1")["ok"])
        request3, _ = self.request(proc, model, track=9)
        header3 = {"request_id": request3["request_id"], "frame_id": request3["frame_id"],
                   "track_id": request3["track_id"],
                   "session_version": request3["session_version"]}
        self.assertEqual(proc.process_crop_response(
            header3, b"not a jpeg", connection_id="conn-1")["reason"], "undecodable")
        tiny = cv2.imencode(".jpg", np.zeros((10, 10, 3), dtype=np.uint8))[1].tobytes()
        request4, _ = self.request(proc, model, track=10)
        header4 = {"request_id": request4["request_id"], "frame_id": request4["frame_id"],
                   "track_id": request4["track_id"],
                   "session_version": request4["session_version"]}
        self.assertEqual(proc.process_crop_response(
            header4, tiny, connection_id="conn-1")["reason"], "too_small")
        flat = cv2.imencode(
            ".jpg", np.full((200, 200, 3), 128, dtype=np.uint8))[1].tobytes()
        request5, _ = self.request(proc, model, track=11)
        header5 = {"request_id": request5["request_id"], "frame_id": request5["frame_id"],
                   "track_id": request5["track_id"],
                   "session_version": request5["session_version"]}
        self.assertEqual(proc.process_crop_response(
            header5, flat, connection_id="conn-1")["reason"], "blurry")

    def test_reset_disconnect_and_expiry_reject(self):
        proc, model, _ = processor_with([])
        request, _ = self.request(proc, model)
        header = {"request_id": request["request_id"], "frame_id": request["frame_id"],
                  "track_id": request["track_id"],
                  "session_version": request["session_version"]}
        proc.reset()
        ack = proc.process_crop_response(header, sharp_crop_jpeg(),
                                         connection_id="conn-1")
        self.assertFalse(ack["ok"])
        self.assertEqual(ack["reason"], "session_mismatch")

        proc2, model2, _ = processor_with([])
        request2, _ = self.request(proc2, model2, connection="conn-A")
        header2 = {"request_id": request2["request_id"],
                   "frame_id": request2["frame_id"],
                   "track_id": request2["track_id"],
                   "session_version": request2["session_version"]}
        proc2.clear_connection("conn-A")
        ack2 = proc2.process_crop_response(header2, sharp_crop_jpeg(),
                                           connection_id="conn-A")
        self.assertFalse(ack2["ok"])
        self.assertEqual(ack2["reason"], "connection_mismatch")

        proc3, model3, service3 = processor_with([])
        request3, _ = self.request(proc3, model3, track=21)
        header3 = {"request_id": request3["request_id"],
                   "frame_id": request3["frame_id"],
                   "track_id": request3["track_id"],
                   "session_version": request3["session_version"]}
        # Expire by aging the pending entry past the TTL.
        for record in proc3.crop_pending.values():
            record["issued_at"] -= 1000
        ack3 = proc3.process_crop_response(header3, sharp_crop_jpeg(),
                                           connection_id="conn-1")
        self.assertFalse(ack3["ok"])
        self.assertEqual(ack3["reason"], "expired")

    def test_track_expiry_cannot_attach_to_reused_id(self):
        proc, model, service = processor_with([])
        request, _ = issue_request(proc, model, track=7)
        header = {"request_id": request["request_id"], "frame_id": request["frame_id"],
                  "track_id": request["track_id"],
                  "session_version": request["session_version"]}
        old_ref = proc.crop_pending[request["request_id"]]["evidence_ref"]
        self.assertIsNotNone(old_ref)
        # New bottle reuses ID 7 after the old evidence expired.
        from config import IDENT_ABSENT_FRAMES
        for _ in range(IDENT_ABSENT_FRAMES + 2):
            service.prune(set())
        service.touch(7, PRIMARY)
        ack = proc.process_crop_response(header, sharp_crop_jpeg(),
                                         connection_id="conn-1")
        self.assertFalse(ack["ok"])
        self.assertEqual(ack["reason"], "track_expired")
        self.assertNotIn("SENEZHSKAYA", str(service.tracks[7].lines))

    def test_connection_mismatch_rejected(self):
        proc, model, _ = processor_with([])
        request, _ = self.request(proc, model, connection="conn-A")
        header = {"request_id": request["request_id"], "frame_id": request["frame_id"],
                  "track_id": request["track_id"],
                  "session_version": request["session_version"]}
        ack = proc.process_crop_response(header, sharp_crop_jpeg(),
                                         connection_id="conn-B")
        self.assertFalse(ack["ok"])
        self.assertEqual(ack["reason"], "connection_mismatch")


class InFlightRequestTests(unittest.TestCase):
    """A newer detection must not invalidate a travelling crop request."""

    def test_expired_request_does_not_starve_the_next_request(self):
        proc, model, service = processor_with([])
        first = detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        first_id = first["crop_requests"][0]["request_id"]
        # Age past the TTL and past the submit throttle, then detect again.
        for record in proc.crop_pending.values():
            record["issued_at"] -= 1000
        service.tracks[7].last_submit_time -= 1000
        second = detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        self.assertEqual(len(second["crop_requests"]), 1)
        self.assertNotEqual(second["crop_requests"][0]["request_id"], first_id)
        self.assertEqual(proc.crop_tombstones.get(first_id), "expired")
        self.assertEqual(len(proc.crop_pending), 1)

    def test_invalidated_track_request_is_retired_and_replaced(self):
        proc, model, service = processor_with([])
        request, _ = issue_request(proc, model, track=7)
        old_ref = proc.crop_pending[request["request_id"]]["evidence_ref"]
        from config import IDENT_ABSENT_FRAMES
        for _ in range(IDENT_ABSENT_FRAMES + 2):
            service.prune(set())
        service.touch(7, PRIMARY)
        response = detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        self.assertEqual(len(response["crop_requests"]), 1)
        self.assertNotEqual(response["crop_requests"][0]["request_id"],
                            request["request_id"])
        self.assertEqual(proc.crop_tombstones.get(request["request_id"]),
                         "track_expired")
        # The old crop can never attach to the reused numeric ID.
        header = {"request_id": request["request_id"], "frame_id": request["frame_id"],
                  "track_id": request["track_id"],
                  "session_version": request["session_version"]}
        ack = proc.process_crop_response(header, sharp_crop_jpeg(),
                                         connection_id="conn-1")
        self.assertFalse(ack["ok"])
        self.assertEqual(ack["reason"], "track_expired")

    def test_out_of_order_crop_cannot_replace_newer_pending_work(self):
        service = IdentificationService(
            config=IdentConfig(submit_interval_empty_s=0))
        self.assertTrue(service.submit(7, "bottle", b"newer", frame_id=10))
        self.assertEqual(service.submit(7, "bottle", b"older", frame_id=2),
                         "stale")
        self.assertEqual(service._take_slot(), (7, b"newer"))
        # The stale drop leaves the track able to submit newer work.
        self.assertTrue(service.submit(7, "bottle", b"newest", frame_id=11))
        self.assertEqual(service._take_slot(), (7, b"newest"))

    def test_per_track_outstanding_limit_is_explicit(self):
        proc, model, _ = processor_with([])
        detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        self.assertLessEqual(len(proc.crop_by_track),
                             MAX_OUTSTANDING_CROP_REQUESTS_PER_TRACK)
        self.assertEqual(len(proc.crop_by_track), 1)

    def test_tombstones_are_bounded(self):
        proc, model, service = processor_with([])
        for index in range(150):
            response = detect(proc, model, [])
            proc._tombstone(f"unknown-{index}", "duplicate")
        self.assertLessEqual(len(proc.crop_tombstones), MAX_CROP_TOMBSTONES)

    def test_detection_continues_while_crop_pending(self):
        proc, model, _ = processor_with([])
        first = detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        self.assertEqual(len(first["crop_requests"]), 1)
        # The pending request does not gate detection frames.
        second = detect(proc, model, [(BOX, CLASS_IDS[PRIMARY], 7)])
        self.assertEqual(second["frame_id"], first["frame_id"] + 1)
        self.assertEqual(second["crop_requests"], [])


class Stage2Tests(unittest.TestCase):
    """Small detection upload keeps diagnostics and OCR eligibility."""

    def test_detection_response_carries_backend_timings(self):
        proc, model, _ = processor_with([])
        payload = jpeg_bytes()
        model.frames.append([(BOX, CLASS_IDS[PRIMARY], 7)])
        response = proc.process_detect(payload, connection_id="conn-1")
        self.assertEqual(response["upload_bytes"], len(payload))
        self.assertIsInstance(response["detect_ms"], float)
        self.assertGreaterEqual(response["detect_ms"], 0.0)
        # 640x480 uploads still yield requests tagged with upload dims.
        self.assertEqual(len(response["crop_requests"]), 1)
        request = response["crop_requests"][0]
        self.assertEqual((request["upload_width"], request["upload_height"]), (640, 480))
        self.assertEqual(request["coord_space"], CROP_COORD_SPACE)

    def test_rect_scales_from_detect_space_to_full_resolution(self):
        # Same 640x480 box maps larger (never smaller) on bigger originals.
        base = upload_crop_rect(BOX, 640, 480)
        full = upload_crop_rect(BOX, 1280, 960)
        wide = upload_crop_rect(BOX, 1280, 720)
        self.assertIsNotNone(base)
        self.assertIsNotNone(full)
        self.assertIsNotNone(wide)
        self.assertGreater(full[2] - full[0], base[2] - base[0])
        self.assertLessEqual(wide[3], 720)


class WebSocketTests(unittest.TestCase):
    def test_crop_ack_flows_without_advancing_detection(self):
        import app as application
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        model = FakeModel()
        with patch.object(application, "load_model", return_value=model):
            client = TestClient(application.app)
            with client:
                with client.websocket_connect("/ws/detect") as ws:
                    model.frames.append([(BOX, CLASS_IDS[PRIMARY], 7)])
                    ws.send_bytes(jpeg_bytes())
                    response = ws.receive_json()
                    self.assertEqual(response["type"], "detection")
                    self.assertEqual(len(response["crop_requests"]), 1)
                    request = response["crop_requests"][0]
                    frames = application.app.state.processor.tracker.frame_number
                    header = {"request_id": request["request_id"],
                              "frame_id": request["frame_id"],
                              "track_id": request["track_id"],
                              "session_version": request["session_version"]}
                    ws.send_bytes(build_crop_envelope(header, sharp_crop_jpeg()))
                    ack = ws.receive_json()
                    self.assertEqual(ack["type"], "crop_ack")
                    self.assertTrue(ack["ok"], ack)
                    self.assertEqual(
                        application.app.state.processor.tracker.frame_number, frames)
                    # Detection pipeline still works after a crop round-trip.
                    model.frames.append([(BOX, CLASS_IDS[PRIMARY], 7)])
                    ws.send_bytes(jpeg_bytes())
                    second = ws.receive_json()
                    self.assertEqual(second["type"], "detection")
                    self.assertEqual(second["frame_id"], response["frame_id"] + 1)

    def test_text_frame_rejected_without_killing_stream(self):
        import app as application
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        model = FakeModel()
        with patch.object(application, "load_model", return_value=model):
            client = TestClient(application.app)
            with client:
                with client.websocket_connect("/ws/detect") as ws:
                    ws.send_text('{"type":"ping"}')
                    self.assertIn("error", ws.receive_json())
                    model.frames.append([])
                    ws.send_bytes(jpeg_bytes())
                    self.assertEqual(ws.receive_json()["type"], "detection")


if __name__ == "__main__":
    unittest.main()
