"""Verify PP-YOLOE preprocessing, class IDs, suppression and tracking on CPU."""

import json
from pathlib import Path
import socket
import struct
import unittest
from unittest.mock import Mock, patch

import numpy as np

import app
from config import TRACKER_CONFIG
from ppyoloe_model import PPYOLOEModel, PaddleWorker, load_ppyoloe
from ppyoloe_postprocess import preprocess, select_candidates, postprocess
from ppyoloe_protocol import receive, send, MAX_MESSAGE_BYTES
from ppyoloe_assets import CHECKPOINT_SHA256

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "objects365_classes.json").read_text())
LABELS = CATALOG["labels"]
SELECTED = {name for group in CATALOG["grocery_groups"].values() for name in group}
CLASS_IDS = [i for i, name in enumerate(LABELS) if name in SELECTED]


def predictions(rows):
    """Rows contain a square-input box and a mapping of competing label scores."""
    boxes = np.array([row[0] for row in rows], dtype=np.float32).reshape(1, -1, 4)
    scores = np.zeros((1, 365, len(rows)), dtype=np.float32)
    for i, (_, labels) in enumerate(rows):
        for name, confidence in labels.items():
            scores[0, LABELS.index(name), i] = confidence
    return boxes, scores


def decode_detections(boxes, scores, shape, size, classes, conf, iou, agnostic):
    return postprocess(select_candidates(boxes, scores, conf), shape, size, classes, conf, iou, agnostic)


class PPYOLOETests(unittest.TestCase):
    def test_exact_catalog_and_selected_class_ids(self):
        self.assertEqual(len(LABELS), 365)
        self.assertEqual(len(set(LABELS)), 365)
        self.assertEqual(len(CLASS_IDS), 78)
        self.assertEqual(len(SELECTED), sum(map(len, CATALOG["grocery_groups"].values())))
        self.assertEqual(LABELS[8], "Bottle")
        self.assertEqual(LABELS[64], "Canned")
        self.assertEqual(LABELS[82], "Apple")
        self.assertEqual(LABELS[303], "Chips")
        self.assertNotIn("Person", SELECTED)
        self.assertNotIn("Cleaning Products", SELECTED)

    def test_profile_routes_to_paddle_loader(self):
        with patch.object(app, "MODEL_PROFILE", "ppyoloe_objects365"), patch("ppyoloe_model.load_ppyoloe") as load:
            self.assertIs(app.load_model(), load.return_value)
            load.assert_called_once_with(ROOT)

    def test_preprocessing_rgb_normalization_and_square_resize(self):
        frame = np.full((480, 640, 3), (0, 128, 255), dtype=np.uint8)
        tensor = preprocess(frame, 320)
        self.assertEqual(tensor.shape, (1, 3, 320, 320))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_allclose(tensor[0, :, 0, 0], [1, 128 / 255, 0])

    def test_excluded_winner_is_not_relabelled_as_food(self):
        raw = predictions([([100, 100, 200, 200], {"Person": .95, "Apple": .8}),
                           ([300, 100, 400, 200], {"Apple": .9})])
        rows = decode_detections(*raw, (480, 640, 3), 640, CLASS_IDS, .1, .45, True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0, 5], LABELS.index("Apple"))
        # y coordinates undo the stretched 640x640 inference input.
        np.testing.assert_allclose(rows[0, :4], [300, 75, 400, 150])

    def test_cross_class_duplicates_suppressed_with_separate_item_retained(self):
        raw = predictions([([100, 100, 200, 200], {"Apple": .9}),
                           ([110, 110, 210, 210], {"Orange/Tangerine": .85}),
                           ([300, 100, 400, 200], {"Banana": .8})])
        rows = decode_detections(*raw, (480, 640, 3), 640, CLASS_IDS, .1, .45, True)
        self.assertEqual([LABELS[int(i)] for i in rows[:, 5]], ["Apple", "Banana"])

    def test_empty_invalid_and_wrong_head_outputs(self):
        for raw in [predictions([]), predictions([([5, 5, 5, 10], {"Apple": .9})]),
                    predictions([([0, 0, float("nan"), 10], {"Apple": .9})])]:
            result = decode_detections(*raw, (480, 640, 3), 640, CLASS_IDS, .1, .45, True)
            self.assertEqual(result.shape, (0, 6))
        with self.assertRaisesRegex(ValueError, "365"):
            decode_detections(np.zeros((1, 5, 4)), np.zeros((1, 80, 5)),
                              (480, 640, 3), 640, CLASS_IDS, .1, .45, True)

    def call_model(self, model, **overrides):
        kwargs = dict(persist=True, tracker=TRACKER_CONFIG, classes=CLASS_IDS,
                      conf=.1, imgsz=640, device="cpu", agnostic_nms=True, iou=.45, nms=True)
        kwargs.update(overrides)
        return model.track(np.zeros((480, 640, 3), np.uint8), **kwargs)[0]

    def test_all_78_labels_reach_real_bytetrack_with_stable_ids(self):
        rows = []
        for i, name in enumerate(sorted(SELECTED)):
            x, y = (i % 13) * 45, (i // 13) * 70
            rows.append(([x, y, x + 25, y + 25], {name: .9}))
        session = Mock()
        session.predict.return_value = select_candidates(*predictions(rows), .1)
        model = PPYOLOEModel(session, LABELS, 640)
        first, second = self.call_model(model), self.call_model(model)
        self.assertEqual(len(first.boxes), 78)
        self.assertEqual({model.names[int(i)] for i in first.boxes.cls}, SELECTED)
        np.testing.assert_array_equal(first.boxes.id, second.boxes.id)
        session.predict.return_value = select_candidates(*predictions([]), .1)
        self.assertEqual(len(self.call_model(model).boxes), 0)
        self.assertEqual(model.tracker.frame_id, 3)
        session.predict.return_value = select_candidates(*predictions(rows), .1)
        recovered = self.call_model(model)
        np.testing.assert_array_equal(first.boxes.id, recovered.boxes.id)

    def test_wrong_device_or_size_is_rejected(self):
        session = Mock()
        model = PPYOLOEModel(session, LABELS, 640)
        for override in ({"device": "mps"}, {"imgsz": 320}, {"nms": False}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.call_model(model, **override)
        session.predict.assert_not_called()

    def test_missing_worker_environment_fails_before_spawning(self):
        with patch.dict("os.environ", {"LIGHTSTORE_PADDLE_PYTHON": "/nonexistent/paddle/python"}), \
                patch("ppyoloe_model.subprocess.Popen") as spawn:
            with self.assertRaises(FileNotFoundError):
                PaddleWorker(ROOT, 640, .1)
            spawn.assert_not_called()

    def test_wrong_worker_identity_is_rejected_and_closed(self):
        good = dict(labels=LABELS, device="cpu", size=640, checkpoint_sha256=CHECKPOINT_SHA256)
        for change in ({"device": "gpu:0"}, {"labels": LABELS[::-1]}, {"checkpoint_sha256": "wrong"}):
            worker = Mock(metadata={**good, **change})
            with self.subTest(change=change), patch("ppyoloe_model.PaddleWorker", return_value=worker), \
                    patch("config.FOOD_CLASSES", SELECTED), patch("config.INFERENCE_SIZE", 640):
                with self.assertRaises(ValueError):
                    load_ppyoloe(ROOT)
                worker.close.assert_called_once()

    def test_worker_failure_closes_stream(self):
        worker = object.__new__(PaddleWorker)
        import threading
        worker.lock = threading.Lock()
        worker.process = Mock()
        worker.process.poll.return_value = None
        worker.connection = Mock()
        worker.close = Mock()
        worker._response = Mock(side_effect=RuntimeError("worker crashed"))
        with self.assertRaisesRegex(RuntimeError, "worker crashed"):
            worker.predict(np.zeros((480, 640, 3), dtype=np.uint8))
        worker.close.assert_called_once()

    def test_protocol_framing_limits_and_eof(self):
        left, right = socket.socketpair()
        try:
            left.settimeout(1)
            send(right, b"first")
            send(right, b"second")
            self.assertEqual(receive(left), b"first")
            self.assertEqual(receive(left), b"second")
            right.sendall(struct.pack("!I", MAX_MESSAGE_BYTES + 1))
            with self.assertRaises(ValueError):
                receive(left)
            right.close()
            with self.assertRaises(EOFError):
                receive(left)
        finally:
            left.close()
            right.close()

    def test_ppyoloe_new_track_threshold(self):
        worker = Mock()
        model = PPYOLOEModel(worker, LABELS, 640)
        tracker = str(ROOT / "bytetrack-ppyoloe.yaml")
        worker.predict.return_value = select_candidates(*predictions([([100,100,200,200], {"Apple": .49})]), .1)
        self.assertEqual(len(self.call_model(model, tracker=tracker).boxes), 0)
        worker.predict.return_value = select_candidates(*predictions([([100,100,200,200], {"Apple": .51})]), .1)
        self.call_model(model, tracker=tracker)
        self.assertEqual(len(self.call_model(model, tracker=tracker).boxes), 1)


if __name__ == "__main__":
    unittest.main()
