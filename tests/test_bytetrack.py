"""Check the detector threshold against real ByteTrack, without loading weights."""

import unittest

import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.nms import non_max_suppression

from config import AGNOSTIC_NMS, CONF_THRESHOLD, NMS_IOU_THRESHOLD, TRACKER_CONFIG


class ByteTrackConfidenceTests(unittest.TestCase):
    def setUp(self):
        settings = IterableSimpleNamespace(**YAML.load(check_yaml(TRACKER_CONFIG)))
        self.tracker = BYTETracker(settings)

    def update(self, confidence):
        # A stationary product whose detector confidence drops during occlusion.
        rows = [] if confidence is None else [[50, 50, 100, 100, confidence, 0]]
        boxes = Boxes(np.array(rows, dtype=np.float32).reshape(-1, 6), (480, 640))
        boxes = boxes[boxes.conf >= CONF_THRESHOLD]
        return self.tracker.update(boxes)

    def test_low_confidence_frame_keeps_the_existing_product_id(self):
        initial = self.update(0.90)
        self.assertEqual(len(initial), 1)
        track_id = initial[0, 4]
        for confidence in (0.20, 0.20, 0.90):
            with self.subTest(confidence=confidence):
                tracked = self.update(confidence)
                self.assertEqual(len(tracked), 1)
                self.assertEqual(tracked[0, 4], track_id)

    def test_low_confidence_boxes_do_not_start_a_product_track(self):
        for _ in range(3):
            self.assertEqual(len(self.update(0.20)), 0)

    def test_uncertain_detection_cannot_start_a_track_until_confident(self):
        for _ in range(3):
            self.assertEqual(len(self.update(0.45)), 0)
        self.update(0.85)
        self.assertEqual(len(self.update(0.85)), 1)

    def test_brief_occlusion_recovers_the_same_id(self):
        initial = self.update(0.90)
        track_id = initial[0, 4]
        for _ in range(45):
            self.assertEqual(len(self.update(None)), 0)
        recovered = self.update(0.90)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0, 4], track_id)


class DuplicateBoxTests(unittest.TestCase):
    def test_cross_class_duplicate_is_suppressed_but_second_product_remains(self):
        # Two competing labels on one product (IoU 0.67), plus a separate product.
        prediction = torch.tensor([[
            [100, 100, 100, 100, 0.90, 0.01],
            [120, 100, 100, 100, 0.01, 0.85],
            [290, 100, 100, 100, 0.80, 0.01],
        ]], dtype=torch.float32).transpose(1, 2)
        boxes = non_max_suppression(
            prediction, conf_thres=CONF_THRESHOLD, iou_thres=NMS_IOU_THRESHOLD,
            agnostic=AGNOSTIC_NMS, nc=2,
        )[0]
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[:, 0].tolist(), [50.0, 240.0])

    def test_nearby_products_are_not_merged(self):
        prediction = torch.tensor([[
            [100, 100, 100, 100, 0.90, 0.01],
            [180, 100, 100, 100, 0.85, 0.01],
        ]], dtype=torch.float32).transpose(1, 2)
        boxes = non_max_suppression(
            prediction, conf_thres=CONF_THRESHOLD, iou_thres=NMS_IOU_THRESHOLD,
            agnostic=AGNOSTIC_NMS, nc=2,
        )[0]
        self.assertEqual(len(boxes), 2)


if __name__ == "__main__":
    unittest.main()
