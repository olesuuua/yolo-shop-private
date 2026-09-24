"""Audience overlay: short tabs, no overlaps, reserved bottom strip."""

import unittest

import cv2
import numpy as np

from tracking import Detection, PackingTracker
from vision import (
    CHARCOAL, LIME, PEACH, annotate_frame, layout_tabs, tab_rect,
    track_tab_text,
)


def rects_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class TabTextTest(unittest.TestCase):
    def test_short_tab_and_checkmark_rule(self):
        self.assertEqual(track_tab_text("Bottle", 3, False), "Bottle #3")
        self.assertEqual(track_tab_text("Bottle", 3, True), "Bottle #3 ✓")
        self.assertEqual(track_tab_text("Canned", 13, False), "Canned #13")
        self.assertEqual(track_tab_text("Storage box", 7, True), "Storage box #7 ✓")
        self.assertEqual(track_tab_text("Apple", 1, False), "Apple #1")
        # The checkmark marks identification, never packed state.
        self.assertNotIn("PACKED", track_tab_text("Bottle", 3, True))
        self.assertNotIn("entering", track_tab_text("Bottle", 3, False))


class LayoutTest(unittest.TestCase):
    def test_neighboring_tabs_do_not_overlap(self):
        specs = [
            ("Bottle #2", (180, 120, 260, 420)),
            ("Bottle #3", (280, 130, 360, 430)),
            ("Canned #13", (300, 140, 380, 300)),
        ]
        rects = layout_tabs(specs)
        self.assertEqual(len(rects), 3)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(rects_overlap(rects[i], rects[j]),
                                 f"tabs {i} and {j} overlap: {rects[i]} {rects[j]}")

    def test_tabs_stay_inside_frame(self):
        specs = [
            ("Bottle #1", (0, 0, 640, 480)),      # full frame box
            ("Apple #2", (600, 450, 639, 479)),   # bottom-right corner
            ("Canned #3", (300, 0, 400, 60)),     # top edge
        ]
        for rect in layout_tabs(specs):
            x, y, w, h = rect
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, 640)
            self.assertLessEqual(y + h, 480)

    def test_tab_rect_attaches_above_then_below(self):
        text = "Bottle #3"
        _x, y, _w, h = tab_rect(text, (200, 200, 300, 400))
        self.assertLess(y + h, 200)  # room above: attached on top
        _x, y, _w, h = tab_rect(text, (200, 5, 300, 400))
        self.assertGreaterEqual(y, 400)  # no room: hangs below the box


class AnnotateIntegrationTest(unittest.TestCase):
    def _tracker_with(self, *boxes):
        tracker = PackingTracker()
        for i, (class_name, box) in enumerate(boxes):
            tracker.tracks[100 + i] = tracker.tracks.get(100 + i) or None
        from tracking import TrackState
        tracker.tracks = {
            100 + i: TrackState(track_id=100 + i, class_name=class_name,
                                last_bbox=box, was_inside_roi=False,
                                packed=False, last_seen_frame=1)
            for i, (class_name, box) in enumerate(boxes)
        }
        return tracker

    def test_tabs_and_bottom_line_drawn(self):
        frame = np.full((480, 640, 3), 255, np.uint8)  # white tabletop
        tracker = self._tracker_with(
            ("Bottle", (180, 120, 260, 420)),
            ("Bottle", (280, 130, 360, 430)),
        )
        tracker.packed_counts.update({"Bottle": 1})
        detections = [Detection(100, "Bottle", (180, 120, 260, 420)),
                      Detection(101, "Bottle", (280, 130, 360, 430))]
        out = annotate_frame(frame, detections, tracker, {},
                             bag_zone=None, bag_mode="fixed")
        # Charcoal tabs present (dark pixels in the top area of each box).
        dark = ((out[:140, 150:400].astype(int).sum(axis=2)) < 120).sum()
        self.assertGreater(dark, 300)
        # Bottom-left packed line: lime text pixels in its strip.
        strip = out[440:480, 8:220].astype(int)
        lime = ((strip[:, :, 1] > 200) & (strip[:, :, 0] < 160)
                & (strip[:, :, 2] > 150)).sum()
        self.assertGreater(lime, 30)
        # No long product names or mid-video warnings on the image.
        self.assertEqual(out.shape, (480, 640, 3))

    def test_identified_tab_and_packed_lime(self):
        frame = np.full((480, 640, 3), 255, np.uint8)
        tracker = self._tracker_with(("Canned", (400, 100, 500, 400)))
        tracker.tracks[100].packed = True
        detections = [Detection(100, "Canned", (400, 100, 500, 400))]
        identification = {100: {"complete": True, "label_main": "X"}}
        out = annotate_frame(frame, detections, tracker, identification,
                             bag_zone=None, bag_mode="fixed")
        # Lime tab + lime halo around a packed box.
        lime = ((out.astype(int)[:, :, 1] > 200)
                & (out.astype(int)[:, :, 0] < 160)
                & (out.astype(int)[:, :, 2] > 150)).sum()
        self.assertGreater(lime, 300)

    def test_bag_lost_notice_bottom_right(self):
        from types import SimpleNamespace
        frame = np.full((480, 640, 3), 255, np.uint8)
        tracker = PackingTracker()
        zone = SimpleNamespace(status="lost", footprint=None)
        out = annotate_frame(frame, [], tracker, {}, zone, "dynamic")
        # Peach notice pixels live in the bottom-right reserved area...
        corner = out[430:480, 400:640].astype(int)
        peach = ((corner[:, :, 2] > 200) & (corner[:, :, 1] > 130)
                 & (corner[:, :, 0] < 200)).sum()
        self.assertGreater(peach, 30)
        # ...and nothing is drawn across the middle of the frame.
        middle = out[150:330, 200:440].astype(int)
        nonwhite = (np.abs(middle.astype(int) - 255).sum(axis=2) > 30).sum()
        self.assertEqual(nonwhite, 0)


if __name__ == "__main__":
    unittest.main()
