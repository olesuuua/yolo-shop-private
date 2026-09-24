"""Regression: automatic bag recovery and 70%-overlap packing.

Covers the LightStore recovery spec without model weights or video:
synthetic masks drive BagZoneTracker, synthetic boxes drive PackingTracker.
"""

import unittest

import numpy as np

from bag_zone import (
    BagZoneTracker, NoZone, ZoneTest, box_overlap_fraction, fill_footprint,
)
from tracking import Detection, PackingTracker


def disc_mask(cx=320, cy=240, radius=120):
    mask = np.zeros((480, 640), bool)
    ys, xs = np.mgrid[0:480, 0:640]
    mask[np.hypot(xs - cx, ys - cy) <= radius] = True
    return mask


def ellipse_mask(cx=320, cy=240, rx=150, ry=80):
    mask = np.zeros((480, 640), bool)
    ys, xs = np.mgrid[0:480, 0:640]
    mask[((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1] = True
    return mask


class FakeLocalizer:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def localize(self, frame):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if item is None:
            return None
        return {"mask": item, "conf": 0.4, "inference_ms": 1.0}


def blank_frame():
    return np.zeros((480, 640, 3), np.uint8)


def lock_zone(zone, mask, start_id=1, now=100.0):
    for frame_id in range(start_id, start_id + 3):
        zone.update(blank_frame(), frame_id, now=now)
    assert zone.status == "stable", zone.status
    return zone.snapshot()["contour"]


class ReacquisitionTest(unittest.TestCase):
    def _lose_bag(self, zone, next_id, now=110.0):
        # Misses drive stable -> grace -> lost through the real path.
        zone.update(blank_frame(), next_id, now=now)
        self.assertEqual(zone.status, "grace")
        zone.update(blank_frame(), next_id + 1, now=now + 1.0)
        zone.update(blank_frame(), next_id + 2, now=now + 2.1)
        self.assertEqual(zone.status, "lost")
        self.assertIsNone(zone.last_locked_grid)
        self.assertEqual(zone.snapshot()["contour"], [])
        return next_id + 3

    def test_reacquire_after_position_change(self):
        zone = BagZoneTracker(FakeLocalizer([disc_mask()] * 3 + [None] * 3
                                             + [disc_mask(200, 150, 100)] * 5))
        old = lock_zone(zone, disc_mask())
        fid = self._lose_bag(zone, 4)
        for step in range(5):
            state = zone.update(blank_frame(), fid + step, now=120.0 + step * 0.1)
        self.assertEqual(state["status"], "stable")
        self.assertNotEqual(zone.snapshot()["contour"], old)
        self.assertIsNotNone(zone.last_locked_grid)

    def test_reacquire_after_size_change(self):
        # A smaller bag at the same spot is ambiguous with occlusion, so it
        # is held while prior context is fresh, then accepted once the prior
        # lapses (bounded delay, never a permanent block).
        zone = BagZoneTracker(FakeLocalizer([disc_mask(320, 240, 120)] * 3
                                             + [None] * 3
                                             + [disc_mask(320, 240, 70)] * 9),
                              prior_ttl_s=1.0)
        old = lock_zone(zone, disc_mask())
        fid = self._lose_bag(zone, 4)
        # Prior fresh (lost at t=112.1, TTL 1s): same-position subset held.
        for step, moment in enumerate((112.2, 112.3, 112.4)):
            state = zone.update(blank_frame(), fid + step, now=moment)
        self.assertEqual(state["status"], "lost")
        self.assertIn("occlusion", zone.snapshot()["bag_diag"]["reason"])
        self.assertEqual(zone.snapshot()["contour"], [])
        # Prior expired: the stably visible shape locks as the new bag.
        for step, moment in enumerate((114.0, 114.1, 114.2), start=3):
            state = zone.update(blank_frame(), fid + step, now=moment)
        self.assertEqual(state["status"], "stable")
        self.assertNotEqual(zone.snapshot()["contour"], old)

    def test_reacquire_after_shape_change(self):
        zone = BagZoneTracker(FakeLocalizer([disc_mask()] * 3 + [None] * 3
                                             + [ellipse_mask()] * 5))
        old = lock_zone(zone, disc_mask())
        fid = self._lose_bag(zone, 4)
        for step in range(5):
            state = zone.update(blank_frame(), fid + step, now=120.0 + step * 0.1)
        self.assertEqual(state["status"], "stable")
        self.assertNotEqual(zone.snapshot()["contour"], old)

    def test_partial_mask_rejected_while_locked(self):
        base = disc_mask()
        notch = base.copy()
        notch[205:270, 425:460] = False
        zone = BagZoneTracker(FakeLocalizer([base] * 3 + [notch] * 3))
        locked = lock_zone(zone, base)
        zone.update(blank_frame(), 4, now=110.0)
        self.assertEqual(zone.status, "grace")
        # The full outline survives: geometry, contour, and reference kept.
        self.assertEqual(zone.snapshot()["contour"], locked)
        self.assertIsNotNone(zone.last_locked_grid)
        self.assertEqual(zone.snapshot()["last_refresh_rejection"], "shape")


def footprint_grid():
    grid = np.zeros((120, 160), np.uint8)
    grid[30:90, 40:120] = 1  # pixels x160-480, y120-360
    return grid


class OverlapRuleTest(unittest.TestCase):
    def test_seventy_percent_boundary(self):
        grid = footprint_grid()
        test = ZoneTest(grid, 0.70, version=1)
        # Fully inside small product: overlap ~1.0 -> inside.
        self.assertTrue(test((300.0, 200.0, 360.0, 280.0)))
        # ~80% overlap -> inside.
        self.assertTrue(test((400.0, 200.0, 500.0, 280.0)))
        self.assertGreaterEqual(
            box_overlap_fraction((400.0, 200.0, 500.0, 280.0), grid), 0.70)
        # ~60% overlap -> outside even though the center is inside.
        mostly_out = (420.0, 200.0, 520.0, 280.0)
        self.assertLess(
            box_overlap_fraction(mostly_out, grid), 0.70)
        self.assertFalse(test(mostly_out))
        # Huge center-inside box with small overlap -> never inside.
        huge = (-100.0, -100.0, 400.0, 400.0)
        self.assertLess(box_overlap_fraction(huge, grid), 0.70)
        self.assertFalse(test(huge))
        # Far outside -> not inside and clearly outside.
        self.assertFalse(test((500.0, 300.0, 600.0, 400.0)))
        self.assertTrue(test.clearly_outside((500.0, 300.0, 600.0, 400.0)))

    def test_one_frame_inside_never_counts(self):
        tracker = PackingTracker(
            zone_test=ZoneTest(footprint_grid(), 0.70, version=1),
            min_inside_frames=3, min_outside_frames=1)
        tracker.zone_version = 1
        inside = (300.0, 200.0, 360.0, 280.0)
        self.assertEqual(tracker.update([Detection(5, "Bottle", inside)]), [])
        self.assertEqual(tracker.update([Detection(5, "Bottle", inside)]), [])
        self.assertEqual(tracker.packed_counts.get("Bottle", 0), 0)
        events = tracker.update([Detection(5, "Bottle", inside)])
        self.assertEqual(len(events), 1)


class AcquisitionCountingTest(unittest.TestCase):
    def test_unseen_item_inside_at_acquisition_counts(self):
        tracker = PackingTracker(
            zone_test=ZoneTest(footprint_grid(), 0.70, version=1),
            min_inside_frames=3, min_outside_frames=3)
        tracker.zone_version = 1
        inside = (300.0, 200.0, 360.0, 280.0)
        # Never seen outside: consistent inside still counts (no one-frame).
        self.assertEqual(tracker.update([Detection(9, "Bottle", inside)]), [])
        self.assertEqual(tracker.update([Detection(9, "Bottle", inside)]), [])
        events = tracker.update([Detection(9, "Bottle", inside)])
        self.assertEqual(len(events), 1)
        self.assertEqual(tracker.packed_counts["Bottle"], 1)

    def test_unseen_item_inside_after_reacquisition_counts(self):
        tracker = PackingTracker(
            zone_test=ZoneTest(footprint_grid(), 0.70, version=1),
            min_inside_frames=3, min_outside_frames=1)
        tracker.zone_version = 1
        inside_old = (300.0, 200.0, 360.0, 280.0)
        for _ in range(3):
            tracker.update([Detection(7, "Bottle", inside_old)])
        self.assertEqual(tracker.packed_counts["Bottle"], 1)
        # Bag lost and reacquired elsewhere; a NEW item sits in the new bag.
        tracker.note_zone_relocation()
        grid2 = np.zeros((120, 160), np.uint8)
        grid2[70:110, 10:50] = 1  # pixels x40-200, y280-440 (elsewhere)
        tracker.zone_test = ZoneTest(grid2, 0.70, version=2)
        tracker.zone_version = 2
        inside_new = (80.0, 320.0, 140.0, 400.0)  # fully inside grid2
        self.assertTrue(tracker.zone_test(inside_new))
        self.assertEqual(tracker.update([Detection(11, "Bottle", inside_new)]), [])
        self.assertEqual(tracker.update([Detection(11, "Bottle", inside_new)]), [])
        events = tracker.update([Detection(11, "Bottle", inside_new)])
        self.assertEqual(len(events), 1)
        self.assertEqual(tracker.packed_counts["Bottle"], 2)

    def test_no_duplicate_across_loss_and_reacquisition(self):
        tracker = PackingTracker(
            zone_test=ZoneTest(footprint_grid(), 0.70, version=1),
            min_inside_frames=3, min_outside_frames=1)
        tracker.zone_version = 1
        inside = (300.0, 200.0, 360.0, 280.0)
        for _ in range(3):
            tracker.update([Detection(7, "Bottle", inside)])
        self.assertEqual(tracker.packed_counts["Bottle"], 1)
        tracker.note_zone_relocation()
        tracker.zone_test = ZoneTest(footprint_grid(), 0.70, version=2)
        tracker.zone_version = 2
        # Same track ID, still visible: evaluated afresh but never recounted.
        for _ in range(5):
            self.assertEqual(
                tracker.update([Detection(7, "Bottle", inside)]), [])
        self.assertEqual(tracker.packed_counts["Bottle"], 1)


class DisappearanceTest(unittest.TestCase):
    def test_disappearance_during_loss_never_counts(self):
        tracker = PackingTracker(
            zone_test=ZoneTest(footprint_grid(), 0.70, version=1),
            min_inside_frames=3, min_outside_frames=1,
            track_ttl_frames=60, pending_pack_seconds=2.0)
        tracker.zone_version = 1
        outside = (10.0, 10.0, 60.0, 60.0)
        tracker.update([Detection(7, "Bottle", outside)], now=100.0)
        # Bag lost: relocation clears evidence, NoZone pauses counting.
        tracker.note_zone_relocation()
        tracker.zone_test = NoZone()
        tracker.zone_version = -1
        tracker.set_packing_paused(True)
        for moment in (100.0, 101.0, 102.0, 103.0, 110.0):
            self.assertEqual(tracker.update([], now=moment), [])
        # Reacquired, item still invisible: still nothing, no late pending.
        tracker.note_zone_relocation()
        tracker.zone_test = ZoneTest(footprint_grid(), 0.70, version=2)
        tracker.zone_version = 2
        tracker.set_packing_paused(False)
        for moment in (111.0, 112.0, 113.0, 120.0):
            self.assertEqual(tracker.update([], now=moment), [])
        self.assertEqual(dict(tracker.packed_counts), {})

    def test_obsolete_pending_never_fires_after_version_change(self):
        tracker = PackingTracker(
            zone_test=ZoneTest(footprint_grid(), 0.70, version=1),
            min_inside_frames=3, min_outside_frames=1,
            track_ttl_frames=60, pending_pack_seconds=2.0)
        tracker.zone_version = 1
        outside = (10.0, 10.0, 60.0, 60.0)
        boundary = (150.0, 200.0, 230.0, 300.0)
        tracker.update([Detection(7, "Bottle", outside)], now=100.0)
        tracker.update([Detection(7, "Bottle", boundary)], now=100.0)
        self.assertEqual(tracker.update([], now=100.0), [])  # watch starts
        state = tracker.tracks[7]
        self.assertIsNotNone(state.pending_deadline)
        # Outline changes before expiry: the old watch must die silently.
        tracker.note_zone_relocation()
        tracker.zone_test = ZoneTest(footprint_grid(), 0.70, version=2)
        tracker.zone_version = 2
        self.assertEqual(tracker.update([], now=105.0), [])
        self.assertEqual(dict(tracker.packed_counts), {})


class ResetAndFootprintTest(unittest.TestCase):
    def test_reset_clears_all_bag_and_packing_state(self):
        zone = BagZoneTracker(FakeLocalizer([disc_mask()] * 5))
        lock_zone(zone, disc_mask())
        self.assertIsNotNone(zone.last_locked_grid)
        tracker = PackingTracker(
            zone_test=zone.zone_test, min_inside_frames=1, min_outside_frames=1)
        tracker.zone_version = zone.zone_version
        tracker.update([Detection(7, "Bottle", (300.0, 200.0, 360.0, 280.0))])
        zone.reset()
        self.assertEqual(zone.status, "locating")
        self.assertIsNone(zone.last_locked_grid)
        self.assertIsNone(zone.candidate_grid)
        self.assertEqual(zone.consecutive, 0)
        self.assertEqual(zone.misses, 0)
        self.assertEqual(zone.zone_frame_id, -1)
        self.assertIsInstance(zone.zone_test, NoZone)
        tracker.reset()
        self.assertEqual(dict(tracker.packed_counts), {})
        self.assertEqual(tracker.tracks, {})
        self.assertEqual(tracker.packed_ids, set())

    def test_frame_processor_reset_clears_zone_version_and_counts(self):
        from types import SimpleNamespace

        from config import FOOD_CLASSES
        from vision import FrameProcessor

        names = {i: name for i, name in enumerate(sorted(FOOD_CLASSES))}

        class StubModel:
            def __init__(self):
                self.names = names

        zone = BagZoneTracker(FakeLocalizer([disc_mask()] * 5))
        processor = FrameProcessor(StubModel(), identifier=None, bag_zone=zone)
        lock_zone(zone, disc_mask())
        processor.tracker.zone_test = zone.zone_test
        processor.tracker.zone_version = zone.zone_version
        processor.tracker.update(
            [Detection(7, "Bottle", (300.0, 200.0, 360.0, 280.0))])
        state = processor.reset()
        self.assertEqual(state["packed_counts"], {})
        self.assertEqual(processor.tracker.zone_version, -1)
        self.assertIsInstance(processor.tracker.zone_test, NoZone)
        self.assertIsNone(zone.last_locked_grid)

    def test_corner_touching_mask_does_not_flood_frame(self):
        mask = np.zeros((480, 640), bool)
        mask[0:100, 0:100] = True  # touches the top-left corner
        filled = fill_footprint(mask)
        self.assertLess(float(filled.mean()), 0.10)
        self.assertTrue(filled[50, 50])
        self.assertFalse(filled[400, 500])
        # Hollow bags still fill their interior.
        ys, xs = np.mgrid[0:480, 0:640]
        ring = ((np.hypot(xs - 320, ys - 240) <= 120)
                & (np.hypot(xs - 320, ys - 240) >= 60))
        self.assertFalse(ring[240, 320])
        self.assertTrue(fill_footprint(ring)[240, 320])


if __name__ == "__main__":
    unittest.main()
