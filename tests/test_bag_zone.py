"""Dynamic bag zone: geometry, tracker pause/relocation, zone state machine.

No model weights needed: the zone tracker is driven by a fake localizer
returning synthetic masks.
"""

import unittest

import numpy as np

from bag_zone import (
    BagZoneTracker, NoZone, ZoneTest, box_overlap_fraction, fill_footprint,
    grid_iou, motion_energy, rasterize,
)
from tracking import Detection, PackingTracker


def ring_mask(cx=320, cy=240, outer=120, inner=60):
    mask = np.zeros((480, 640), bool)
    ys, xs = np.mgrid[0:480, 0:640]
    dist = np.hypot(xs - cx, ys - cy)
    mask[(dist <= outer) & (dist >= inner)] = True
    return mask


def disc_mask(cx=320, cy=240, radius=120):
    mask = np.zeros((480, 640), bool)
    ys, xs = np.mgrid[0:480, 0:640]
    mask[np.hypot(xs - cx, ys - cy) <= radius] = True
    return mask


class FakeLocalizer:
    """Returns a scripted mask per call (or None for a miss)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def localize(self, frame):
        self.calls += 1
        mask = self.script[min(self.calls - 1, len(self.script) - 1)]
        if mask is None:
            return None
        return {"mask": mask, "conf": 0.4, "inference_ms": 1.0}


def blank_frame():
    return np.zeros((480, 640, 3), np.uint8)


class FootprintTest(unittest.TestCase):
    def test_fill_covers_hollow_center(self):
        ring = ring_mask()
        self.assertFalse(ring[240, 320])  # hollow before fill
        filled = fill_footprint(ring)
        self.assertTrue(filled[240, 320])
        self.assertGreater(filled.mean(), ring.mean())

    def test_empty_mask_stays_empty(self):
        self.assertFalse(fill_footprint(np.zeros((480, 640), bool)).any())

    def test_overlap_fraction(self):
        grid = rasterize(None)
        grid[:, :] = 0
        grid[30:90, 40:120] = 1  # left-middle block in 160x120 grid
        # Box fully inside the block.
        self.assertGreater(box_overlap_fraction((160, 120, 300, 300), grid), 0.9)
        # Box far outside.
        self.assertEqual(box_overlap_fraction((500, 300, 600, 400), grid), 0.0)

    def test_zone_test_center_or_overlap(self):
        grid = np.zeros((120, 160), np.uint8)
        grid[30:90, 40:120] = 1
        test = ZoneTest(grid, 0.30, version=1)
        self.assertTrue(test((160, 120, 300, 300)))  # center inside
        self.assertFalse(test((500, 300, 600, 400)))
        self.assertFalse(test((0, 0, 0, 0)))
        self.assertIsInstance(NoZone()((10, 10, 50, 50)), bool)
        self.assertFalse(NoZone()((10, 10, 50, 50)))

    def test_grid_iou(self):
        grid = np.zeros((120, 160), np.uint8)
        grid[30:90, 40:120] = 1
        self.assertEqual(grid_iou(grid, grid.copy()), 1.0)
        self.assertEqual(grid_iou(grid, np.zeros_like(grid)), 0.0)

    def test_rim_hysteresis(self):
        grid = np.zeros((120, 160), np.uint8)
        grid[30:90, 40:120] = 1  # footprint block in 160x120 cells
        test = ZoneTest(grid, 0.30, version=1, margin_px=16.0)
        # Well inside / well outside agree on both predicates.
        self.assertTrue(test((160, 120, 300, 300)))
        self.assertFalse(test((500, 300, 600, 400)))
        self.assertTrue(test.clearly_outside((500, 300, 600, 400)))
        # Straddling the rim: not inside, but not clearly outside either.
        rim_box = (430, 300, 550, 420)  # overlaps the block's corner lightly
        self.assertFalse(test(rim_box))
        self.assertFalse(test.clearly_outside(rim_box))


class TrackerPauseTest(unittest.TestCase):
    def test_pause_blocks_inside_streak_but_keeps_outside(self):
        tracker = PackingTracker()
        outside = (10.0, 10.0, 60.0, 60.0)
        inside = (250.0, 200.0, 330.0, 300.0)
        for _ in range(3):
            tracker.update([Detection(1, "Bottle", outside)])
        tracker.set_packing_paused(True)
        for _ in range(10):
            events = tracker.update([Detection(1, "Bottle", inside)])
            self.assertEqual(events, [])
        tracker.set_packing_paused(False)
        events = []
        for _ in range(3):
            events += tracker.update([Detection(1, "Bottle", inside)])
        self.assertEqual(len(events), 1)

    def test_relocation_discards_streaks_keeps_packed(self):
        tracker = PackingTracker()
        outside = (10.0, 10.0, 60.0, 60.0)
        inside = (250.0, 200.0, 330.0, 300.0)
        for _ in range(3):  # track 1: seen outside
            tracker.update([Detection(1, "Bottle", outside)])
        for _ in range(3):  # track 2: packed
            tracker.update([Detection(2, "Bottle", outside)])
        for _ in range(3):
            tracker.update([Detection(2, "Bottle", inside)])
        self.assertEqual(tracker.packed_counts["Bottle"], 1)
        tracker.note_zone_relocation()
        state = tracker.tracks[1]
        self.assertFalse(state.seen_outside)
        self.assertEqual((state.inside_frames, state.outside_frames), (0, 0))
        # Packed track survives relocation untouched.
        self.assertTrue(tracker.tracks[2].packed)
        self.assertEqual(tracker.packed_counts["Bottle"], 1)

    def test_zone_motion_alone_cannot_pack_stationary_product(self):
        """Stationary bottle crossed by a moving bag: no packing event."""
        tracker = PackingTracker()
        bottle = (500.0, 300.0, 580.0, 420.0)  # never moves
        far = ZoneTest(rasterize(None), 0.30, version=1)  # empty zone
        near_grid = np.zeros((120, 160), np.uint8)
        near_grid[60:110, 110:150] = 1  # covers the bottle
        near = ZoneTest(near_grid, 0.30, version=2)
        tracker.zone_test = far
        for _ in range(3):
            self.assertEqual(tracker.update([Detection(7, "Bottle", bottle)]), [])
        # Bag jumps onto the stationary bottle: pause + relocation, as the
        # zone tracker does on a significant move.
        tracker.set_packing_paused(True)
        tracker.zone_test = near
        tracker.note_zone_relocation()
        for _ in range(5):
            self.assertEqual(tracker.update([Detection(7, "Bottle", bottle)]), [])
        # ... and even after the bag stabilizes, the motionless bottle must
        # re-prove itself from outside first.
        tracker.set_packing_paused(False)
        for _ in range(3):
            self.assertEqual(tracker.update([Detection(7, "Bottle", bottle)]), [])
        self.assertEqual(tracker.packed_counts.get("Bottle", 0), 0)

    def test_outside_evidence_goes_stale_on_zone_version_bump(self):
        """Gradual zone creep onto a stationary product cannot pack it."""
        tracker = PackingTracker()
        bottle = (300.0, 220.0, 340.0, 260.0)
        far_grid = np.zeros((120, 160), np.uint8)
        near_grid = np.zeros((120, 160), np.uint8)
        near_grid[40:100, 60:110] = 1  # covers the bottle
        tracker.zone_test = ZoneTest(far_grid, 0.30, version=1)
        tracker.zone_version = 1
        for _ in range(3):  # outside under v1
            tracker.update([Detection(7, "Bottle", bottle)])
        # Zone creeps (no explicit relocation): version bump invalidates it.
        tracker.zone_test = ZoneTest(near_grid, 0.30, version=2)
        tracker.zone_version = 2
        for _ in range(5):  # inside under v2, no fresh outside evidence
            self.assertEqual(tracker.update([Detection(7, "Bottle", bottle)]), [])
        # Fresh outside observations under v2 re-arm the transfer.
        tracker.zone_test = ZoneTest(far_grid, 0.30, version=2)
        for _ in range(3):
            tracker.update([Detection(7, "Bottle", bottle)])
        tracker.zone_test = ZoneTest(near_grid, 0.30, version=2)
        events = []
        for _ in range(3):
            events += tracker.update([Detection(7, "Bottle", bottle)])
        self.assertEqual(len(events), 1)


class ZoneStateMachineTest(unittest.TestCase):
    def test_acquire_stable_then_heartbeat(self):
        zone = BagZoneTracker(FakeLocalizer([disc_mask()] * 10),
                              heartbeat_frames=10)
        for frame_id in (1, 2):
            state = zone.update(blank_frame(), frame_id)
            self.assertEqual(state["status"], "locating")
        state = zone.update(blank_frame(), 3)
        self.assertEqual(state["status"], "stable")
        self.assertFalse(zone.packing_paused)
        calls = zone.localizer.calls
        # Stable: no inference until the heartbeat frame.
        zone.update(blank_frame(), 4)
        self.assertEqual(zone.localizer.calls, calls)
        zone.update(blank_frame(), 13)
        self.assertGreater(zone.localizer.calls, calls)

    def test_stale_frame_id_cannot_move_zone_back(self):
        first = disc_mask(320, 240)
        zone = BagZoneTracker(FakeLocalizer([first] * 5))
        for frame_id in (1, 2, 3):
            zone.update(blank_frame(), frame_id)
        bbox_before = list(zone.zone_bbox)
        # An old prediction re-delivered late must not shift the zone.
        zone.update(blank_frame(), 2)
        self.assertEqual(list(zone.zone_bbox), bbox_before)

    def test_relocation_pauses_and_requires_relock(self):
        relocations = []
        script = [disc_mask(320, 240)] * 3 + [disc_mask(100, 100, 60)] * 6
        zone = BagZoneTracker(FakeLocalizer(script), heartbeat_frames=1,
                              on_relocation=lambda: relocations.append(1))
        for frame_id in (1, 2, 3):
            zone.update(blank_frame(), frame_id)
        self.assertEqual(zone.status, "stable")
        state = zone.update(blank_frame(), 4)
        self.assertEqual(state["status"], "moving")
        self.assertTrue(zone.packing_paused)
        # Relocation fires on the move and again on the re-lock adopt.
        self.assertEqual(len(relocations), 2)
        # Frozen zone keeps the old footprint while moving.
        self.assertIsNotNone(zone.zone_bbox)
        for frame_id in (5, 6, 7):
            state = zone.update(blank_frame(), frame_id)
        self.assertEqual(state["status"], "stable")
        self.assertFalse(zone.packing_paused)

    def test_loss_after_misses(self):
        script = [disc_mask()] * 3 + [None] * 4
        zone = BagZoneTracker(FakeLocalizer(script), heartbeat_frames=1,
                              misses_to_lose=2)
        for frame_id in (1, 2, 3):
            zone.update(blank_frame(), frame_id)
        self.assertEqual(zone.status, "stable")
        zone.update(blank_frame(), 4)
        zone.update(blank_frame(), 5)
        self.assertEqual(zone.status, "lost")
        self.assertTrue(zone.packing_paused)

    def test_motion_trigger(self):
        zone = BagZoneTracker(FakeLocalizer([disc_mask()] * 10),
                              heartbeat_frames=100, motion_threshold=12.0)
        for frame_id in (1, 2, 3):
            zone.update(blank_frame(), frame_id)
        calls = zone.localizer.calls
        moved = np.full((480, 640, 3), 255, np.uint8)
        zone.update(moved, 4)
        self.assertGreater(zone.localizer.calls, calls)

    def test_motion_energy_separation(self):
        still = np.zeros((480, 640, 3), np.uint8)
        import cv2

        prev = cv2.cvtColor(cv2.resize(still, (160, 120)), cv2.COLOR_BGR2GRAY)
        self.assertLess(motion_energy(prev, still, (60, 80, 580, 470)), 1.0)
        changed = still.copy()
        changed[100:400, 100:500] = 200
        self.assertGreater(motion_energy(prev, changed, (60, 80, 580, 470)), 12.0)


class _Array:
    """Minimal tensor stand-in for the YOLO result adapter."""

    def __init__(self, values):
        self.values = values

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class DynamicPipelineTest(unittest.TestCase):
    """Full FrameProcessor run: scripted bag zone + stub product detector."""

    def _processor(self, script):
        import cv2
        from types import SimpleNamespace

        from config import FOOD_CLASSES
        from vision import FrameProcessor

        names = {i: name for i, name in enumerate(sorted(FOOD_CLASSES))}
        bottle_id = next(i for i, n in names.items() if n == "Bottle")

        class StubModel:
            def __init__(self):
                self.names = names
                self.rows = []

            def track(self, frame, **kwargs):
                boxes = SimpleNamespace(
                    xyxy=_Array([r[0] for r in self.rows]),
                    cls=_Array([r[1] for r in self.rows]),
                    id=_Array([r[2] for r in self.rows]) if self.rows else None,
                )
                return [SimpleNamespace(boxes=boxes, names=names)]

        model = StubModel()
        # Heartbeat every frame: the scripted localizer drives the zone
        # through acquire -> stable -> lost within a few processed frames.
        zone = BagZoneTracker(FakeLocalizer(script), heartbeat_frames=1)
        processor = FrameProcessor(model, identifier=None, bag_zone=zone)
        ok, jpeg = cv2.imencode(
            ".jpg", np.zeros((480, 640, 3), np.uint8))
        self.assertTrue(ok)
        return processor, model, bottle_id, jpeg.tobytes()

    def test_packing_flows_through_dynamic_zone_then_pauses_on_loss(self):
        import base64

        import cv2

        script = [disc_mask(320, 240, 120)] * 9 + [None] * 10
        processor, model, bottle_id, jpeg = self._processor(script)
        outside = (40.0, 40.0, 100.0, 100.0)
        inside = (300.0, 220.0, 340.0, 260.0)  # center in the disc
        # Frames 1-3: bag acquisition (no product yet).
        for _ in range(3):
            model.rows = []
            response = processor.process_detect(jpeg)
        self.assertEqual(response["bag_zone"]["status"], "stable")
        # Genuine transfer: outside x3, then inside x3 -> one packed event.
        for index in range(3):
            model.rows = [(outside, bottle_id, 7)]
            response = processor.process_detect(jpeg)
        self.assertEqual(response["packed_total"], 0)
        for index in range(3):
            model.rows = [(inside, bottle_id, 7)]
            response = processor.process_detect(jpeg)
        self.assertEqual(response["packed_total"], 1)
        self.assertEqual(response["events"][0]["track_id"], 7)
        # Fitted contour is drawn (green channel dominates on the outline).
        image = cv2.imdecode(
            np.frombuffer(base64.b64decode(response["image"]), np.uint8), 1)
        green = image[:, :, 1].astype(int) - image[:, :, 2].astype(int)
        self.assertGreater((green > 60).sum(), 200)
        # Bag disappears -> lost -> packing pauses; the same inside box
        # sitting in the frozen zone produces no further events.
        for _ in range(4):
            model.rows = [(inside, bottle_id, 9)]
            response = processor.process_detect(jpeg)
        self.assertEqual(response["bag_zone"]["status"], "lost")
        self.assertTrue(response["bag_zone"]["packing_paused"])
        self.assertEqual(response["packed_total"], 1)
        self.assertEqual(response["events"], [])


if __name__ == "__main__":
    unittest.main()
