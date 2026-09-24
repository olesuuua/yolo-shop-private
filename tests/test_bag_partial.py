"""Partial-occlusion regressions: trusted outline vs visible segmentation.

Live failure: a ~67%-area candidate was correctly rejected by refresh, then
wrongly adopted after three similar observations, visibly shrinking the
outline around hands/bottles. Consistency never proves completeness:
adoption additionally requires spatial relocation evidence, and
same-position subsets are held as likely occlusion (also across the
loss/reacquisition boundary via prior context).

Synthetic masks only; no model weights needed.
"""

import unittest

import numpy as np

from bag_zone import (
    BagZoneTracker, classify_candidate,
)
from tracking import Detection, PackingTracker


def disc_mask(cx=320, cy=240, radius=120):
    mask = np.zeros((480, 640), bool)
    ys, xs = np.mgrid[0:480, 0:640]
    mask[np.hypot(xs - cx, ys - cy) <= radius] = True
    return mask


def partial_67_mask():
    """Same-position 66%-area subset: the live hand/bottle occlusion shape."""
    mask = disc_mask()
    mask[:, 350:] = False
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
        if isinstance(item, dict):
            found = dict(item)
            found.setdefault("inference_ms", 1.0)
            return found
        return {"mask": item, "conf": 0.4, "inference_ms": 1.0}


def blank_frame():
    return np.zeros((480, 640, 3), np.uint8)


def lock(zone, start=1, now=100.0):
    for frame_id in range(start, start + 3):
        zone.update(blank_frame(), frame_id, now=now)
    assert zone.status == "stable", zone.status


def drive(zone, tracker, frame_id, detections, now):
    """Advance zone + packing together, as FrameProcessor does."""
    zone.update(blank_frame(), frame_id, now=now)
    tracker.zone_test = zone.zone_test
    tracker.zone_version = zone.zone_test.version
    tracker.set_packing_paused(zone.packing_paused)
    return tracker.update(detections, now=now)


class VerdictTest(unittest.TestCase):
    def test_occlusion_verdict_for_67_percent_partial(self):
        from bag_zone import rasterize, footprint_contour, fill_footprint
        import numpy as np
        trusted = rasterize(footprint_contour(fill_footprint(disc_mask())))
        candidate = rasterize(
            footprint_contour(fill_footprint(partial_67_mask())))
        verdict = classify_candidate(trusted, candidate)
        self.assertEqual(verdict["verdict"], "occlusion")
        self.assertLess(verdict["area_ratio"], 0.88)
        self.assertGreaterEqual(verdict["containment"], 0.90)
        self.assertLess(verdict["extra_frac"], 0.05)
        for key in ("retained_boundary", "shift_cells", "missing_frac",
                    "iou"):
            self.assertIn(key, verdict)

    def test_relocation_verdict_for_moved_bag(self):
        from bag_zone import rasterize, footprint_contour, fill_footprint
        trusted = rasterize(footprint_contour(fill_footprint(disc_mask())))
        moved = rasterize(footprint_contour(
            fill_footprint(disc_mask(200, 150, 100))))
        verdict = classify_candidate(trusted, moved)
        self.assertEqual(verdict["verdict"], "relocation")
        self.assertGreaterEqual(verdict["extra_frac"], 0.05)

    def test_same_verdict_for_notch(self):
        from bag_zone import rasterize, footprint_contour, fill_footprint
        trusted = rasterize(footprint_contour(fill_footprint(disc_mask())))
        notch = disc_mask()
        notch[205:270, 425:460] = False
        candidate = rasterize(footprint_contour(fill_footprint(notch)))
        verdict = classify_candidate(trusted, candidate)
        self.assertEqual(verdict["verdict"], "same")

    def test_uncertain_verdict_for_rim_straddler(self):
        import numpy as np
        from bag_zone import classify_candidate
        trusted = np.zeros((120, 160), np.uint8)
        trusted[30:90, 40:120] = 1
        candidate = np.zeros((120, 160), np.uint8)
        candidate[45:85, 35:65] = 1  # small box straddling the left rim
        verdict = classify_candidate(trusted, candidate)
        self.assertEqual(verdict["verdict"], "uncertain")


class PartialEpisodeTest(unittest.TestCase):
    """The live repro: stable 67% partials must never shrink the outline."""

    def _episode_zone(self, extra_partials=4):
        script = ([disc_mask()] * 3 + [partial_67_mask()] * (2 + extra_partials))
        return BagZoneTracker(FakeLocalizer(script))

    def test_consistent_partials_never_replace_trusted(self):
        zone = self._episode_zone(extra_partials=6)
        lock(zone)
        trusted = zone.snapshot()["contour"]
        self.assertTrue(trusted)
        # Eight consecutive identical 67% partials: grace, then loss once
        # the grace period expires — adoption must never happen.
        adoption = []
        for index, moment in enumerate(
                (110.0, 110.5, 111.0, 111.5, 112.1, 113.0, 114.0, 115.0),
                start=4):
            zone.update(blank_frame(), index, now=moment)
            for transition in zone.snapshot()["bag_diag"]["transitions"]:
                if transition["to"] == "stable" and transition["frame"] >= 4:
                    adoption.append(transition)
        self.assertEqual(adoption, [])
        # Trusted outline survived until loss; then dropped, never shrunk.
        self.assertEqual(zone.status, "lost")
        self.assertEqual(zone.snapshot()["contour"], [])
        self.assertIn("occlusion", zone.snapshot()["bag_diag"]["reason"])

    def test_partials_beyond_grace_not_acquired_via_loophole(self):
        zone = self._episode_zone(extra_partials=8)
        lock(zone)
        trusted = zone.snapshot()["contour"]
        self.assertTrue(trusted)
        frame_id = 4
        for moment in (110.0, 110.5, 111.0, 111.5, 112.1):
            zone.update(blank_frame(), frame_id, now=moment)
            frame_id += 1
        self.assertEqual(zone.status, "lost")
        # Prior context is fresh: more identical partials stay held.
        for moment in (113.0, 114.0, 115.0, 116.0):
            state = zone.update(blank_frame(), frame_id, now=moment)
            frame_id += 1
            self.assertEqual(state["status"], "lost")
        reasons = [r.get("outcome") for r in
                   zone.snapshot()["bag_diag"]["recent"]]
        self.assertIn("prior_hold", reasons)
        self.assertTrue(zone.snapshot()["bag_diag"]["prior"]["active"])

    def test_recovery_to_full_outline_after_occlusion_clears(self):
        script = ([disc_mask()] * 3 + [partial_67_mask()] * 3
                  + [disc_mask()] * 4)
        zone = BagZoneTracker(FakeLocalizer(script))
        lock(zone)
        trusted = zone.snapshot()["contour"]
        zone.update(blank_frame(), 4, now=110.0)  # partial -> grace
        self.assertEqual(zone.status, "grace")
        zone.update(blank_frame(), 5, now=110.5)
        zone.update(blank_frame(), 6, now=111.0)
        zone.update(blank_frame(), 7, now=111.5)  # full bag back
        zone.update(blank_frame(), 8, now=111.6)
        zone.update(blank_frame(), 9, now=111.7)
        self.assertEqual(zone.status, "stable")
        self.assertEqual(zone.snapshot()["contour"], trusted)

    def test_actual_move_adopts_with_relocation_evidence(self):
        relocations = []
        moved = disc_mask(380, 240, 120)  # dragged half a width
        zone = BagZoneTracker(FakeLocalizer([disc_mask()] * 3 + [moved] * 4),
                              on_relocation=lambda: relocations.append(1))
        lock(zone)
        relocations.clear()
        old = zone.snapshot()["contour"]
        zone.update(blank_frame(), 4, now=110.0)
        self.assertEqual(zone.status, "grace")
        zone.update(blank_frame(), 5, now=110.5)
        zone.update(blank_frame(), 6, now=111.0)
        self.assertEqual(zone.status, "stable")
        self.assertNotEqual(zone.snapshot()["contour"], old)
        self.assertEqual(len(relocations), 1)

    def test_smaller_bag_elsewhere_adopts(self):
        small = disc_mask(200, 150, 70)
        zone = BagZoneTracker(FakeLocalizer([disc_mask()] * 3 + [small] * 4))
        lock(zone)
        old = zone.snapshot()["contour"]
        zone.update(blank_frame(), 4, now=110.0)
        zone.update(blank_frame(), 5, now=110.5)
        zone.update(blank_frame(), 6, now=111.0)
        self.assertEqual(zone.status, "stable")
        self.assertNotEqual(zone.snapshot()["contour"], old)

    def test_refresh_interval_is_fifteen_seconds(self):
        from config import BAG_REFRESH_PERIOD_S
        self.assertEqual(BAG_REFRESH_PERIOD_S, 15.0)


class EpisodePackingTest(unittest.TestCase):
    INSIDE = (300.0, 220.0, 340.0, 260.0)  # fully inside the disc

    def test_no_count_while_outline_uncertain_then_counts_after(self):
        zone = BagZoneTracker(FakeLocalizer(
            [disc_mask()] * 3 + [partial_67_mask()] * 3 + [disc_mask()] * 6))
        tracker = PackingTracker(min_inside_frames=3, min_outside_frames=1)
        lock(zone)
        frame_id, now = 4, 110.0
        # Occlusion episode: visible inside product must not count (paused).
        for _ in range(3):
            events = drive(zone, tracker, frame_id,
                           [Detection(7, "Bottle", self.INSIDE)], now)
            self.assertEqual(events, [])
            frame_id += 1
            now += 0.5
        self.assertEqual(tracker.packed_counts.get("Bottle", 0), 0)
        # Full outline returns and locks: the same product then counts once.
        for _ in range(6):
            events = drive(zone, tracker, frame_id,
                           [Detection(7, "Bottle", self.INSIDE)], now)
            frame_id += 1
            now += 0.5
        self.assertEqual(tracker.packed_counts.get("Bottle", 0), 1)
        self.assertEqual(len(tracker.packed_events), 1)

    def test_no_duplicate_across_occlusion_episode(self):
        zone = BagZoneTracker(FakeLocalizer(
            [disc_mask()] * 3 + [partial_67_mask()] * 6 + [disc_mask()] * 6))
        tracker = PackingTracker(min_inside_frames=3, min_outside_frames=1)
        lock(zone)
        frame_id = 4
        # Count while stable (refresh not due: locked at t=100).
        for moment in (100.5, 100.6, 100.7):
            drive(zone, tracker, frame_id,
                  [Detection(7, "Bottle", self.INSIDE)], moment)
            frame_id += 1
        self.assertEqual(tracker.packed_counts.get("Bottle", 0), 1)
        # Occlusion (calls 4-8 are partials), loss, hold, then recovery
        # (calls 9+ are full): the same track ID never recounts.
        for moment in (110.0, 110.5, 111.0, 111.5, 112.1, 113.0,
                       114.0, 114.1, 114.2, 115.0, 115.5, 116.0):
            events = drive(zone, tracker, frame_id,
                           [Detection(7, "Bottle", self.INSIDE)], moment)
            self.assertEqual(events, [])
            frame_id += 1
        self.assertEqual(zone.status, "stable")
        self.assertEqual(tracker.packed_counts.get("Bottle", 0), 1)
        self.assertEqual(len(tracker.packed_events), 1)


class ProductDiagTest(unittest.TestCase):
    def _tracker(self, paused=False):
        from bag_zone import ZoneTest
        grid = np.zeros((120, 160), np.uint8)
        grid[30:90, 40:120] = 1
        tracker = PackingTracker(
            zone_test=ZoneTest(grid, 0.70, version=1),
            min_inside_frames=3, min_outside_frames=1)
        tracker.zone_version = 1
        tracker.set_packing_paused(paused)
        return tracker

    def test_confirming_streak_reports_overlap_and_blocker(self):
        tracker = self._tracker()
        tracker.update([Detection(7, "Bottle", (300.0, 200.0, 360.0, 280.0))])
        diag = {entry["track_id"]: entry for entry in tracker.track_diagnostics()}
        self.assertAlmostEqual(diag[7]["overlap"], 1.0)
        self.assertEqual(diag[7]["inside"], "1/3")
        self.assertIn("confirming", diag[7]["blocked"])
        self.assertFalse(diag[7]["packed"])

    def test_partial_overlap_and_rim_reported(self):
        tracker = self._tracker()
        tracker.update([Detection(7, "Bottle", (420.0, 200.0, 520.0, 280.0))])
        tracker.update([Detection(8, "Bottle", (430.0, 300.0, 550.0, 420.0))])
        diag = {entry["track_id"]: entry for entry in tracker.track_diagnostics()}
        self.assertLess(diag[7]["overlap"], 0.70)
        self.assertEqual(diag[8]["blocked"], "at footprint rim")

    def test_paused_and_packed_blockers(self):
        tracker = self._tracker(paused=True)
        tracker.update([Detection(7, "Bottle", (300.0, 200.0, 360.0, 280.0))])
        diag = {entry["track_id"]: entry for entry in tracker.track_diagnostics()}
        self.assertIn("paused", diag[7]["blocked"])
        tracker.set_packing_paused(False)
        for _ in range(3):
            tracker.update([Detection(7, "Bottle", (300.0, 200.0, 360.0, 280.0))])
        diag = {entry["track_id"]: entry for entry in tracker.track_diagnostics()}
        self.assertTrue(diag[7]["packed"])
        self.assertEqual(diag[7]["blocked"], "already counted")
        self.assertIn("track_diag", tracker.snapshot())


if __name__ == "__main__":
    unittest.main()
