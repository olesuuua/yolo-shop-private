"""Live-instability regressions: uncertain refresh, miss-tolerant acquisition.

Covers the failure modes behind "bag lost with no scene change, no
recovery" and "bottle in view blocks acquisition", plus the bounded
no-video diagnostics. Synthetic masks only; no model weights needed.
"""

import unittest

import numpy as np

from bag_zone import BagZoneTracker, NoZone


def disc_mask(cx=320, cy=240, radius=120):
    mask = np.zeros((480, 640), bool)
    ys, xs = np.mgrid[0:480, 0:640]
    mask[np.hypot(xs - cx, ys - cy) <= radius] = True
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


class UncertainRefreshTest(unittest.TestCase):
    def test_rejection_is_uncertain_not_gone(self):
        relocations = []
        base = disc_mask()
        notch = base.copy()
        notch[205:270, 425:460] = False  # bottle-width cut in the rim
        zone = BagZoneTracker(FakeLocalizer([base] * 3 + [notch]),
                              on_relocation=lambda: relocations.append(1))
        lock(zone)
        relocations.clear()  # the initial lock itself relocates; ignore it
        contour = zone.snapshot()["contour"]
        zone.update(blank_frame(), 4, now=110.0)
        # Outline retained for display, counting paused, streaks kept: a
        # single rejected refresh wipes nothing and proves no disappearance.
        self.assertEqual(zone.status, "grace")
        self.assertTrue(zone.packing_paused)
        self.assertEqual(zone.snapshot()["contour"], contour)
        self.assertIsNotNone(zone.last_locked_grid)
        self.assertEqual(relocations, [])
        self.assertEqual(zone.snapshot()["last_refresh_rejection"], "shape")
        diag = zone.snapshot()["bag_diag"]
        self.assertIn("shape", diag["reason"])
        self.assertEqual(diag["last_refresh"]["reason"], "shape")
        self.assertIn("missing_frac", diag["last_refresh"])

    def test_transient_jitter_recovers_without_relocation(self):
        relocations = []
        base = disc_mask()
        notch = base.copy()
        notch[205:270, 425:460] = False
        zone = BagZoneTracker(FakeLocalizer([base] * 3 + [notch, base]),
                              on_relocation=lambda: relocations.append(1))
        lock(zone)
        relocations.clear()  # the initial lock itself relocates; ignore it
        contour = zone.snapshot()["contour"]
        zone.update(blank_frame(), 4, now=110.0)  # rejected -> grace
        self.assertEqual(zone.status, "grace")
        zone.update(blank_frame(), 5, now=110.5)  # credible again
        self.assertEqual(zone.status, "stable")
        self.assertEqual(zone.snapshot()["contour"], contour)
        self.assertEqual(relocations, [])  # identical outline: no wipe
        kinds = [t["from"] + "->" + t["to"]
                 for t in zone.snapshot()["bag_diag"]["transitions"]]
        self.assertIn("stable->grace", kinds)
        self.assertIn("grace->stable", kinds)

    def test_consistent_new_outline_adopts_as_moved_bag(self):
        relocations = []
        base = disc_mask()
        moved = disc_mask(200, 150, 100)  # different place and size
        zone = BagZoneTracker(
            FakeLocalizer([base] * 3 + [moved] * 4),
            on_relocation=lambda: relocations.append(1))
        lock(zone)
        relocations.clear()  # the initial lock itself relocates; ignore it
        old = zone.snapshot()["contour"]
        zone.update(blank_frame(), 4, now=110.0)  # rejected -> grace
        self.assertEqual(zone.status, "grace")
        zone.update(blank_frame(), 5, now=110.5)  # 2nd consistent reject
        self.assertEqual(zone.status, "grace")
        zone.update(blank_frame(), 6, now=111.0)  # 3rd: adopt, no loss wait
        self.assertEqual(zone.status, "stable")
        self.assertNotEqual(zone.snapshot()["contour"], old)
        self.assertEqual(len(relocations), 1)  # one wipe, at the real move


class AcquisitionToleranceTest(unittest.TestCase):
    def test_two_misses_preserve_streak_then_lock(self):
        base = disc_mask()
        zone = BagZoneTracker(FakeLocalizer([base] * 2 + [None] * 2 + [base] * 3))
        zone.update(blank_frame(), 1, now=100.0)
        zone.update(blank_frame(), 2, now=100.1)
        self.assertEqual(zone.consecutive, 2)
        zone.update(blank_frame(), 3, now=100.2)  # miss tolerated
        zone.update(blank_frame(), 4, now=100.3)  # miss tolerated
        self.assertEqual(zone.status, "locating")
        self.assertEqual(zone.consecutive, 2)
        zone.update(blank_frame(), 5, now=100.4)
        zone.update(blank_frame(), 6, now=100.5)
        zone.update(blank_frame(), 7, now=100.6)
        self.assertEqual(zone.status, "stable")

    def test_three_misses_restart_acquisition(self):
        base = disc_mask()
        zone = BagZoneTracker(FakeLocalizer([base] * 2 + [None] * 3 + [base] * 5))
        zone.update(blank_frame(), 1, now=100.0)
        zone.update(blank_frame(), 2, now=100.1)
        for frame_id in (3, 4, 5):
            zone.update(blank_frame(), frame_id, now=100.0 + frame_id * 0.1)
        self.assertEqual(zone.consecutive, 0)  # bounded: streak restarted
        self.assertEqual(zone.status, "locating")
        for frame_id in (6, 7, 8):
            zone.update(blank_frame(), frame_id, now=101.0 + frame_id * 0.1)
        self.assertEqual(zone.status, "stable")

    def test_inconsistent_shapes_still_restart_streak(self):
        base = disc_mask()
        other = disc_mask(150, 120, 80)
        zone = BagZoneTracker(FakeLocalizer([base, base, other, base, base, base]))
        zone.update(blank_frame(), 1, now=100.0)
        zone.update(blank_frame(), 2, now=100.1)
        zone.update(blank_frame(), 3, now=100.2)  # different shape: restart
        self.assertEqual(zone.consecutive, 1)
        self.assertEqual(zone.status, "locating")


class DiagnosticsTest(unittest.TestCase):
    def test_records_distinguish_absence_from_rejection(self):
        base = disc_mask()
        notch = base.copy()
        notch[205:270, 425:460] = False
        zone = BagZoneTracker(FakeLocalizer([base] * 3 + [None, notch, base],
                                             ))
        lock(zone)
        zone.update(blank_frame(), 10, now=110.0)  # no candidate
        zone.update(blank_frame(), 11, now=110.5)  # rejected shape
        recent = zone.snapshot()["bag_diag"]["recent"]
        outcomes = [r["outcome"] for r in recent[-2:]]
        self.assertEqual(outcomes, ["no_candidate", "refresh_rejected"])
        rejected = recent[-1]
        self.assertIn("shape", rejected["reason"])
        self.assertIn("same", rejected["reason"])
        self.assertIn("conf", rejected)
        self.assertIn("raw_frac", rejected)
        self.assertIn("filled_frac", rejected)

    def test_acquisition_streak_and_iou_recorded(self):
        base = disc_mask()
        zone = BagZoneTracker(FakeLocalizer([base] * 2))
        zone.update(blank_frame(), 1, now=100.0)
        zone.update(blank_frame(), 2, now=100.1)
        diag = zone.snapshot()["bag_diag"]
        self.assertEqual(diag["streak"], "2/3")
        self.assertIsNotNone(diag["streak_iou"])
        self.assertGreater(diag["streak_iou"], 0.9)
        self.assertIn("acquiring", diag["reason"])

    def test_reset_preserves_finished_log(self):
        base = disc_mask()
        zone = BagZoneTracker(FakeLocalizer([base] * 4))
        lock(zone)
        zone.update(blank_frame(), 4, now=110.0)
        self.assertTrue(zone.snapshot()["bag_diag"]["recent"])
        zone.reset()
        snapshot = zone.snapshot()
        self.assertEqual(snapshot["status"], "locating")
        self.assertIsInstance(snapshot["bag_diag"]["last_log"], dict)
        self.assertTrue(snapshot["bag_diag"]["last_log"]["records"])
        # A fresh session starts recording anew, keeping the old log aside.
        self.assertEqual(snapshot["bag_diag"]["recent"], [])
        zone.update(blank_frame(), 5, now=120.0)
        self.assertTrue(zone.snapshot()["bag_diag"]["recent"])
        self.assertIsNotNone(zone.snapshot()["bag_diag"]["last_log"])


class NoCandidateDebugTest(unittest.TestCase):
    def _stub_result(self, masks, scores):
        class Tensor:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        class Masks:
            def __init__(self, items):
                self.data = [Tensor(m) for m in items]

            def __len__(self):
                return len(self.data)

        class Boxes:
            def __init__(self, values):
                self.conf = Tensor(np.array(values, dtype=np.float32))

        class Result:
            def __init__(self):
                self.masks = Masks(masks) if masks is not None else None
                self.boxes = Boxes(scores) if scores is not None else None

        class Model:
            def predict(self, frame, **kwargs):
                return [Result()]

        return Model()

    def test_masks_below_min_frac_report_sizes(self):
        from bag_zone import BagLocalizer

        small = np.zeros((480, 640), np.float32)
        small[0:100, 0:100] = 1.0  # ~3% of the frame
        model = self._stub_result([small, small], [0.30, 0.25])
        localizer = BagLocalizer(model, conf=0.10, min_frac=0.08)
        self.assertIsNone(localizer.localize(np.zeros((480, 640, 3), np.uint8)))
        self.assertEqual(localizer.last_debug["n_masks"], 2)
        self.assertLess(localizer.last_debug["top_frac"], 0.08)
        self.assertAlmostEqual(localizer.last_debug["top_conf"], 0.30)

    def test_no_masks_reports_blind(self):
        from bag_zone import BagLocalizer

        model = self._stub_result(None, None)
        localizer = BagLocalizer(model, conf=0.10, min_frac=0.08)
        self.assertIsNone(localizer.localize(np.zeros((480, 640, 3), np.uint8)))
        self.assertEqual(localizer.last_debug["n_masks"], 0)

    def test_tracker_records_sub_reason_without_debug_attr(self):
        # Plain fake localizers (no last_debug) must not crash the update.
        zone = BagZoneTracker(FakeLocalizer([None]))
        zone.update(blank_frame(), 1, now=100.0)
        recent = zone.snapshot()["bag_diag"]["recent"]
        self.assertEqual(recent[-1]["outcome"], "no_candidate")
        self.assertIn("no proposals", recent[-1]["reason"])

    def test_tracker_survives_debug_with_masks_but_no_usable_bag(self):
        # Regression: the no_candidate detail path once read min_frac off
        # the tracker and crashed every inference, freezing the pipeline
        # with an empty log. A localizer reporting small masks must record
        # "masks too small" and keep acquiring.
        class DebugLocalizer:
            conf = 0.10
            min_frac = 0.08

            def __init__(self):
                self.last_debug = {"n_masks": 2, "top_conf": 0.30,
                                   "top_frac": 0.05}

            def localize(self, frame):
                return None

        zone = BagZoneTracker(DebugLocalizer())
        zone.update(blank_frame(), 1, now=100.0)
        zone.update(blank_frame(), 2, now=100.1)
        recent = zone.snapshot()["bag_diag"]["recent"]
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[-1]["outcome"], "no_candidate")
        self.assertIn("masks too small", recent[-1]["reason"])
        self.assertIn("bag_diag", zone.snapshot())

    def test_lost_transition_carries_frame_id(self):
        base = disc_mask()
        zone = BagZoneTracker(FakeLocalizer([base] * 3 + [None] * 4))
        for frame_id in (1, 2, 3):
            zone.update(blank_frame(), frame_id, now=100.0)
        zone.update(blank_frame(), 4, now=110.0)  # -> grace
        zone.update(blank_frame(), 5, now=111.0)
        zone.update(blank_frame(), 6, now=112.1)  # -> lost
        self.assertEqual(zone.status, "lost")
        lost = [t for t in zone.snapshot()["bag_diag"]["transitions"]
                if t["to"] == "lost"]
        self.assertTrue(lost)
        self.assertEqual(lost[-1]["frame"], 6)


class MinFracCalibrationTest(unittest.TestCase):
    """BAG_MIN_FRAC admits the measured live blue bag (frac ~0.076)."""

    def _localizer_with_frac(self, frac, conf=0.30):
        from bag_zone import BagLocalizer
        from config import BAG_MIN_FRAC

        mask = np.zeros((480, 640), np.float32)
        side = int((frac * 480 * 640) ** 0.5)
        mask[0:side, 0:side] = 1.0

        class Tensor:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        class Masks:
            def __init__(self, items):
                self.data = [Tensor(m) for m in items]

            def __len__(self):
                return len(self.data)

        class Result:
            def __init__(self):
                self.masks = Masks([mask])
                self.boxes = type("B", (), {
                    "conf": Tensor(np.array([conf], dtype=np.float32))})()

        class Model:
            def predict(self, frame, **kwargs):
                return [Result()]

        return BagLocalizer(Model(), conf=0.10, min_frac=BAG_MIN_FRAC)

    def test_live_blue_bag_frac_accepted(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        found = self._localizer_with_frac(0.076).localize(frame)
        self.assertIsNotNone(found)
        self.assertGreaterEqual(found["conf"], 0.10)

    def test_small_fragment_still_rejected(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        localizer = self._localizer_with_frac(0.04)
        self.assertIsNone(localizer.localize(frame))
        self.assertLess(localizer.last_debug["top_frac"], 0.05)


if __name__ == "__main__":
    unittest.main()
