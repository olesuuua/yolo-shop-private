import unittest

from config import FOOD_CLASSES, MIN_INSIDE_FRAMES, MIN_OUTSIDE_FRAMES
from tracking import (
    Detection, PackingTracker, bbox_center,
    intersection_area, intersection_ratio, is_inside_bag,
)

PRIMARY_CLASS = sorted(FOOD_CLASSES)[0]
ROI = (100, 100, 200, 200)
OUTSIDE = (0, 0, 40, 40)
INSIDE = (120, 120, 160, 160)


def food(track_id=7, bbox=OUTSIDE, class_name=PRIMARY_CLASS):
    return Detection(track_id, class_name, bbox)


class GeometryTests(unittest.TestCase):
    def test_completely_outside(self):
        self.assertFalse(is_inside_bag(OUTSIDE, ROI))
        self.assertEqual(intersection_area(OUTSIDE, ROI), 0)

    def test_center_inside_even_with_small_overlap(self):
        bbox = (-100, -100, 400, 400)
        self.assertEqual(bbox_center(bbox), (150, 150))
        self.assertLess(intersection_ratio(bbox, ROI), 0.3)
        self.assertTrue(is_inside_bag(bbox, ROI))

    def test_overlap_when_center_is_outside(self):
        bbox = (40, 110, 140, 150)
        self.assertLess(bbox_center(bbox)[0], ROI[0])
        self.assertAlmostEqual(intersection_ratio(bbox, ROI), 0.4)
        self.assertTrue(is_inside_bag(bbox, ROI))

    def test_exact_overlap_threshold(self):
        self.assertTrue(is_inside_bag((30, 110, 130, 150), ROI))

    def test_tiny_overlap(self):
        self.assertFalse(is_inside_bag((0, 110, 101, 150), ROI))

    def test_center_on_boundary(self):
        self.assertTrue(is_inside_bag((80, 120, 120, 160), ROI, 0.99))

    def test_invalid_boxes(self):
        for bbox in [(120, 120, 120, 150), (150, 150, 110, 110), (float("nan"), 0, 160, 160)]:
            with self.subTest(bbox=bbox):
                self.assertFalse(is_inside_bag(bbox, ROI))


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.tracker = PackingTracker(roi=ROI, min_inside_frames=2, min_outside_frames=1, track_ttl_frames=3)

    def pack(self, track_id=7):
        self.tracker.update([food(track_id)])
        self.assertEqual(self.tracker.update([food(track_id, INSIDE)]), [])
        return self.tracker.update([food(track_id, INSIDE)])

    def test_transition_registers_one_event_after_confirmation(self):
        events = self.pack()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["track_id"], 7)
        self.assertEqual(events[0]["event"], "packed")
        self.assertTrue(events[0]["timestamp"].endswith("+00:00"))
        self.assertEqual(self.tracker.packed_counts, {PRIMARY_CLASS: 1})

    def test_remaining_inside_and_reentry_never_duplicate(self):
        self.pack()
        for bbox in [INSIDE] * 30 + [OUTSIDE] + [INSIDE] * 5:
            self.assertEqual(self.tracker.update([food(bbox=bbox)]), [])
        self.assertEqual(len(self.tracker.packed_events), 1)

    def test_two_ids_same_class(self):
        for bbox in [OUTSIDE, INSIDE, INSIDE]:
            self.tracker.update([food(7, bbox), food(8, bbox)])
        self.assertEqual(self.tracker.packed_counts, {PRIMARY_CLASS: 2})
        self.assertEqual([e["track_id"] for e in self.tracker.packed_events], [7, 8])

    def test_first_seen_inside_requires_observed_outside(self):
        for _ in range(10):
            self.tracker.update([food(bbox=INSIDE)])
        self.assertEqual(self.tracker.packed_counts, {})
        self.pack()
        self.assertEqual(self.tracker.packed_counts, {PRIMARY_CLASS: 1})

    def test_boundary_jitter_restarts_confirmation(self):
        for bbox in [OUTSIDE, INSIDE, OUTSIDE, INSIDE]:
            self.assertEqual(self.tracker.update([food(bbox=bbox)]), [])
        self.assertEqual(len(self.tracker.update([food(bbox=INSIDE)])), 1)

    def test_missing_frame_breaks_consecutive_confirmation(self):
        # Vanished while clearly outside: no watch starts, the gap breaks.
        self.tracker.update([food()])
        self.tracker.update([food()])
        self.tracker.update([])
        self.assertEqual(self.tracker.update([food(bbox=INSIDE)]), [])
        self.assertEqual(len(self.tracker.update([food(bbox=INSIDE)])), 1)

    def test_hidden_at_boundary_reappearing_inside_completes(self):
        # Hand hides the item mid-insertion: it still counts on return.
        self.tracker.update([food()])
        self.tracker.update([food(bbox=INSIDE)])
        self.tracker.update([])
        self.assertEqual(len(self.tracker.update([food(bbox=INSIDE)])), 1)

    def test_hidden_at_boundary_reappearing_outside_cancels(self):
        self.tracker.update([food()])
        self.tracker.update([food(bbox=INSIDE)])
        self.tracker.update([])
        self.assertEqual(self.tracker.update([food()]), [])
        # No phantom watch left behind: a later genuine transfer counts once.
        self.assertEqual(self.tracker.update([food(bbox=INSIDE)]), [])
        self.assertEqual(len(self.tracker.update([food(bbox=INSIDE)])), 1)

    def test_hidden_at_boundary_expiry_counts_once(self):
        tracker = PackingTracker(roi=ROI, min_inside_frames=2,
                                 min_outside_frames=1, track_ttl_frames=60,
                                 pending_pack_frames=4)
        tracker.update([food()])
        tracker.update([food(bbox=INSIDE)])
        events = []
        for _ in range(3):
            events += tracker.update([])
        self.assertEqual(events, [])
        self.assertEqual(tracker.update([]), [])  # still inside the wait
        events += tracker.update([])  # ~2 s elapsed: counts once
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["track_id"], 7)
        # Expiry consumed the watch: nothing more follows.
        for _ in range(3):
            self.assertEqual(tracker.update([]), [])

    def test_disappearance_while_clearly_outside_starts_no_watch(self):
        tracker = PackingTracker(roi=ROI, min_inside_frames=2,
                                 min_outside_frames=1, track_ttl_frames=60,
                                 pending_pack_frames=4)
        tracker.update([food()])
        for _ in range(6):
            self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.packed_counts, {})

    def test_new_id_inside_next_to_pending_counts_once(self):
        tracker = PackingTracker(roi=ROI, min_inside_frames=2,
                                 min_outside_frames=1, track_ttl_frames=60,
                                 pending_pack_frames=4)
        tracker.update([food(7)])
        tracker.update([food(7, INSIDE)])
        tracker.update([])  # hidden at the boundary: watch starts
        nearby = (130, 130, 170, 170)
        events = tracker.update([food(9, nearby)])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["track_id"], 9)
        self.assertEqual(tracker.packed_counts, {PRIMARY_CLASS: 1})

    def test_behind_bag_limitation_documents_false_count(self):
        # A bottle slid BEHIND an upright bag vanishes at the bag's rim
        # exactly like a hand-hidden insertion: reliably outside, last seen
        # not clearly outside, gone for the whole wait. The camera cannot
        # distinguish the two, so this counts (falsely). Honest limitation.
        tracker = PackingTracker(roi=ROI, min_inside_frames=3,
                                 min_outside_frames=3, track_ttl_frames=60,
                                 pending_pack_frames=4)
        for _ in range(3):
            tracker.update([food(7)])
        tracker.update([food(7, INSIDE)])  # reaches behind the bag rim
        events = []
        for _ in range(4):
            events += tracker.update([])
        self.assertEqual(events, [])
        events += tracker.update([])  # wait elapsed: counts (falsely)
        self.assertEqual(len(events), 1)
        self.assertEqual(tracker.packed_counts, {PRIMARY_CLASS: 1})

    def test_disappearance_has_no_event_and_prunes_stale_state(self):
        self.pack()
        for _ in range(4):
            self.assertEqual(self.tracker.update([]), [])
        self.assertNotIn(7, self.tracker.tracks)
        self.assertEqual(self.tracker.packed_counts, {PRIMARY_CLASS: 1})
        # The same packed ID stays deduplicated even after geometry expires.
        for bbox in [OUTSIDE, INSIDE, INSIDE]:
            self.assertEqual(self.tracker.update([food(bbox=bbox)]), [])

    def test_expired_unpacked_track_must_be_seen_outside_again(self):
        self.tracker.update([food()])
        for _ in range(4):
            self.tracker.update([])
        for _ in range(3):
            self.tracker.update([food(bbox=INSIDE)])
        self.assertEqual(self.tracker.packed_counts, {})

    def test_non_food_classes_never_enter_state(self):
        for bbox in [OUTSIDE, INSIDE, INSIDE]:
            self.tracker.update([food(bbox=bbox, class_name="Person")])
        self.assertEqual(self.tracker.tracks, {})
        self.assertEqual(self.tracker.packed_events, [])

    def test_all_enabled_classes_use_the_same_transfer_confirmation_and_deduplication(self):
        for class_name in FOOD_CLASSES:
            with self.subTest(class_name=class_name):
                self.tracker.reset()
                for bbox, expected_events in [(OUTSIDE, 0), (INSIDE, 0), (INSIDE, 1), (INSIDE, 0)]:
                    events = self.tracker.update([food(bbox=bbox, class_name=class_name)])
                    self.assertEqual(len(events), expected_events)
                self.assertEqual(self.tracker.packed_counts, {class_name: 1})

    def test_reset_clears_all_packing_state(self):
        self.pack()
        self.tracker.reset()
        self.assertEqual(self.tracker.tracks, {})
        self.assertEqual(self.tracker.packed_ids, set())
        self.assertEqual(self.tracker.packed_events, [])
        self.assertEqual(self.tracker.snapshot()["packed_total"], 0)
        self.pack()
        self.assertEqual(self.tracker.packed_counts, {PRIMARY_CLASS: 1})

    def test_duplicate_detection_does_not_fake_two_frames(self):
        self.tracker.update([food()])
        self.assertEqual(self.tracker.update([food(bbox=INSIDE)] * 2), [])
        self.assertEqual(len(self.tracker.update([food(bbox=INSIDE)])), 1)

    def test_configurable_confirmation_length(self):
        self.tracker = PackingTracker(roi=ROI, min_inside_frames=3, min_outside_frames=1)
        for bbox in [OUTSIDE, INSIDE, INSIDE]:
            self.assertEqual(self.tracker.update([food(bbox=bbox)]), [])
        self.assertEqual(len(self.tracker.update([food(bbox=INSIDE)])), 1)


class StableTransferTests(unittest.TestCase):
    def test_new_id_inside_with_one_frame_outside_jitter_does_not_count(self):
        tracker = PackingTracker(roi=ROI)
        for bbox in [INSIDE, OUTSIDE] + [INSIDE] * (MIN_INSIDE_FRAMES + 2):
            self.assertEqual(tracker.update([food(bbox=bbox)]), [])
        self.assertEqual(tracker.snapshot()["packed_total"], 0)

    def test_gaps_do_not_accumulate_outside_confirmation(self):
        tracker = PackingTracker(roi=ROI)
        for _ in range(MIN_OUTSIDE_FRAMES + 1):
            tracker.update([food(bbox=OUTSIDE)])
            tracker.update([])
        for _ in range(MIN_INSIDE_FRAMES):
            self.assertEqual(tracker.update([food(bbox=INSIDE)]), [])

    def test_two_products_each_count_once_after_stable_transfer(self):
        tracker = PackingTracker(roi=ROI)
        for bbox in [OUTSIDE] * MIN_OUTSIDE_FRAMES + [INSIDE] * MIN_INSIDE_FRAMES:
            tracker.update([food(7, bbox), food(8, bbox)])
        self.assertEqual(tracker.snapshot()["packed_total"], 2)
        for bbox in [OUTSIDE] * MIN_OUTSIDE_FRAMES + [INSIDE] * MIN_INSIDE_FRAMES:
            self.assertEqual(tracker.update([food(7, bbox), food(8, bbox)]), [])
        self.assertEqual(tracker.snapshot()["packed_total"], 2)


if __name__ == "__main__":
    unittest.main()
