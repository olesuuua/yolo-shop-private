"""Ownership investigation fixtures. No production behavior changed here.

Labels:
  OBSERVED  = saved report/crop evidence (see request IDs below).
  CONSTRUCTED = synthetic fake-OCR/Jev sequence demonstrating a mechanism.
  Never present a CONSTRUCTED case as a measured video incident.

Observed anchors (read-only, no network):
  O1 (388:2:31, v1-ocr-cold, W): correct owner+correct text, rejected blurry.
  O2 (463:7:160, v3-warm-a, W): correct owner, neighbor Domik text dominates.
  O3 (580:3:329, v3-warm-b, S): correct owner S + neighbor generic 2,5%.
  O4 (172:2:232, v1-b2 quality-gate, track 2): Aqua+Senezhskaya crop with
     ghost W text (OCR hallucination). Jev disabled in that experiment, so
     unresolved by design; validity=invalid. Attribution of the ghost line
     to a specific bottle is UNCERTAIN.
  O5 V1 track 2 spans physical A,S,W,P; V3a track 2 spans W/D/ambiguous
     (per-request physical_id sets). Numeric reuse/fragmentation observed.
  O6 No wrong *completed* identity in recorded footage (video-comparison
     report: "No wrong completed identity was observed").
"""
import threading
import unittest

from identification import IdentConfig, IdentificationService


def candidate(choice, confidence=0.8):
    return {"answer": {"choice": choice, "confidence": confidence,
                       "probabilities": {choice: confidence}},
            "status": "candidate", "exact_sku_verified": False}


def needs_more():
    return {"answer": {"choice": "insufficient_evidence", "confidence": 0.9,
                       "probabilities": {"insufficient_evidence": 0.9}},
            "status": "needs_more_evidence", "exact_sku_verified": False}


def make_service(jev_fn):
    cfg = IdentConfig(jev_inline=True)
    return IdentificationService(jev_fn=jev_fn, config=cfg)


class OwnershipFixtures(unittest.TestCase):
    # a) correct owner, correct text incl. the distinguishing volume ->
    #    completes (CONSTRUCTED analogue of O1 had the gate accepted it;
    #    O1 itself never reached OCR). Brand text alone must NOT finalize
    #    a sibling volume: the size gate keeps it provisional.
    def test_a_correct_owner_correct_text_completes(self):
        svc = make_service(lambda *a: candidate("saint-spring-0-33l", 0.95))
        ev = svc._merge(1, "Bottle", [{"text": "СВЯТОЙ ИСТОЧНИК", "score": 0.9}],
                        request_id="a1")
        svc._identify(1, ev)
        self.assertFalse(svc.is_complete(1))  # brand alone: provisional only
        self.assertEqual(svc.snapshot()[1]["choice"], "saint-spring-0-33l")
        self.assertEqual(svc.snapshot()[1]["label_sub"],
                         "Likely match · checking label")
        svc._merge(1, "Bottle", [{"text": "ОБЪЁМ 0,33 Л", "score": 0.9}],
                   request_id="a2")
        svc._identify(1, ev)
        self.assertTrue(svc.is_complete(1))
        self.assertEqual(svc.snapshot()[1]["label_sub"], "Recognized")

    # b) correct owner with neighboring text -> stays unresolved (OBSERVED
    #    pattern O2/O3: neighbor/generic text, needs_more, no completion).
    def test_b_correct_owner_neighbor_text_stays_unresolved(self):
        svc = make_service(lambda *a: needs_more())
        ev = svc._merge(7, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}],
                        request_id="580:3:329-like")
        svc._merge(7, "Bottle", [{"text": "2,5%", "score": 0.96}],
                   request_id="neighbor-generic")
        svc._identify(7, ev)
        self.assertFalse(svc.is_complete(7))
        self.assertIsNone(svc.snapshot()[7]["choice"])
        self.assertEqual(len(ev.fingerprint), 2)  # both texts accumulated

    # c) OCR hallucination of brand text alone can NO LONGER finalize a
    #    sibling volume (CONSTRUCTED analogue of O4; O4 itself stayed
    #    unresolved because Jev was disabled in that experiment). The track
    #    stays a provisional likely-match until volume text is visible.
    def test_c_single_hallucination_can_complete_constructed(self):
        svc = make_service(lambda *a: candidate("saint-spring-0-33l", 0.95))
        ev = svc._merge(2, "Bottle", [{"text": "СВЯТОЙ ИСТОЧНИК", "score": 0.6}],
                        request_id="172:2:232-like")
        svc._identify(2, ev)
        self.assertFalse(svc.is_complete(2))  # size gate: brand alone never finalizes
        self.assertEqual(svc.snapshot()[2]["label_sub"],
                         "Likely match · checking label")

    # d) numeric ID switches bottles with NO absence -> mixed evidence
    #    (CONSTRUCTED mechanism for OBSERVED O5 reuse pattern).
    def test_d_same_id_switch_without_absence_mixes(self):
        svc = make_service(lambda *a: needs_more())
        ev = svc._merge(2, "Bottle", [{"text": "AQUA", "score": 0.9}],
                        request_id="bottle-A")
        svc.prune({2})  # still present: absent_frames reset, same object kept
        svc._merge(2, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}],
                   request_id="bottle-S-after-switch")
        norms = set(ev.fingerprint)
        self.assertIn("AQUA", norms)
        self.assertIn("СЕНЕЖСКАЯ", norms)  # mixed across physical bottles

    # d2) completed identity then same-ID replacement without sufficient
    #     absence sticks (CONSTRUCTED; no such completed transfer observed).
    def test_d2_completed_then_replacement_sticks_constructed(self):
        svc = make_service(lambda *a: candidate("aqua-minerale-0-5l", 0.95))
        ev = svc._merge(2, "Bottle", [{"text": "AQUA", "score": 0.9}],
                        request_id="bottle-A")
        svc._identify(2, ev)
        self.assertTrue(svc.is_complete(2))
        from config import IDENT_ABSENT_FRAMES
        self.assertGreater(IDENT_ABSENT_FRAMES, 1)
        svc.prune(set())  # only 1 absent frame: insufficient, object retained
        self.assertIn(2, svc.tracks)
        self.assertIsNone(svc._merge(2, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}],
                                        request_id="bottle-S-new"))
        self.assertTrue(svc.is_complete(2))
        self.assertEqual(svc.snapshot()[2]["choice"], "aqua-minerale-0-5l")

    # e) fragmentation into a new ID starts empty (existing protection).
    def test_e_fragmentation_new_id_starts_empty(self):
        svc = make_service(lambda *a: needs_more())
        svc._merge(2, "Bottle", [{"text": "AQUA", "score": 0.9}], request_id="r1")
        from config import IDENT_ABSENT_FRAMES
        for _ in range(IDENT_ABSENT_FRAMES + 1):
            svc.prune(set())
        self.assertNotIn(2, svc.tracks)
        ev4 = svc._merge(4, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}],
                         request_id="r2")
        self.assertEqual(set(ev4.fingerprint), {"СЕНЕЖСКАЯ"})

    # In-flight result after ownership change is rejected (existing guard).
    def test_inflight_after_ownership_change_protected(self):
        entered, release = threading.Event(), threading.Event()

        def slow(lines, products, hint):
            entered.set()
            release.wait(5)
            return candidate("aqua-minerale-0-5l", 0.95)

        svc = IdentificationService(jev_fn=slow)
        old = svc._merge(7, "bottle", [{"text": "WATER", "score": 0.9}], request_id="a")
        thread = threading.Thread(target=svc._identify, args=(7, old))
        thread.start()
        self.assertTrue(entered.wait(2))
        svc.reset()
        new = svc.touch(7, "new")
        svc.jev_inflight.add(7)
        release.set()
        thread.join(2)
        self.assertEqual(new.result, {"status": "needs_more_evidence"})
        self.assertIn(7, svc.jev_inflight)
        self.assertEqual(svc.jev_stale_drops, 1)

    # Stale expected-object work is rejected (existing guards).
    def test_stale_expected_rejected(self):
        svc = make_service(lambda *a: needs_more())
        svc.submit(7, "bottle", b"one", diagnostic={"request_id": "first"})
        old = svc.tracks[7]
        svc.reset()
        self.assertFalse(svc.submit(7, "bottle", b"late", expected=old))
        svc.touch(7, "new")
        self.assertIsNone(svc._merge(7, "bottle", [{"text": "OLD", "score": 1}],
                                     expected=old))

    # Sufficient absence clears (existing guard).
    def test_sufficient_absence_clears(self):
        svc = make_service(lambda *a: needs_more())
        svc._merge(9, "Bottle", [{"text": "AQUA", "score": 0.9}], request_id="r1")
        from config import IDENT_ABSENT_FRAMES
        for _ in range(IDENT_ABSENT_FRAMES + 1):
            svc.prune(set())
        self.assertNotIn(9, svc.tracks)


if __name__ == "__main__":
    unittest.main()
