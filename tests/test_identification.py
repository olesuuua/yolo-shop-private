"""Per-track OCR accumulation and Jev identification without weights or network."""

import time
import unittest

from identification import (IdentConfig, IdentificationService, hamming,
                            normalize, useful_evidence)


class FakeOcr:
    def __init__(self, lines_by_call=None, latency=0.0):
        self.lines_by_call = lines_by_call or []
        self.latency = latency
        self.calls = 0

    def predict(self, jpeg_bytes):
        self.calls += 1
        if self.latency:
            time.sleep(self.latency)
        if self.lines_by_call:
            index = min(self.calls - 1, len(self.lines_by_call) - 1)
            return self.lines_by_call[index]
        return [{"text": "СЕНЕЖСКАЯ", "score": 0.9}]

    def close(self):
        pass


def candidate(choice, confidence=0.8):
    return {"answer": {"choice": choice, "confidence": confidence,
                       "probabilities": {choice: confidence}},
            "status": "candidate", "exact_sku_verified": False}


class IdentificationTests(unittest.TestCase):
    def make_service(self, ocr=None, jev=None, config=None):
        service = IdentificationService(
            ocr_factory=lambda: ocr or FakeOcr(),
            jev_fn=jev or (lambda *a: candidate("senezhskaya-0-5l")),
            config=config)
        service.start()
        self.addCleanup(service.close)
        return service

    def test_catalog_has_two_unique_bottle_products(self):
        service = self.make_service()
        skus = [p["sku"] for p in service.products]
        self.assertEqual(len(skus), 4)
        self.assertEqual(len(set(skus)), 4)
        self.assertIn("aqua-minerale-0-5l", skus)
        self.assertIn("senezhskaya-0-5l", skus)
        required = {"name", "brand", "packaging", "category", "variant", "size",
                    "aliases", "verified_label_text", "identity_clues", "cautions"}
        for product in service.products:
            self.assertTrue(required.issubset(product),
                            f"{product.get('sku')} is missing Jev fields")

    def test_normalize_and_useful_evidence_gate(self):
        self.assertEqual(normalize("  Вода\tпитьевая "), "ВОДАПИТЬЕВАЯ")
        self.assertFalse(useful_evidence([]))
        self.assertFalse(useful_evidence(["2", "EAC"]))
        self.assertTrue(useful_evidence(["AQUA"]))
        self.assertTrue(useful_evidence(["EAC", "PET"]))

    def test_evidence_stays_per_track(self):
        service = self.make_service()
        service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
        service._merge(2, "Bottle", [{"text": "AQUA", "score": 0.9}])
        snapshot = service.snapshot()
        self.assertEqual(snapshot[1]["lines"], 1)
        self.assertEqual(snapshot[2]["lines"], 1)
        self.assertNotEqual(
            list(service.tracks[1].lines), list(service.tracks[2].lines))

    def test_short_noise_lines_are_dropped(self):
        service = self.make_service()
        service._merge(1, "Bottle", [{"text": "2", "score": 0.5},
                                     {"text": "", "score": 0.0},
                                     {"text": "x", "score": 0.4}])
        self.assertEqual(service.snapshot()[1]["lines"], 0)

    def test_jev_outcomes_map_to_status_without_verifying_sku(self):
        calls = []

        def jev(lines, products, hint):
            calls.append((lines, [p["sku"] for p in products], hint))
            return candidate("aqua-minerale-0-5l", 0.7)

        service = self.make_service(jev=jev)
        evidence = service._merge(5, "Bottle", [{"text": "AQUA", "score": 0.9}])
        service._identify(5, evidence)
        snapshot = service.snapshot()[5]
        self.assertEqual(snapshot["status"], "candidate")
        self.assertEqual(snapshot["choice"], "aqua-minerale-0-5l")
        self.assertEqual(snapshot["confidence"], 0.7)
        lines, skus, hint = calls[0]
        self.assertEqual(hint, "Bottle")  # Hint travels, never filters.
        self.assertEqual(set(skus), {"aqua-minerale-0-5l", "senezhskaya-0-5l",
                                       "prostokvashino-2-5-930ml", "domik-v-derevne-2-5-930ml"})
        self.assertTrue(all(line["text"] for line in lines))

    def test_uncertain_and_unknown_stay_unresolved(self):
        outcomes = [
            {"answer": {"choice": "insufficient_evidence", "confidence": 0.9,
                        "probabilities": {"insufficient_evidence": 0.9}},
             "status": "needs_more_evidence", "exact_sku_verified": False},
            {"answer": {"choice": "other_product", "confidence": 0.8,
                        "probabilities": {"other_product": 0.8}},
             "status": "unknown", "exact_sku_verified": False},
        ]
        service = self.make_service(jev=lambda *a: outcomes.pop(0))
        first = service._merge(1, "Bottle", [{"text": "ВОДА", "score": 0.6}])
        service._identify(1, first)
        self.assertEqual(service.snapshot()[1]["status"], "needs_more_evidence")
        self.assertIsNone(service.snapshot()[1]["choice"])
        second = service._merge(2, "Cup", [{"text": "COLA", "score": 0.9}])
        service._identify(2, second)
        self.assertEqual(service.snapshot()[2]["status"], "unknown")

    def test_no_repeat_jev_call_without_new_evidence(self):
        calls = []

        def jev(lines, products, hint):
            calls.append(True)
            return candidate("senezhskaya-0-5l")

        service = self.make_service(jev=jev)
        evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
        service._identify(1, evidence)
        self.assertEqual(len(calls), 1)
        # Same fingerprint must not call again, even past the debounce.
        evidence.last_jev_time -= 1000
        self.assertFalse(service._jev_due(evidence))
        # New evidence re-arms after the debounce.
        service._merge(1, "Bottle", [{"text": "НЕГАЗИРОВАННАЯ", "score": 0.8}])
        evidence.last_jev_time = time.monotonic()
        self.assertFalse(service._jev_due(evidence))  # Debounce still holds.
        evidence.last_jev_time -= 1000
        self.assertTrue(service._jev_due(evidence))

    def test_weak_evidence_never_calls_jev(self):
        service = self.make_service(jev=lambda *a: self.fail("Jev must not be called"))
        evidence = service._merge(1, "Bottle", [{"text": "EAC", "score": 0.8}])
        self.assertFalse(service._jev_due(evidence))
        self.assertEqual(service.snapshot()[1]["status"], "needs_more_evidence")

    def test_jev_failure_keeps_track_unresolved(self):
        def broken(lines, products, hint):
            raise RuntimeError("Jev HTTP 500; no result accepted.")

        service = self.make_service(jev=broken)
        evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
        service._identify(1, evidence)
        self.assertEqual(service.snapshot()[1]["status"], "needs_more_evidence")

    def test_note_counters_and_debug(self):
        service = self.make_service()
        service.note(9, "Bottle", 12.5)
        snapshot = service.snapshot()[9]
        self.assertEqual(snapshot["status"], "needs_more_evidence")
        self.assertEqual(snapshot["submitted"], 0)
        self.assertEqual(snapshot["sharpness"], 12.5)
        debug = service.debug()["tracks"]["9"]
        self.assertEqual(debug["lines"], [])
        self.assertEqual(debug["hint"], "Bottle")

    def test_prune_and_reset(self):
        service = self.make_service()
        service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
        service._merge(2, "Bottle", [{"text": "AQUA", "score": 0.9}])
        # A brief flicker (up to IDENT_ABSENT_FRAMES) preserves evidence.
        service.prune({1})
        service.prune({1})
        self.assertIn(1, service.snapshot())
        self.assertIn(2, service.snapshot())
        # A removed bottle loses its evidence and identity: a reused ID
        # must restart at "need evidence", never inherit the old bottle.
        service.prune({1})
        service.prune({1})
        self.assertNotIn(2, service.snapshot())
        service.close()  # Freeze background work: the reappearing ID must start empty.
        service.submit(2, "Bottle", b"fake-jpeg")
        self.assertEqual(service.snapshot()[2]["status"], "needs_more_evidence")
        self.assertEqual(service.snapshot()[2]["lines"], 0)
        service.reset()
        self.assertEqual(service.snapshot(), {})

    def test_readiness_reports_without_secrets(self):
        service = self.make_service()
        readiness = service.readiness()
        self.assertTrue(readiness["catalog_ok"])
        self.assertEqual(len(readiness["products"]), 4)
        self.assertEqual(readiness["ocr_device"], "cpu")
        self.assertNotIn("TYPESAFE_API_KEY", str(readiness).upper().replace("JEV_KEY_PRESENT", ""))
        self.assertIsInstance(readiness["jev_key_present"], bool)

    def test_end_to_end_submit_ocr_and_identify(self):
        ocr = FakeOcr([[{"text": "СЕНЕЖСКАЯ", "score": 0.9}]])
        service = self.make_service(ocr=ocr)
        self.assertTrue(service.submit(7, "Bottle", b"fake-jpeg"))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            snapshot = service.snapshot()
            if snapshot.get(7, {}).get("status") == "candidate":
                break
            time.sleep(0.1)
        self.assertEqual(service.snapshot()[7]["choice"], "senezhskaya-0-5l")
        self.assertEqual(ocr.calls, 1)

    def test_stale_slot_replaced_by_newer_crop(self):
        service = IdentificationService(
            ocr_factory=lambda: FakeOcr(), jev_fn=lambda *a: candidate("senezhskaya-0-5l"),
            config=IdentConfig(submit_interval_s=0, submit_interval_empty_s=0))
        # No thread started: pure slot mechanics.
        service.submit(1, "Bottle", b"crop-one")
        service.submit(1, "Bottle", b"crop-two")
        with service.slot_lock:
            self.assertEqual(service.slots[1], b"crop-two")
        self.assertEqual(service.snapshot()[1]["submitted"], 2)
        self.assertEqual(service.snapshot()[1]["stale_replaced"], 1)
        service.close()

    def test_round_robin_serves_both_tracks(self):
        ocr = FakeOcr(latency=0.2)
        service = self.make_service(
            ocr=ocr, config=IdentConfig(submit_interval_s=0, submit_interval_empty_s=0))
        service.submit(1, "Bottle", b"crop-one")
        service.submit(2, "Bottle", b"crop-two")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            snap = service.snapshot()
            if snap.get(1, {}).get("ocr_runs", 0) >= 1 and snap.get(2, {}).get("ocr_runs", 0) >= 1:
                break
            time.sleep(0.05)
        self.assertGreaterEqual(service.snapshot()[1]["ocr_runs"], 1)
        self.assertGreaterEqual(service.snapshot()[2]["ocr_runs"], 1)

    def test_duplicate_crop_skips_ocr(self):
        import cv2
        import numpy as np
        frame = np.full((120, 80, 3), 200, dtype=np.uint8)
        cv2.putText(frame, "TEXT", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        ok, jpeg = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        payload = jpeg.tobytes()
        ocr = FakeOcr()
        service = self.make_service(
            ocr=ocr, config=IdentConfig(submit_interval_s=0, submit_interval_empty_s=0))
        service.submit(1, "Bottle", payload)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and ocr.calls < 1:
            time.sleep(0.05)
        self.assertEqual(ocr.calls, 1)
        # Same view again: processed once, then skipped as duplicate.
        service.submit(1, "Bottle", payload)
        time.sleep(1.0)
        self.assertEqual(ocr.calls, 1)
        self.assertEqual(service.snapshot()[1]["dedup_skips"], 1)

    def test_ocr_continues_while_jev_runs(self):
        import threading
        entered = threading.Event()
        release = threading.Event()
        ocr = FakeOcr(latency=0.1)

        def slow_jev(lines, products, hint):
            entered.set()
            release.wait(timeout=10)
            return candidate("senezhskaya-0-5l")

        service = self.make_service(
            ocr=ocr, jev=slow_jev,
            config=IdentConfig(submit_interval_s=0, submit_interval_empty_s=0))
        service.submit(1, "Bottle", b"first-view")
        self.assertTrue(entered.wait(timeout=5))
        # Jev is now blocked; a new view must still be OCR'd, not queued behind it.
        service.tracks[1].last_ocr_hash = None  # Force a visibly new view.
        service.submit(1, "Bottle", b"second-view")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and service.snapshot()[1]["ocr_runs"] < 2:
            time.sleep(0.05)
        release.set()
        self.assertEqual(ocr.calls, 2)
        self.assertGreaterEqual(service.snapshot()[1]["ocr_runs"], 2)

    def test_first_submit_retry_is_faster(self):
        from identification import TrackEvidence
        service = self.make_service(
            config=IdentConfig(submit_interval_s=100, submit_interval_empty_s=0))
        service.submit(1, "Bottle", b"one")
        # Empty track: fast retry path allows an immediate second submit.
        self.assertTrue(service.submit(1, "Bottle", b"two"))
        service.close()
        full = self.make_service(
            config=IdentConfig(submit_interval_s=100, submit_interval_empty_s=100))
        full.tracks[2] = TrackEvidence(hint="Bottle")
        full.tracks[2].ocr_runs = 1
        full.tracks[2].last_submit_time = time.monotonic()
        self.assertFalse(full.submit(2, "Bottle", b"three"))

    def test_hamming_distance(self):
        self.assertEqual(hamming(0b1010, 0b1010), 0)
        self.assertEqual(hamming(0b1010, 0b0101), 4)


if __name__ == "__main__":
    unittest.main()
