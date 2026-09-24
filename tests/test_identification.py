"""Per-track OCR accumulation and Jev identification without weights or network."""

import time
import threading
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
    def test_confident_result_freezes_crop_and_clears_pending_work(self):
        service = IdentificationService(jev_fn=lambda *a: candidate("senezhskaya-0-5l", 0.7))
        self.assertTrue(service.submit(1, "Bottle", b"first", (100, 200)))
        evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
        service._identify(1, evidence)
        self.assertTrue(service.is_complete(1))
        self.assertIsNone(service._take_slot())
        self.assertFalse(service.submit(1, "Bottle", b"second", (200, 300)))
        service.note(1, "Bottle", 99, (200, 300))
        self.assertEqual(evidence.last_crop_jpeg, b"first")
        self.assertEqual(evidence.last_crop_wh, (100, 200))
        self.assertIsNone(service._merge(1, "Bottle", [{"text": "NEW", "score": 1}]))
        service._identify(1, evidence)
        self.assertEqual(service.jev_calls, 1)
        self.assertFalse(service._jev_due(evidence))
        self.assertTrue(service.snapshot()[1]["complete"])
        service.prune({1})
        self.assertTrue(service.is_complete(1))
        from config import IDENT_ABSENT_FRAMES
        for _ in range(IDENT_ABSENT_FRAMES + 1):
            service.prune(set())
        self.assertFalse(service.is_complete(1))
        self.assertTrue(service.submit(1, "Bottle", b"new item"))
        service.reset()
        self.assertTrue(service.submit(1, "Bottle", b"reset item"))

    def test_low_confidence_and_unknown_keep_collecting(self):
        outcomes = [candidate("senezhskaya-0-5l", 0.699),
                    candidate("other_product", 1),
                    candidate("insufficient_evidence", 1)]
        for outcome in outcomes:
            with self.subTest(outcome=outcome):
                service = IdentificationService(jev_fn=lambda *a: outcome)
                evidence = service._merge(1, "Bottle", [{"text": "WATER", "score": 1}])
                service._identify(1, evidence)
                self.assertFalse(service.is_complete(1))
                self.assertTrue(service.submit(1, "Bottle", b"next"))

    def test_ocr_in_flight_cannot_change_completed_identity(self):
        started, release = threading.Event(), threading.Event()
        class BlockingOcr(FakeOcr):
            def predict(self, jpeg):
                started.set()
                release.wait(3)
                return [{"text": "LATE TEXT", "score": 1}]
        service = IdentificationService(ocr_factory=BlockingOcr,
            jev_fn=lambda *a: candidate("senezhskaya-0-5l", 0.95),
            config=IdentConfig(jev_inline=True, dedup=False))
        service.start()
        try:
            service.submit(1, "Bottle", b"crop")
            self.assertTrue(started.wait(2))
            evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 1}])
            service._identify(1, evidence)
            release.set()
            service.close()
            self.assertEqual(evidence.ocr_runs, 0)
            self.assertNotIn("LATETEXT", evidence.lines)
            self.assertTrue(service.is_complete(1))
        finally:
            release.set()
            service.close()

    def make_service(self, ocr=None, jev=None, config=None):
        service = IdentificationService(
            ocr_factory=lambda: ocr or FakeOcr(),
            jev_fn=jev or (lambda *a: candidate("senezhskaya-0-5l")),
            config=config)
        service.start()
        self.addCleanup(service.close)
        return service

    def test_catalog_has_twelve_unique_products(self):
        service = self.make_service()
        skus = [p["sku"] for p in service.products]
        self.assertEqual(len(skus), 12)
        self.assertEqual(len(set(skus)), 12)
        self.assertIn("aqua-minerale-0-5l", skus)
        self.assertIn("senezhskaya-0-5l", skus)
        self.assertIn("saint-spring-0-75l", skus)
        self.assertNotIn("saint-spring-0-33l", skus)
        self.assertIn("stantsiya-molochnaya-kefir-1-0-430g", skus)
        self.assertIn("dobryi-cola-no-sugar-0-5l", skus)
        self.assertIn("red-bull-sugar-free-0-25l", skus)
        self.assertIn("dobryi-kiwi-1l", skus)
        self.assertIn("dobryi-orange-1l", skus)
        self.assertIn("severnaya-dolina-milk-3-2-925ml", skus)
        self.assertIn("selyanochka-5-zlakov-400g", skus)
        required = {"sku", "name", "object_classes", "brand", "category",
                    "variant", "size", "aliases", "verified_label_text"}
        for product in service.products:
            self.assertEqual(set(product), required,
                             f"{product.get('sku')} has unexpected catalog fields")

    def test_jev_config_rejects_invalid_worker_counts(self):
        for value in (0, 9, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                IdentConfig(jev_workers=value)

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
                                       "saint-spring-0-75l", "prostokvashino-2-5-930ml",
                                       "domik-v-derevne-2-5-930ml",
                                       "stantsiya-molochnaya-kefir-1-0-430g",
                                       "dobryi-cola-no-sugar-0-5l",
                                       "red-bull-sugar-free-0-25l",
                                       "dobryi-kiwi-1l",
                                       "dobryi-orange-1l",
                                       "severnaya-dolina-milk-3-2-925ml",
                                       "selyanochka-5-zlakov-400g"})
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
            return candidate("senezhskaya-0-5l", 0.6)

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
        self.assertEqual(len(readiness["products"]), 12)
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
            ocr=ocr, jev=lambda *a: candidate("senezhskaya-0-5l", 0.6),
            config=IdentConfig(submit_interval_s=0, submit_interval_empty_s=0))
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

    def test_different_tracks_are_identified_in_parallel(self):
        both_entered = threading.Event()
        release = threading.Event()
        active = 0
        active_lock = threading.Lock()

        def blocking_jev(lines, products, hint):
            nonlocal active
            with active_lock:
                active += 1
                if active == 2:
                    both_entered.set()
            release.wait(timeout=5)
            return candidate("senezhskaya-0-5l")

        service = self.make_service(
            jev=blocking_jev, config=IdentConfig(jev_workers=2))
        service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
        service._merge(2, "Bottle", [{"text": "AQUА", "score": 0.9}])
        service.jev_wake.set()
        self.assertTrue(both_entered.wait(timeout=5))
        with active_lock:
            self.assertEqual(active, 2)
        release.set()

    def test_one_track_never_has_overlapping_jev_calls(self):
        entered = threading.Event()
        release = threading.Event()
        calls = 0
        calls_lock = threading.Lock()

        def blocking_jev(lines, products, hint):
            nonlocal calls
            with calls_lock:
                calls += 1
            entered.set()
            release.wait(timeout=5)
            return candidate("senezhskaya-0-5l", 0.6)

        service = self.make_service(
            jev=blocking_jev, config=IdentConfig(jev_workers=4, jev_debounce_s=0))
        evidence = service._merge(
            1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
        service.jev_wake.set()
        self.assertTrue(entered.wait(timeout=5))
        service._merge(1, "Bottle", [{"text": "ВОДА", "score": 0.8}])
        service.jev_wake.set()
        time.sleep(0.2)
        with calls_lock:
            self.assertEqual(calls, 1)
        release.set()

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

    def test_remaining_saint_spring_uses_seventy_percent_threshold(self):
        service = IdentificationService(
            jev_fn=lambda *a: candidate("saint-spring-0-75l", 0.70))
        evidence = service._merge(1, "Bottle", [
            {"text": "СВЯТОЙ ИСТОЧНИК", "score": 0.9},
            {"text": "SAINT SPRING STILL", "score": 0.8}])
        service._identify(1, evidence)
        snap = service.snapshot()[1]
        self.assertTrue(service.is_complete(1))
        self.assertEqual(snap["choice"], "saint-spring-0-75l")
        self.assertEqual(snap["label_sub"], "Recognized")
        self.assertIn("0,75", snap["label_main"])
        self.assertFalse(snap["size_supported"])

    def test_dobryi_flavors_are_not_size_siblings(self):
        cases = (
            ("dobryi-orange-1l", "ДОБРЫЙ АПЕЛЬСИН"),
            ("dobryi-cola-no-sugar-0-5l", "ДОБРЫЙ КОЛА БЕЗ САХАРА"),
        )
        for sku, text in cases:
            with self.subTest(sku=sku):
                service = IdentificationService(
                    jev_fn=lambda *a, sku=sku: candidate(sku, 0.70))
                evidence = service._merge(1, "Bottle", [{"text": text, "score": 0.9}])
                service._identify(1, evidence)
                self.assertTrue(service.is_complete(1))
                self.assertFalse(service.snapshot()[1]["size_supported"])
                self.assertFalse(service._choice_size_info(sku)[1])

    def test_future_identical_size_siblings_require_printed_volume(self):
        service = IdentificationService(
            jev_fn=lambda *a: candidate("saint-spring-0-75l", 0.70))
        sibling = service._product_by_sku("saint-spring-0-75l").copy()
        sibling.update(sku="future-saint-spring-0-33l", size="0,33 л")
        service.products.append(sibling)
        self.assertTrue(service._choice_size_info("saint-spring-0-75l")[1])
        evidence = service._merge(1, "Bottle", [{"text": "СВЯТОЙ ИСТОЧНИК", "score": 0.9}])
        service._identify(1, evidence)
        self.assertFalse(service.is_complete(1))
        service._merge(1, "Bottle", [{"text": "ОБЪЁМ 0,75 Л", "score": 0.9}])
        service._identify(1, evidence)
        self.assertTrue(service.is_complete(1))
        self.assertTrue(service.snapshot()[1]["size_supported"])

    def test_unique_products_keep_confidence_only_completion(self):
        # No size siblings: the original confidence rule is untouched.
        service = IdentificationService(
            jev_fn=lambda *a: candidate("senezhskaya-0-5l", 0.7))
        evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
        service._identify(1, evidence)
        self.assertTrue(service.is_complete(1))
        self.assertEqual(service.snapshot()[1]["label_sub"], "Recognized")

    def test_audience_labels_never_expose_sku_or_confidence(self):
        service = IdentificationService(
            jev_fn=lambda *a: candidate("saint-spring-0-75l", 0.65))
        evidence = service._merge(1, "Bottle", [{"text": "СВЯТОЙ ИСТОЧНИК", "score": 0.9}])
        service._identify(1, evidence)
        snap = service.snapshot()[1]
        for text in (snap["label_main"], snap["label_sub"]):
            self.assertNotIn("saint-spring-0-75l", text)
            self.assertNotIn("0.65", text)
            self.assertNotIn("Candidate", text)
        from vision import ident_label
        main, sub = ident_label(snap)
        self.assertNotIn("saint-spring", main + sub)
        self.assertNotIn("Candidate", main + sub)


if __name__ == "__main__":
    unittest.main()
