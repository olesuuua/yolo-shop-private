"""Jev failure-recovery: fake Jev, controllable clock, sync events. No network/models."""
import threading
import time as _real_time
import unittest
from unittest import mock

import identification as ident
from identification import IdentConfig, IdentificationService
from jev_catalog import JevRetryableError, JevTerminalError


def candidate(choice="senezhskaya-0-5l", confidence=0.8):
    return {"answer": {"choice": choice, "confidence": confidence,
                       "probabilities": {choice: confidence}},
            "status": "candidate", "exact_sku_verified": False}


class Clock:
    def __init__(self, start=1000.0):
        self.now = start
    def monotonic(self):
        return self.now
    def advance(self, seconds):
        self.now += seconds


class RecoveryTests(unittest.TestCase):
    def make_service(self, jev, max_attempts=40, debounce=12.0):
        cfg = IdentConfig(jev_inline=True, jev_max_attempts=max_attempts,
                          jev_debounce_s=debounce)
        return IdentificationService(jev_fn=jev, config=cfg)

    def test_transient_failure_recovers_with_unchanged_ocr(self):
        clock = Clock()
        calls = []
        def flaky(lines, products, hint):
            calls.append(list(lines))
            if len(calls) == 1:
                raise JevRetryableError("Jev HTTP 503; no result accepted.")
            return candidate("senezhskaya-0-5l", 0.6)
        service = self.make_service(flaky)
        with mock.patch.object(ident.time, "monotonic", clock.monotonic):
            evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}],
                                      request_id="r1")
            fp = evidence.fingerprint
            service._identify(1, evidence)
            # Failure: no success-debounce stamp, backoff scheduled at +1s.
            self.assertEqual(service.jev_attempts, 1)
            self.assertEqual(service.jev_failures_total, 1)
            self.assertEqual(evidence.jev_failures, 1)
            self.assertIsNone(evidence.last_jev_fingerprint)
            self.assertFalse(evidence.jev_exhausted)
            self.assertEqual(evidence.jev_next_retry_delay, 1.0)
            self.assertFalse(service._jev_due(evidence))
            clock.advance(0.9)
            self.assertFalse(service._jev_due(evidence))
            clock.advance(0.2)  # t+1.1s nominal
            self.assertTrue(service._jev_due(evidence))
            # Unchanged evidence recovers without new OCR.
            self.assertEqual(evidence.fingerprint, fp)
            service._identify(1, evidence)
            self.assertEqual(service.jev_attempts, 2)
            self.assertEqual(service.jev_successes, 1)
            self.assertEqual(service.jev_retries, 1)
            self.assertEqual(evidence.jev_failures, 0)
            self.assertEqual(evidence.last_jev_fingerprint, fp)
            self.assertEqual(evidence.jev_last_attempt_fingerprint, fp)
            self.assertEqual(evidence.jev_last_attempt_source_ids, ["r1"])
            # Successful unchanged evidence never repeats.
            clock.advance(1000)
            self.assertFalse(service._jev_due(evidence))

    def test_retry_uses_newest_evidence_but_records_dispatched(self):
        clock = Clock()
        seen = []
        def ok(lines, products, hint):
            seen.append([l["text"] for l in lines])
            return candidate("senezhskaya-0-5l", 0.6)
        service = self.make_service(
            (lambda lines, products, hint: (_ for _ in ()).throw(
                JevRetryableError("boom")) if not seen else ok(lines, products, hint)))
        # Simpler: fail once then succeed, with new OCR during backoff.
        attempts = []
        def jev(lines, products, hint):
            attempts.append([l["text"] for l in lines])
            if len(attempts) == 1:
                raise JevRetryableError("transient")
            return candidate("senezhskaya-0-5l", 0.6)
        service.jev_fn = jev
        with mock.patch.object(ident.time, "monotonic", clock.monotonic):
            evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}],
                                      request_id="r1")
            service._identify(1, evidence)
            self.assertFalse(service._jev_due(evidence))
            # New OCR during backoff.
            service._merge(1, "Bottle", [{"text": "НЕГАЗИРОВАННАЯ", "score": 0.8}],
                           request_id="r2")
            new_fp = evidence.fingerprint
            clock.advance(1.1)
            self.assertTrue(service._jev_due(evidence))
            service._identify(1, evidence)
            # Retry sent the newest evidence.
            self.assertEqual(evidence.jev_last_attempt_fingerprint, new_fp)
            self.assertEqual(sorted(evidence.jev_last_attempt_source_ids), ["r1", "r2"])
            self.assertEqual(len(attempts), 2)
            # Second dispatch contained both lines (newest).
            self.assertIn("СЕНЕЖСКАЯ", attempts[1])
            self.assertIn("НЕГАЗИРОВАННАЯ", attempts[1])

    def test_changed_ocr_during_inflight_uses_old_then_rearms(self):
        clock = Clock()
        entered, release = threading.Event(), threading.Event()
        dispatched = []
        def slow(lines, products, hint):
            dispatched.append([l["text"] for l in lines])
            entered.set()
            release.wait(5)
            return candidate("senezhskaya-0-5l", 0.6)
        service = self.make_service(slow)
        with mock.patch.object(ident.time, "monotonic", clock.monotonic):
            evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
            old_fp = evidence.fingerprint
            thread = threading.Thread(target=service._identify, args=(1, evidence))
            thread.start()
            self.assertTrue(entered.wait(2))
            # OCR adds evidence while HTTP is in flight.
            service._merge(1, "Bottle", [{"text": "ВОДА", "score": 0.8}])
            new_fp = evidence.fingerprint
            self.assertNotEqual(old_fp, new_fp)
            release.set()
            thread.join(2)
            # Recorded exactly what was dispatched (old), success stamp is old,
            # so newer evidence can trigger a later call after debounce.
            self.assertEqual(evidence.jev_last_attempt_fingerprint, old_fp)
            self.assertEqual(evidence.last_jev_fingerprint, old_fp)
            self.assertFalse(service._jev_due(evidence))  # debounce holds
            clock.advance(12.1)
            self.assertTrue(service._jev_due(evidence))

    def test_retry_backoff_delays_and_retry_after(self):
        clock = Clock()
        def always_fail(lines, products, hint):
            raise JevRetryableError("down")
        service = self.make_service(always_fail)
        with mock.patch.object(ident.time, "monotonic", clock.monotonic):
            evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
            service._identify(1, evidence)
            self.assertEqual(evidence.jev_next_retry_delay, 1.0)
            clock.advance(1.1)
            service._identify(1, evidence)
            self.assertEqual(evidence.jev_failures, 2)
            self.assertEqual(evidence.jev_next_retry_delay, 3.0)
            t = clock.now
            # New fingerprints must not replenish: still in backoff.
            service._merge(1, "Bottle", [{"text": "ВОДА", "score": 0.8}])
            self.assertFalse(service._jev_due(evidence))
            clock.advance(2.9)
            self.assertFalse(service._jev_due(evidence))
            clock.advance(0.2)
            self.assertTrue(service._jev_due(evidence))
            service._identify(1, evidence)  # third attempt fails -> exhausted
            self.assertTrue(evidence.jev_exhausted)
            self.assertFalse(service._jev_due(evidence))
            clock.advance(100)
            self.assertFalse(service._jev_due(evidence))
            # Even newer evidence stays blocked.
            service._merge(1, "Bottle", [{"text": "AQUA", "score": 0.9}])
            self.assertFalse(service._jev_due(evidence))
            self.assertEqual(service.jev_attempts, 3)
            stats = service.jev_stats()
            self.assertEqual(stats["attempts"], 3)
            self.assertEqual(stats["failures"], 3)
            self.assertEqual(stats["retries"], 2)
        # Retry-After extends the nominal delay.
        clock2 = Clock()
        def fail_with_retry_after(lines, products, hint):
            raise JevRetryableError("rate limited", retry_after=5.0)
        service2 = self.make_service(fail_with_retry_after)
        with mock.patch.object(ident.time, "monotonic", clock2.monotonic):
            evidence2 = service2._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
            service2._identify(1, evidence2)
            self.assertEqual(evidence2.jev_next_retry_delay, 5.0)
            clock2.advance(1.1)
            self.assertFalse(service2._jev_due(evidence2))
            clock2.advance(4.0)
            self.assertTrue(service2._jev_due(evidence2))

    def test_terminal_errors_do_not_retry_and_generic_is_terminal(self):
        clock = Clock()
        for exc in (JevTerminalError("bad request"),
                    RuntimeError("Jev HTTP 500; no result accepted.")):
            service = self.make_service(lambda *a, _e=exc: (_ for _ in ()).throw(_e))
            with mock.patch.object(ident.time, "monotonic", clock.monotonic):
                evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
                service._identify(1, evidence)
                self.assertTrue(evidence.jev_terminal)
                self.assertIsNone(evidence.jev_next_retry_at)
                self.assertFalse(service._jev_due(evidence))
                clock.advance(100)
                self.assertFalse(service._jev_due(evidence))
                self.assertEqual(service.jev_attempts, 1)
                self.assertEqual(service.jev_retries, 0)

    def test_jev_catalog_categories_without_message_parsing(self):
        import jev_catalog
        from unittest import mock as _mock
        # Transport -> retryable.
        with _mock.patch.object(jev_catalog.requests, "post",
                                side_effect=jev_catalog.requests.RequestException("net")):
            with _mock.patch.dict("os.environ", {"TYPESAFE_API_KEY": "k"}):
                with self.assertRaises(JevRetryableError):
                    jev_catalog.classify([{"text": "AQUA", "score": 1}])
        # 429 and 503 -> retryable with Retry-After preserved.
        for code, retryable in ((429, True), (503, True), (500, True),
                                (400, False), (401, False), (403, False), (422, False)):
            resp = _mock.Mock(status_code=code, headers={"Retry-After": "7"})
            with _mock.patch.object(jev_catalog.requests, "post", return_value=resp):
                with _mock.patch.dict("os.environ", {"TYPESAFE_API_KEY": "k"}):
                    try:
                        jev_catalog.classify([{"text": "AQUA", "score": 1}])
                        self.fail(f"expected error for {code}")
                    except JevRetryableError as exc:
                        self.assertTrue(retryable, code)
                        if code == 429:
                            self.assertEqual(exc.retry_after, 7.0)
                    except JevTerminalError:
                        self.assertFalse(retryable, code)
        # Missing key and bad payload -> terminal.
        with _mock.patch.object(jev_catalog, "load_dotenv", lambda *a, **k: False):
            with _mock.patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(JevTerminalError):
                    jev_catalog.classify([{"text": "AQUA", "score": 1}])
        bad = _mock.Mock(status_code=200, headers={})
        bad.json.return_value = {"answers": {"product": {
            "choice": "x", "probabilities": {"x": 0.5}, "confidence": 0.5}}}
        with _mock.patch.object(jev_catalog.requests, "post", return_value=bad):
            with _mock.patch.dict("os.environ", {"TYPESAFE_API_KEY": "k"}):
                with self.assertRaises(JevTerminalError):
                    jev_catalog.classify([{"text": "AQUA", "score": 1}])

    def test_success_debounce_preserved(self):
        clock = Clock()
        service = self.make_service(lambda *a: candidate("senezhskaya-0-5l", 0.6))
        with mock.patch.object(ident.time, "monotonic", clock.monotonic):
            evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
            service._identify(1, evidence)
            # Unchanged success never repeats, even past debounce.
            clock.advance(1000)
            self.assertFalse(service._jev_due(evidence))
            # Fresh evidence for debounce check: reset clock reference.
            clock2 = Clock(start=5000.0)
            service2 = self.make_service(lambda *a: candidate("senezhskaya-0-5l", 0.6))
            with mock.patch.object(ident.time, "monotonic", clock2.monotonic):
                evidence2 = service2._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
                service2._identify(1, evidence2)
                service2._merge(1, "Bottle", [{"text": "ВОДА", "score": 0.8}])
                self.assertFalse(service2._jev_due(evidence2))  # changed: debounce holds
                clock2.advance(11.9)
                self.assertFalse(service2._jev_due(evidence2))
                clock2.advance(0.2)
                self.assertTrue(service2._jev_due(evidence2))

    def test_global_budget_blocks_and_survives_reset(self):
        clock = Clock()
        service = self.make_service(lambda *a: candidate("senezhskaya-0-5l", 0.6),
                                    max_attempts=2)
        with mock.patch.object(ident.time, "monotonic", clock.monotonic):
            for track in (1, 2):
                evidence = service._merge(track, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
                service._identify(track, evidence)
            self.assertEqual(service.jev_attempts, 2)
            self.assertEqual(service.jev_stats()["budget_remaining"], 0)
            evidence3 = service._merge(3, "Bottle", [{"text": "AQUA", "score": 0.9}])
            # Due but blocked at dispatch.
            self.assertTrue(service._jev_due(evidence3))
            service._identify(3, evidence3)
            self.assertEqual(service.jev_attempts, 2)
            self.assertEqual(service.jev_budget_blocked, 1)
            self.assertIn("budget exhausted", evidence3.jev_error)
            self.assertNotIn("TYPESAFE_API_KEY", service.readiness().__str__())
            # UI reset must not replenish.
            service.reset()
            self.assertEqual(service.jev_attempts, 2)
            evidence4 = service._merge(4, "Bottle", [{"text": "AQUA", "score": 0.9}])
            service._identify(4, evidence4)
            self.assertEqual(service.jev_attempts, 2)
            self.assertEqual(service.jev_budget_blocked, 2)

    def test_concurrent_workers_no_overspend_or_duplicate(self):
        entered = threading.Event()
        release = threading.Event()
        active = 0
        max_active = 0
        per_track = {}
        lock = threading.Lock()
        def blocking(lines, products, hint):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                key = tuple(sorted(l["text"] for l in lines))
                per_track[key] = per_track.get(key, 0) + 1
            entered.set()
            release.wait(5)
            with lock:
                active -= 1
            return candidate("senezhskaya-0-5l", 0.6)
        cfg = IdentConfig(jev_inline=False, jev_workers=4, jev_max_attempts=3)
        service = IdentificationService(jev_fn=blocking, config=cfg)
        service.start()
        self.addCleanup(service.close)
        for track in range(10):
            service._merge(track, "Bottle", [{"text": f"ITEM{track}X", "score": 0.9}])
        service.jev_wake.set()
        self.assertTrue(entered.wait(3))
        _real_time.sleep(0.3)
        release.set()
        deadline = _real_time.monotonic() + 5
        while _real_time.monotonic() < deadline and service.jev_attempts < 3:
            _real_time.sleep(0.05)
        _real_time.sleep(0.5)
        # Budget caps total dispatches despite 10 due tracks and 4 workers.
        self.assertLessEqual(service.jev_attempts, 3)
        # One track never has overlapping calls: same fingerprint dispatched once.
        for count in per_track.values():
            self.assertEqual(count, 1)
        # Same-track duplicate reservation: overlapping threads, one dispatch.
        service2 = self.make_service(lambda *a: candidate("senezhskaya-0-5l", 0.6))
        evidence = service2._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
        calls = []
        first_entered = threading.Event()
        hold = threading.Event()
        orig = service2.jev_fn
        def counting(lines, products, hint):
            calls.append(1)
            first_entered.set()
            hold.wait(5)
            return orig(lines, products, hint)
        service2.jev_fn = counting
        first = threading.Thread(target=service2._identify, args=(1, evidence))
        first.start()
        self.assertTrue(first_entered.wait(2))
        rest = [threading.Thread(target=service2._identify, args=(1, evidence))
                for _ in range(7)]
        for thread in rest:
            thread.start()
        _real_time.sleep(0.2)  # all challengers attempt while first holds inflight
        hold.set()
        first.join(3)
        for thread in rest:
            thread.join(3)
        self.assertEqual(len(calls), 1)

    def test_reset_prune_reuse_and_stale_drop(self):
        entered, release = threading.Event(), threading.Event()
        def slow(lines, products, hint):
            entered.set()
            release.wait(5)
            return candidate("senezhskaya-0-5l", 0.95)
        service = IdentificationService(jev_fn=slow)
        evidence = service._merge(7, "bottle", [{"text": "WATER", "score": 0.9}],
                                  request_id="a")
        thread = threading.Thread(target=service._identify, args=(7, evidence))
        thread.start()
        self.assertTrue(entered.wait(2))
        service.reset()
        new = service.touch(7, "new")
        service.jev_inflight.add(7)  # replacement reserves
        release.set()
        thread.join(2)
        # Late success must not touch replacement nor clear its reservation.
        self.assertEqual(new.result, {"status": "needs_more_evidence"})
        self.assertFalse(new.complete)
        self.assertIn(7, service.jev_inflight)
        self.assertEqual(service.jev_stale_drops, 1)
        self.assertEqual(service.jev_successes, 1)
        # Prune path: old evidence deleted, numeric ID reused.
        entered.clear()
        release.clear()
        service2 = IdentificationService(jev_fn=slow)
        old = service2._merge(9, "bottle", [{"text": "WATER", "score": 0.9}])
        thread2 = threading.Thread(target=service2._identify, args=(9, old))
        thread2.start()
        self.assertTrue(entered.wait(2))
        from config import IDENT_ABSENT_FRAMES
        for _ in range(IDENT_ABSENT_FRAMES + 1):
            service2.prune(set())
        self.assertNotIn(9, service2.tracks)
        reused = service2.touch(9, "fresh")
        self.assertIsNot(reused, old)
        service2.jev_inflight.add(9)
        release.set()
        thread2.join(2)
        self.assertEqual(reused.result, {"status": "needs_more_evidence"})
        self.assertIn(9, service2.jev_inflight)
        self.assertEqual(service2.jev_stale_drops, 1)

    def test_completion_stops_collection(self):
        service = self.make_service(lambda *a: candidate("senezhskaya-0-5l", 0.95))
        service.submit(1, "Bottle", b"crop")
        evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
        service._identify(1, evidence)
        self.assertTrue(service.is_complete(1))
        self.assertFalse(service.submit(1, "Bottle", b"more"))
        self.assertIsNone(service._merge(1, "Bottle", [{"text": "NEW", "score": 1}]))

    def test_other_tracks_proceed_while_one_backs_off(self):
        clock = Clock()
        def selective(lines, products, hint):
            texts = [l["text"] for l in lines]
            if any("FAILTRACK" in t for t in texts):
                raise JevRetryableError("transient")
            return candidate("senezhskaya-0-5l", 0.6)
        service = self.make_service(selective)
        with mock.patch.object(ident.time, "monotonic", clock.monotonic):
            failing = service._merge(1, "Bottle", [{"text": "FAILTRACK", "score": 0.9}])
            passing = service._merge(2, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
            service._identify(1, failing)
            self.assertFalse(service._jev_due(failing))
            # Other track is still due and succeeds while first waits.
            self.assertTrue(service._jev_due(passing))
            service._identify(2, passing)
            self.assertEqual(passing.result["status"], "candidate")
            self.assertEqual(service.jev_successes, 1)
            clock.advance(1.1)
            self.assertTrue(service._jev_due(failing))

    def test_diagnostics_bounded_without_secrets(self):
        service = self.make_service(lambda *a: candidate("senezhskaya-0-5l", 0.6),
                                    max_attempts=5)
        evidence = service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}],
                                  request_id="r1")
        service._identify(1, evidence)
        stats = service.jev_stats()
        for key in ("attempts", "successes", "failures", "retries",
                    "budget_max", "budget_remaining", "stale_drops"):
            self.assertIn(key, stats)
        debug = service.debug()
        self.assertIn("jev", debug)
        track = debug["tracks"]["1"]
        for key in ("jev_failures", "jev_terminal", "jev_exhausted",
                    "jev_next_retry_in_s", "jev_last_attempt_fingerprint"):
            self.assertIn(key, track)
        readiness = service.readiness()
        self.assertEqual(readiness["jev_attempts"], 1)
        blob = str(stats) + str(debug) + str(readiness)
        self.assertNotIn("Bearer", blob)
        self.assertNotIn("TYPESAFE_API_KEY", blob.replace("JEV_KEY_PRESENT", ""))

    def test_stale_selection_no_duplicate_after_success(self):
        service = self.make_service(lambda *a: candidate("senezhskaya-0-5l", 0.6))
        service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
        tid1, ev1, snap1 = service._select_snapshot()
        tid2, ev2, snap2 = service._select_snapshot()
        self.assertEqual((tid1, tid2), (1, 1))
        self.assertIs(ev1, ev2)
        self.assertEqual(snap1, snap2)
        # First worker reserves and succeeds.
        self.assertTrue(service._identify(tid1, ev1, _snapshot=snap1))
        self.assertEqual(service.jev_attempts, 1)
        # Second worker's stale snapshot must not dispatch a duplicate.
        self.assertFalse(service._identify(tid2, ev2, _snapshot=snap2))
        self.assertEqual(service.jev_attempts, 1)
        self.assertEqual(service.jev_calls, 1)
        # Fresh selection finds nothing due for unchanged evidence.
        self.assertIsNone(service._select_snapshot()[0])

    def test_stale_selection_no_early_retry_after_failure(self):
        clock = Clock()
        def fail_once(lines, products, hint):
            raise JevRetryableError("transient")
        service = self.make_service(fail_once)
        with mock.patch.object(ident.time, "monotonic", clock.monotonic):
            service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}])
            _, ev1, snap1 = service._select_snapshot()
            _, ev2, snap2 = service._select_snapshot()
            self.assertEqual(snap1, snap2)
            self.assertTrue(service._identify(1, ev1, _snapshot=snap1))
            self.assertEqual(ev1.jev_failures, 1)
            self.assertIsNotNone(ev1.jev_next_retry_at)
            # Stale peer must not dispatch before the backoff deadline.
            self.assertFalse(service._identify(1, ev2, _snapshot=snap2))
            self.assertEqual(service.jev_attempts, 1)
            self.assertIsNone(service._select_snapshot()[0])
            clock.advance(1.1)
            tid, ev, snap = service._select_snapshot()
            self.assertEqual(tid, 1)
            # Fresh reservation after the deadline dispatches the retry.
            service.jev_fn = lambda *a: candidate("senezhskaya-0-5l", 0.6)
            self.assertTrue(service._identify(tid, ev, _snapshot=snap))
            self.assertEqual(service.jev_attempts, 2)

    def test_budget_exhaustion_parks_workers_stably(self):
        cfg = IdentConfig(jev_inline=False, jev_workers=2, jev_max_attempts=2)
        service = IdentificationService(
            jev_fn=lambda *a: candidate("senezhskaya-0-5l", 0.6), config=cfg)
        for track in (1, 2, 3):
            service._merge(track, "Bottle", [{"text": f"ITEM{track}XX", "score": 0.9}])
        service.start()
        self.addCleanup(service.close)
        service.jev_wake.set()
        deadline = _real_time.monotonic() + 5
        while _real_time.monotonic() < deadline and service.jev_attempts < 2:
            _real_time.sleep(0.05)
        self.assertEqual(service.jev_attempts, 2)
        # Allow one scheduler pass to mark the blocked track.
        deadline = _real_time.monotonic() + 5
        while _real_time.monotonic() < deadline and service.jev_budget_blocked < 1:
            _real_time.sleep(0.05)
        blocked = service.jev_budget_blocked
        attempts = service.jev_attempts
        self.assertGreaterEqual(blocked, 1)
        # Workers return to blocking wait: counters stay stable across polls,
        # threads stay alive, and the leftover track reports exhaustion.
        _real_time.sleep(1.6)
        self.assertEqual(service.jev_attempts, attempts)
        self.assertEqual(service.jev_budget_blocked, blocked)
        for thread in service.jev_threads:
            self.assertTrue(thread.is_alive())
        leftover = [ev for tid, ev in service.tracks.items()
                    if ev.last_jev_fingerprint is None]
        self.assertTrue(leftover)
        self.assertIn("budget exhausted", leftover[0].jev_error)
        # Shutdown still joins promptly.
        service.close()
        self.assertEqual(service.jev_threads, [])

    def test_scheduler_recovers_unchanged_evidence_without_new_ocr(self):
        calls = []
        def flaky(lines, products, hint):
            calls.append([l["text"] for l in lines])
            if len(calls) == 1:
                raise JevRetryableError("transient")
            return candidate("senezhskaya-0-5l", 0.6)
        cfg = IdentConfig(jev_inline=False, jev_workers=2, jev_max_attempts=40)
        service = IdentificationService(jev_fn=flaky, config=cfg)
        service._merge(1, "Bottle", [{"text": "СЕНЕЖСКАЯ", "score": 0.9}],
                       request_id="only")
        fingerprint = service.tracks[1].fingerprint
        service.start()
        self.addCleanup(service.close)
        service.jev_wake.set()  # scheduler wake only; no extra OCR, no direct _identify
        deadline = _real_time.monotonic() + 8
        while _real_time.monotonic() < deadline and service.jev_successes < 1:
            _real_time.sleep(0.05)
        self.assertEqual(service.jev_successes, 1)
        self.assertEqual(service.jev_attempts, 2)
        self.assertEqual(len(calls), 2)
        # Same fingerprint recovered; failure sequence reset by success.
        self.assertEqual(service.tracks[1].fingerprint, fingerprint)
        self.assertEqual(service.tracks[1].last_jev_fingerprint, fingerprint)
        self.assertEqual(service.tracks[1].jev_failures, 0)
        self.assertEqual(service.tracks[1].jev_last_attempt_source_ids, ["only"])


if __name__ == "__main__":
    unittest.main()
