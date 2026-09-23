"""Live per-track OCR accumulation and Jev identification.

One background thread owns CPU OCR (a persistent .venv-ocr worker) and Jev
requests so the detection preview never stalls. Evidence is keyed strictly by
ByteTrack ID: a new ID starts empty and nothing is copied between bottles.
Detector labels travel only as an unreliable Jev hint, never as a filter.
Uncertain and unknown outcomes stay unresolved; exact SKUs are never claimed.
The TypeSafe key stays in the backend process and is only reported as present
or missing, never exposed to the browser.
"""

import json
import logging
import math
import os
import re
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent

# A crop is re-queued for OCR at most this often per track once evidence is
# flowing; the first attempt goes out faster so brief label views are caught.
SUBMIT_INTERVAL_S = 2.0
SUBMIT_INTERVAL_EMPTY_S = 0.75
# One pending slot per track: a newer crop replaces a stale one instead of
# queueing behind it, and tracks take turns (round-robin) for fairness.
# Minimum gap between Jev calls for the same track; a call also requires a
# changed evidence fingerprint, so idle bottles cause no repeat traffic.
JEV_DEBOUNCE_S = 12.0
# Failure recovery is separate from the successful-result debounce above:
# at most two retries per consecutive failure sequence (initial + 2).
# Nominal earliest-eligibility delays are 1s then 3s; a Retry-After hint
# extends via max(nominal, retry_after). Actual dispatch additionally depends
# on worker availability and scheduler wake-up. Success resets consecutive
# failures to 0; changed evidence after success follows the normal debounce.
# New fingerprints never replenish an exhausted consecutive-failure budget.
# Terminal errors never retry. The process-wide attempt cap survives UI
# resets and track replacement; only a process restart clears it.
# LIGHTSTORE_JEV_MAX_ATTEMPTS=0 means unlimited attempts for live MVP
# testing (no process-wide ceiling); any positive integer keeps a bounded
# budget. Debounce, bounded transient retries, and duplicate/stale guards
# are unchanged in both modes.
JEV_RETRY_DELAYS = (1.0, 3.0)
JEV_MAX_RETRIES = 2
JEV_MAX_ATTEMPTS = 0
# Near-identical still frames are not re-OCRed (hamming distance on a 64-bit
# perceptual hash of the crop). New views always re-run.
DUPLICATE_HAMMING_MAX = 5
# Tracks unseen this long lose their evidence (a reappearing bottle restarts
# as unknown rather than inheriting a stale identity).
TRACK_TTL_S = 30.0
MAX_JEV_LINES = 40
MAX_CROP_BYTES = 300_000
try:
    JEV_WORKERS = int(os.environ.get("LIGHTSTORE_JEV_WORKERS", "4"))
except ValueError:
    raise ValueError("LIGHTSTORE_JEV_WORKERS must be an integer between 1 and 8.") from None
try:
    _JEV_MAX_ENV = os.environ.get("LIGHTSTORE_JEV_MAX_ATTEMPTS", str(JEV_MAX_ATTEMPTS))
    JEV_MAX_ATTEMPTS_ENV = int(_JEV_MAX_ENV)
except ValueError:
    raise ValueError("LIGHTSTORE_JEV_MAX_ATTEMPTS must be an integer >= 0 (0 = unlimited).") from None
if JEV_MAX_ATTEMPTS_ENV < 0:
    raise ValueError("LIGHTSTORE_JEV_MAX_ATTEMPTS must be an integer >= 0 (0 = unlimited).")


@dataclass
class IdentConfig:
    """Scheduling knobs. Defaults are the tuned pipeline; the legacy path
    (jev_inline) exists only for before/after measurement harnesses."""
    submit_interval_s: float = SUBMIT_INTERVAL_S
    submit_interval_empty_s: float = SUBMIT_INTERVAL_EMPTY_S
    jev_debounce_s: float = JEV_DEBOUNCE_S
    dedup: bool = True
    jev_inline: bool = False
    stop_confidence: float = 0.7
    jev_workers: int = JEV_WORKERS
    jev_max_attempts: int = JEV_MAX_ATTEMPTS_ENV
    jev_retry_delays: tuple = JEV_RETRY_DELAYS

    def __post_init__(self):
        if not math.isfinite(self.stop_confidence) or not 0 <= self.stop_confidence <= 1:
            raise ValueError("stop_confidence must be between 0 and 1")
        if (not isinstance(self.jev_workers, int) or isinstance(self.jev_workers, bool)
                or not 1 <= self.jev_workers <= 8):
            raise ValueError("jev_workers must be an integer between 1 and 8")
        if (not isinstance(self.jev_max_attempts, int) or isinstance(self.jev_max_attempts, bool)
                or self.jev_max_attempts < 0):
            raise ValueError("jev_max_attempts must be an integer >= 0 (0 = unlimited)")
        delays = tuple(self.jev_retry_delays)
        if len(delays) != 2 or not all(isinstance(d, (int, float)) and math.isfinite(d) and d >= 0 for d in delays):
            raise ValueError("jev_retry_delays must be two non-negative finite seconds (1.0, 3.0).")
        self.jev_retry_delays = (float(delays[0]), float(delays[1]))


def dhash(jpeg_bytes):
    """64-bit perceptual hash of a crop; None when undecodable."""
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for value in diff.flatten():
        bits = (bits << 1) | int(bool(value))
    return bits


def hamming(left, right):
    return bin(left ^ right).count("1")


def normalize(text):
    """Fingerprint form: case- and whitespace-insensitive, keeps letters."""
    return re.sub(r"\s+", "", str(text).upper())


def useful_evidence(norms):
    """Enough substance for a Jev call: one solid token or two small ones."""
    solid = [n for n in norms if len(n) >= 4]
    small = [n for n in norms if len(n) >= 2]
    return bool(solid) or len(small) >= 2


def _diagnostic_line(line):
    """Text/score plus crop-pixel textline geometry (poly/box) when present.

    Coordinates map 1:1 to the exact submitted crop JPEG bytes (the worker
    predicts on the decoded upload), so the evidence panel can overlay text
    boxes on the displayed crop without any further mapping.
    """
    entry = {"text": str(line.get("text", ""))[:1000], "score": line.get("score")}
    poly = line.get("poly")
    if isinstance(poly, list) and len(poly) == 4:
        entry["poly"] = poly
    box = line.get("box")
    if isinstance(box, list) and len(box) == 4:
        entry["box"] = box
    return entry


@dataclass
class TrackEvidence:
    sources: dict = field(default_factory=dict)
    sources_dropped: int = 0
    complete: bool = False
    hint: str = ""
    lines: dict = field(default_factory=dict)  # norm -> {text, score, hits}
    fingerprint: tuple = ()
    last_jev_fingerprint: tuple = None
    last_jev_time: float = 0.0
    last_submit_time: float = 0.0
    last_seen_time: float = field(default_factory=time.monotonic)
    absent_frames: int = 0
    submitted: int = 0
    ocr_runs: int = 0
    dedup_skips: int = 0
    stale_replaced: int = 0
    last_sharpness: float = 0.0
    last_crop_wh: tuple = ()
    last_crop_jpeg: bytes = b""
    last_ocr_hash: int = None
    first_seen_at: float = None
    first_submit_at: float = None
    first_ocr_done_at: float = None
    first_ident_at: float = None
    last_ocr_ms: float = 0.0
    last_jev_ms: float = 0.0
    result: dict = field(default_factory=lambda: {"status": "needs_more_evidence"})
    jev_error: str = ""
    # Failure-recovery state, separate from the successful-result debounce
    # (last_jev_fingerprint/last_jev_time, updated only on success).
    jev_failures: int = 0
    jev_next_retry_at: float = None
    jev_next_retry_delay: float = 0.0
    jev_terminal: bool = False
    jev_terminal_error: str = ""
    jev_exhausted: bool = False
    jev_last_attempt_fingerprint: tuple = None
    jev_last_attempt_source_ids: list = field(default_factory=list)
    jev_last_attempt_time: float = 0.0


class OcrWorkerClient:
    """Spawn ocr_worker.py in .venv-ocr; one request at a time (see lock)."""

    def __init__(self, root=ROOT, device="cpu"):
        python = root / ".venv-ocr/bin/python"
        if not python.is_file():
            raise FileNotFoundError("OCR environment is missing: .venv-ocr is required.")
        self.process = None
        self.lock = threading.Lock()
        self.connection, child = socket.socketpair()
        self.connection.settimeout(180)
        self.log_path = root / ".cache/ocr-worker.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
                            "OPENBLAS_NUM_THREADS": "2"})
        try:
            with self.log_path.open("a") as log:
                self.process = subprocess.Popen(
                    [str(python), str(root / "ocr_worker.py"),
                     "--fd", str(child.fileno()), "--device", device],
                    cwd=root, env=environment, pass_fds=(child.fileno(),),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                )
            child.close()
            hello = json.loads(self._receive())
            if hello.get("type") != "ready":
                raise ValueError(hello.get("message", "Unexpected OCR worker response."))
            self.metadata = hello
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self):
        import struct

        def read_exact(size):
            data = bytearray()
            while len(data) < size:
                chunk = self.connection.recv(size - len(data))
                if not chunk:
                    raise EOFError("OCR worker connection closed.")
                data.extend(chunk)
            return bytes(data)

        size, = struct.unpack("!I", read_exact(4))
        if not 0 < size <= 8_000_000:
            raise ValueError("Invalid OCR worker message size.")
        return read_exact(size)

    def predict(self, jpeg_bytes):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError(f"OCR worker is not running. Log: {self.log_path}")
            import struct
            payload = struct.pack("!I", len(jpeg_bytes)) + jpeg_bytes
            try:
                self.connection.sendall(payload)
                reply = json.loads(self._receive())
            except Exception:
                self.close()
                raise
            if reply.get("type") != "lines":
                raise RuntimeError(reply.get("message", "OCR worker failed."))
            return reply["lines"]

    def close(self):
        try:
            self.connection.close()
        except OSError:
            pass
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None


class IdentificationService:
    """Accumulate OCR text per track and identify via the local catalog."""

    def __init__(self, ocr_factory=None, jev_fn=None, config=None):
        self.config = config or IdentConfig()
        self.tracks = {}
        self.lock = threading.Lock()
        # Per-track pending slots: track_id -> jpeg bytes. A newer crop
        # replaces a stale one; the worker serves tracks round-robin.
        self.slots = {}
        self.slot_meta = {}
        self.diagnostics = {}
        self.diagnostic_dropped = 0
        self.slot_order = deque()
        self.slot_lock = threading.Lock()
        self.wake = threading.Event()
        self.jev_wake = threading.Event()
        self.jev_inflight = set()
        self.stop_event = threading.Event()
        self.thread = None
        self.jev_threads = []
        self.products = []
        self.catalog_error = ""
        self.ocr = None
        self.ocr_error = ""
        self.ocr_gave_up = False
        self.ocr_factory = ocr_factory or (lambda: OcrWorkerClient(ROOT, "cpu"))
        self.jev_fn = jev_fn
        self.jev_calls = 0
        # Service-wide external-attempt budget: every dispatched HTTP attempt
        # (success or failure) reserves one unit atomically before network
        # dispatch. Survives reset()/prune(); only a process restart clears it.
        self.jev_attempts = 0
        self.jev_successes = 0
        self.jev_failures_total = 0
        self.jev_retries = 0
        self.jev_stale_drops = 0
        self.jev_budget_blocked = 0
        self.capture = {"upload_wh": (), "fps": 0.0, "detect_ms": 0.0}
        try:
            from jev_catalog import load_catalog
            self.products = load_catalog()
        except Exception as error:
            self.catalog_error = str(error)

    def start(self):
        if self.thread is not None:
            return
        # The OCR worker spawns on first demand in the background thread,
        # so importing this service (or running offline tests) costs nothing.
        if self.jev_fn is None:
            try:
                from jev_catalog import classify
                self.jev_fn = classify
            except Exception as error:
                self.jev_fn = None
                logger.warning("Jev helper unavailable: %s", error)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name="identification", daemon=True)
        self.thread.start()
        if not self.config.jev_inline:
            # A bounded pool identifies separate tracks concurrently. A track
            # is reserved by only one worker, so it never issues overlapping
            # requests for successive OCR updates.
            self.jev_threads = [
                threading.Thread(target=self._jev_loop, name=f"ident-jev-{index + 1}", daemon=True)
                for index in range(self.config.jev_workers)
            ]
            for thread in self.jev_threads:
                thread.start()

    def close(self):
        self.stop_event.set()
        self.wake.set()
        self.jev_wake.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
            self.thread = None
        for thread in self.jev_threads:
            thread.join(timeout=10)
        self.jev_threads = []
        if self.ocr is not None:
            self.ocr.close()
            self.ocr = None

    def reset(self):
        # UI reset clears per-track evidence (cancelling pending retries) but
        # never the service-wide attempt budget/counters.
        with self.lock:
            self.tracks = {}
            self.jev_inflight.clear()
            self.diagnostics.clear()
            self.diagnostic_dropped = 0
        with self.slot_lock:
            self.slots = {}
            self.slot_meta.clear()
            self.slot_order.clear()

    def prune(self, active_ids):
        """Drop evidence for tracks gone longer than a short visible gap.

        ByteTrack can hand a removed bottle's ID to a newly placed one, so
        OCR text and identities must not survive an absence: the next
        appearance restarts at "need evidence" instead of inheriting the
        previous bottle.
        """
        from config import IDENT_ABSENT_FRAMES
        now = time.monotonic()
        with self.lock:
            for track_id, evidence in list(self.tracks.items()):
                if track_id in active_ids:
                    evidence.last_seen_time = now
                    evidence.absent_frames = 0
                else:
                    evidence.absent_frames += 1
                    if (evidence.absent_frames > IDENT_ABSENT_FRAMES
                            or now - evidence.last_seen_time > TRACK_TTL_S):
                        del self.tracks[track_id]
                        self.jev_inflight.discard(track_id)
                        with self.slot_lock:
                            meta = self.slot_meta.pop(track_id, None)
                            if meta and meta[2] is not None: meta[2]["ocr_status"] = "track_expired"
                            self.slots.pop(track_id, None)

    def submit(self, track_id, hint, jpeg_bytes, crop_wh=None, *,
               expected=None, diagnostic=None, frame_id=None):
        """Queue the newest crop for a track; replaces a stale pending one.

        Returns True on success or a falsy rejection reason. An
        out-of-order crop (an older frame than one already pending for the
        track) is deliberately dropped with the "stale" sentinel instead
        of overwriting newer pending evidence.
        """
        if not jpeg_bytes or self.ocr_gave_up:
            return False
        now = time.monotonic()
        with self.lock:
            evidence = self.tracks.get(track_id)
            if expected is not None and evidence is not expected:
                return False
            if evidence is None:
                evidence = self.tracks[track_id] = TrackEvidence(hint=hint or "")
            if hint:
                evidence.hint = hint
            evidence.last_seen_time = now
            if evidence.first_seen_at is None:
                evidence.first_seen_at = now
            if evidence.complete:
                return False
            with self.slot_lock:
                old_meta = self.slot_meta.get(track_id)
                if (frame_id is not None and old_meta is not None
                        and old_meta[3] is not None and old_meta[3] > frame_id):
                    return "stale"
            # Empty tracks retry faster so a brief label view is still caught.
            interval = (self.config.submit_interval_s if evidence.ocr_runs
                        else self.config.submit_interval_empty_s)
            if now - evidence.last_submit_time < interval:
                return False
            evidence.last_submit_time = now
            if evidence.first_submit_at is None:
                evidence.first_submit_at = now
            evidence.submitted += 1
            if crop_wh:
                evidence.last_crop_wh = tuple(crop_wh)
            if len(jpeg_bytes) <= MAX_CROP_BYTES:
                evidence.last_crop_jpeg = bytes(jpeg_bytes)
            with self.slot_lock:
                if track_id in self.slots:
                    evidence.stale_replaced += 1
                else:
                    self.slot_order.append(track_id)
                old = self.slot_meta.get(track_id)
                if old and old[2]: old[2]["ocr_status"] = "replaced"
                if diagnostic is not None:
                    self.diagnostics[diagnostic["request_id"]] = diagnostic
                    while len(self.diagnostics) > 200:
                        del self.diagnostics[next(iter(self.diagnostics))]
                        self.diagnostic_dropped += 1
                self.slot_meta[track_id] = (evidence, now, diagnostic, frame_id)
                self.slots[track_id] = bytes(jpeg_bytes)
        self.wake.set()
        return True

    def is_complete(self, track_id):
        with self.lock:
            evidence = self.tracks.get(track_id)
            return evidence is not None and evidence.complete

    def note(self, track_id, hint, sharpness, crop_wh=None, expected=None):
        """Record a seen crop (even a rejected one) so the overlay can show
        collection state instead of nothing."""
        with self.lock:
            evidence = self.tracks.get(track_id)
            if expected is not None and evidence is not expected:
                return
            if evidence is None:
                evidence = self.tracks[track_id] = TrackEvidence(hint=hint or "")
            if hint:
                evidence.hint = hint
            evidence.last_seen_time = now = time.monotonic()
            if evidence.first_seen_at is None:
                evidence.first_seen_at = now
            evidence.absent_frames = 0
            if evidence.complete:
                return
            evidence.last_sharpness = sharpness
            if crop_wh:
                evidence.last_crop_wh = tuple(crop_wh)

    def touch(self, track_id, hint):
        """Mark a track seen by detection without touching sharpness/crop
        state. Stage 1 calls this per detection frame; sharpness and crop
        dimensions are recorded later when the browser-supplied crop
        arrives, keeping image-dependent quality checks on the backend."""
        now = time.monotonic()
        with self.lock:
            evidence = self.tracks.get(track_id)
            if evidence is None:
                evidence = self.tracks[track_id] = TrackEvidence(hint=hint or "")
            if hint:
                evidence.hint = hint
            evidence.last_seen_time = now
            if evidence.first_seen_at is None:
                evidence.first_seen_at = now
            evidence.absent_frames = 0
            return evidence

    def request_due(self, track_id, hint=None):
        """Read-only eligibility peek for crop requests (stage 1).

        Mirrors submit() throttling/completion gates without consuming
        quota or queueing work: True means the backend may ask the browser
        for a fresh crop. The actual submit() on crop receipt re-checks
        throttling, completion and duplicate-image state, so requests that
        arrive too often only cost uplink, never duplicate OCR work."""
        now = time.monotonic()
        with self.lock:
            evidence = self.tracks.get(track_id)
            if evidence is not None and evidence.complete:
                return False
            if evidence is not None:
                interval = (self.config.submit_interval_s if evidence.ocr_runs
                            else self.config.submit_interval_empty_s)
                if now - evidence.last_submit_time < interval:
                    return False
            return not self.ocr_gave_up

    def note_capture(self, upload_wh, fps, detect_ms):
        with self.lock:
            self.capture = {"upload_wh": tuple(upload_wh), "fps": round(fps, 2),
                            "detect_ms": round(detect_ms, 1)}

    def snapshot(self):
        with self.lock:
            return {
                track_id: {
                    "status": evidence.result.get("status", "needs_more_evidence"),
                    "complete": evidence.complete,
                    "choice": evidence.result.get("choice"),
                    "confidence": evidence.result.get("confidence"),
                    "probabilities": evidence.result.get("probabilities"),
                    "hint": evidence.hint,
                    "lines": len(evidence.lines),
                    "evidence_chars": sum(len(n) for n in evidence.lines),
                    "ocr_texts": [entry["text"] for _, entry in sorted(
                        evidence.lines.items(),
                        key=lambda kv: (-kv[1].get("score", 0), kv[1].get("text", "")))[:8]],
                    "jev_error": evidence.jev_error or "",
                    "jev_failures": evidence.jev_failures,
                    "jev_retry_in_s": (None if evidence.jev_next_retry_at is None
                                       else round(max(0.0, evidence.jev_next_retry_at
                                                      - time.monotonic()), 1)),
                    "submitted": evidence.submitted,
                    "ocr_runs": evidence.ocr_runs,
                    "dedup_skips": evidence.dedup_skips,
                    "stale_replaced": evidence.stale_replaced,
                    "sharpness": round(evidence.last_sharpness, 1),
                    "crop_wh": list(evidence.last_crop_wh),
                    "pending": track_id in self.slots,
                }
                for track_id, evidence in self.tracks.items()
            }

    def _budget_exhausted(self):
        """True only when a positive cap is configured and fully consumed."""
        budget_max = self.config.jev_max_attempts
        return budget_max > 0 and self.jev_attempts >= budget_max

    def _budget_remaining(self):
        budget_max = self.config.jev_max_attempts
        if budget_max <= 0:
            return None
        return max(0, budget_max - self.jev_attempts)

    def jev_stats(self):
        """Bounded service-wide Jev diagnostics (no secrets)."""
        with self.lock:
            budget_max = self.config.jev_max_attempts
            attempts = self.jev_attempts
            remaining = None if budget_max <= 0 else max(0, budget_max - attempts)
            return {
                "attempts": attempts,
                "successes": self.jev_successes,
                "failures": self.jev_failures_total,
                "retries": self.jev_retries,
                "budget_max": budget_max,
                "budget_unlimited": budget_max <= 0,
                "budget_remaining": remaining,
                "budget_blocked": self.jev_budget_blocked,
                "stale_drops": self.jev_stale_drops,
                "inflight": len(self.jev_inflight),
            }

    def _budget_message(self):
        return (
            f"Jev attempt budget exhausted ({self.jev_attempts}/{self.config.jev_max_attempts} used); "
            "no further external calls will be made in this server process. "
            "UI reset and track changes do not replenish the budget."
        )

    def _mark_budget_blocked_locked(self, evidence):
        """Stable per-track exhaustion note. Returns True if newly marked."""
        if "budget exhausted" in (evidence.jev_error or ""):
            return False
        evidence.jev_error = self._budget_message()
        self.jev_budget_blocked += 1
        return True

    def _snapshot_state(self, evidence):
        """Eligibility generation: success stamp + failure/backoff/terminal state.

        Fingerprint/sources intentionally excluded so a pending retry may send
        the newest useful evidence. Any success, retryable failure (failures /
        next_retry_at), terminal, exhaustion, or completion change invalidates
        a stale worker selection.
        """
        return (evidence.last_jev_fingerprint, evidence.last_jev_time,
                evidence.jev_failures, evidence.jev_next_retry_at,
                evidence.jev_terminal, evidence.jev_exhausted, evidence.complete)

    def _select_snapshot(self):
        """Oldest due selection with snapshot. Caller owns no lock.

        Returns (track_id, evidence, snapshot) or (None, None, None) when
        nothing is dispatchable. When the process-wide budget is exhausted
        and due work exists, newly due tracks are marked once with a clear
        message and None is returned so workers return to a blocking wait
        instead of spinning; counters stay stable across polls.
        """
        with self.lock:
            if self._budget_exhausted():
                newly = 0
                for track_id, evidence in self.tracks.items():
                    if track_id in self.jev_inflight:
                        continue
                    if self._jev_due(evidence):
                        if "budget exhausted" not in (evidence.jev_error or ""):
                            evidence.jev_error = self._budget_message()
                            newly += 1
                self.jev_budget_blocked += newly
                return (None, None, None)
            due = [(track_id, evidence) for track_id, evidence in self.tracks.items()
                   if track_id not in self.jev_inflight and self._jev_due(evidence)]
            if not due:
                return (None, None, None)

            def _key(item):
                ev = item[1]
                if ev.jev_next_retry_at is not None:
                    return ev.jev_next_retry_at
                return ev.last_jev_time
            due.sort(key=_key)
            track_id, evidence = due[0]
            return (track_id, evidence, self._snapshot_state(evidence))

    def debug(self):
        """Full per-track detail for diagnosis; served by /api/ident-debug."""
        with self.lock:
            now = time.monotonic()
            base = min((e.first_seen_at for e in self.tracks.values()
                        if e.first_seen_at is not None), default=now)

            def rel(moment):
                return None if moment is None else round(moment - base, 2)

            budget_max = self.config.jev_max_attempts
            jev = {
                "attempts": self.jev_attempts,
                "successes": self.jev_successes,
                "failures": self.jev_failures_total,
                "retries": self.jev_retries,
                "budget_max": budget_max,
                "budget_unlimited": budget_max <= 0,
                "budget_remaining": (None if budget_max <= 0
                                     else max(0, budget_max - self.jev_attempts)),
                "budget_blocked": self.jev_budget_blocked,
                "stale_drops": self.jev_stale_drops,
                "inflight": len(self.jev_inflight),
            }
            return {
                "capture": dict(self.capture),
                "queue_depth": len(self.slots),
                "jev": jev,
                "tracks": {
                    str(track_id): {
                        "hint": evidence.hint,
                        "status": evidence.result.get("status"),
                        "complete": evidence.complete,
                        "choice": evidence.result.get("choice"),
                        "confidence": evidence.result.get("confidence"),
                        "submitted": evidence.submitted,
                        "ocr_runs": evidence.ocr_runs,
                        "dedup_skips": evidence.dedup_skips,
                        "stale_replaced": evidence.stale_replaced,
                        "sharpness": round(evidence.last_sharpness, 1),
                        "crop_wh": list(evidence.last_crop_wh),
                        "absent_frames": evidence.absent_frames,
                        "jev_error": evidence.jev_error,
                        "jev_failures": evidence.jev_failures,
                        "jev_terminal": evidence.jev_terminal,
                        "jev_terminal_error": evidence.jev_terminal_error,
                        "jev_exhausted": evidence.jev_exhausted,
                        "jev_next_retry_in_s": (
                            None if evidence.jev_next_retry_at is None
                            else round(max(0.0, evidence.jev_next_retry_at - now), 2)
                        ),
                        "jev_last_attempt_fingerprint": list(evidence.jev_last_attempt_fingerprint or ()),
                        "jev_last_attempt_source_ids": list(evidence.jev_last_attempt_source_ids or [])[:20],
                        "t_first_seen": rel(evidence.first_seen_at),
                        "t_first_submit": rel(evidence.first_submit_at),
                        "t_first_ocr_done": rel(evidence.first_ocr_done_at),
                        "t_first_ident": rel(evidence.first_ident_at),
                        "last_ocr_ms": round(evidence.last_ocr_ms, 1),
                        "last_jev_ms": round(evidence.last_jev_ms, 1),
                        "lines": [
                            {"text": entry["text"], "score": round(entry["score"], 3),
                             "hits": entry["hits"]}
                            for _, entry in sorted(evidence.lines.items())
                        ],
                    }
                    for track_id, evidence in self.tracks.items()
                },
            }

    def replay_debug(self):
        import copy
        with self.lock:
            return {"requests": copy.deepcopy(list(self.diagnostics.values())),
                    "dropped": self.diagnostic_dropped}

    def readiness(self):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        key_present = bool(os.environ.get("TYPESAFE_API_KEY", "").strip())
        with self.lock:
            jev_calls = self.jev_calls
            attempts = self.jev_attempts
            successes = self.jev_successes
            failures = self.jev_failures_total
            retries = self.jev_retries
            stale_drops = self.jev_stale_drops
            blocked = self.jev_budget_blocked
            budget_max = self.config.jev_max_attempts
        return {
            "catalog_ok": not self.catalog_error and bool(self.products),
            "catalog_error": self.catalog_error,
            "products": [p["sku"] for p in self.products],
            "catalog_products": [
                {"sku": p.get("sku"), "name": p.get("name"), "brand": p.get("brand"),
                 "size": p.get("size"), "variant": p.get("variant"),
                 "category": p.get("category")}
                for p in self.products
            ],
            "ocr_available": self.ocr is not None,
            "ocr_error": self.ocr_error,
            "ocr_device": "cpu",
            "jev_key_present": key_present,
            "jev_model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
            "jev_calls": jev_calls,
            "jev_workers": self.config.jev_workers,
            "jev_attempts": attempts,
            "jev_successes": successes,
            "jev_failures": failures,
            "jev_retries": retries,
            "jev_budget_max": budget_max,
            "jev_budget_unlimited": budget_max <= 0,
            "jev_budget_remaining": (None if budget_max <= 0 else max(0, budget_max - attempts)),
            "jev_budget_blocked": blocked,
            "jev_stale_drops": stale_drops,
        }

    def _merge(self, track_id, hint, lines, expected=None, request_id=None):
        with self.lock:
            evidence = self.tracks.get(track_id)
            if expected is not None and evidence is not expected:
                return None
            if evidence is not None and evidence.complete:
                return None
            if evidence is None:
                # submit() normally creates the entry first; tolerate direct merges.
                evidence = self.tracks[track_id] = TrackEvidence(hint=hint or "")
            evidence.absent_frames = 0
            evidence.last_seen_time = time.monotonic()
            if hint:
                evidence.hint = hint
            for line in lines:
                text = str(line.get("text", "")).strip()
                if not text:
                    continue
                norm = normalize(text)
                if len(norm) < 2:
                    continue
                if request_id:
                    sources = evidence.sources.get(norm, []) + [request_id]
                    evidence.sources_dropped += max(0, len(sources)-200)
                    evidence.sources[norm] = sources[-200:]
                entry = evidence.lines.get(norm)
                score = float(line.get("score", 0) or 0)
                if entry is None:
                    evidence.lines[norm] = {"text": text, "score": score, "hits": 1}
                else:
                    if len(text) > len(entry["text"]):
                        entry["text"] = text
                    entry["score"] = max(entry["score"], score)
                    entry["hits"] += 1
            evidence.fingerprint = tuple(sorted(evidence.lines))
            return evidence

    def _jev_due(self, evidence):
        """Success debounce and failure backoff are separate.

        - No pending retry: successful-result path. Requires a fingerprint
          different from the last *successful* dispatch and the normal
          12s debounce.
        - Pending retry: failure-recovery path. Fires when the backoff
          deadline (earliest eligibility; actual dispatch also needs a free
          worker and scheduler wake-up) passes, even for unchanged evidence,
          and sends the newest useful evidence.
        Terminal/exhausted evidence is never due. Consecutive retryable
        failures never reset on new fingerprints; only success resets to 0.
        Budget state is checked at dispatch, not here.
        """
        if evidence.complete:
            return False
        if self.jev_fn is None or not self.products:
            return False
        if not useful_evidence(evidence.fingerprint):
            return False
        if evidence.jev_terminal or evidence.jev_exhausted:
            return False
        now = time.monotonic()
        if evidence.jev_next_retry_at is not None:
            return now >= evidence.jev_next_retry_at
        if evidence.fingerprint == evidence.last_jev_fingerprint:
            return False
        return now - evidence.last_jev_time >= self.config.jev_debounce_s

    @staticmethod
    def _classify_jev_exception(exc):
        """Structured category without parsing human-readable messages."""
        try:
            from jev_catalog import JevRetryableError, JevTerminalError
            if isinstance(exc, JevRetryableError):
                return ("retryable", getattr(exc, "retry_after", None))
            if isinstance(exc, JevTerminalError):
                return ("terminal", None)
            base = None
            try:
                from jev_catalog import JevError as _Base
                base = _Base
            except Exception:
                base = None
            if base is not None and isinstance(exc, base):
                return ("terminal", None)
        except Exception:
            pass
        # Unknown exception kinds (e.g. plain RuntimeError from a fake in
        # tests): fail closed as terminal so programming errors never retry.
        return ("terminal", None)

    def _identify(self, track_id, evidence, _snapshot=None):
        # Reserve one in-flight slot plus one budget unit atomically, then
        # dispatch outside the lock using the newest useful evidence.
        # When _snapshot (taken at selection) is supplied, eligibility is
        # revalidated atomically with reservation: a stale selection that lost
        # a race (peer success, backoff scheduled, retries exhausted, or
        # terminal) is rejected without consuming budget or dispatching.
        # Direct calls pass _snapshot=None and keep the legacy allowlist
        # behavior (fresh snapshot always matches); the worker loop always
        # passes the selection snapshot.
        with self.lock:
            if self.tracks.get(track_id) is not evidence or evidence.complete:
                if self.tracks.get(track_id) is evidence:
                    self.jev_inflight.discard(track_id)
                return False
            if track_id in self.jev_inflight:
                return False
            if self._budget_exhausted():
                self._mark_budget_blocked_locked(evidence)
                return False
            if _snapshot is not None and self._snapshot_state(evidence) != _snapshot:
                return False
            if not useful_evidence(evidence.fingerprint):
                return False
            is_retry = evidence.jev_failures > 0
            self.jev_attempts += 1
            self.jev_calls += 1
            if is_retry:
                self.jev_retries += 1
            self.jev_inflight.add(track_id)
            request_fingerprint = evidence.fingerprint
            source_ids = sorted({r for n in request_fingerprint[:MAX_JEV_LINES]
                                 for r in evidence.sources.get(n, [])})
            lines = [{"text": evidence.lines[n]["text"], "score": evidence.lines[n]["score"]}
                     for n in request_fingerprint[:MAX_JEV_LINES]]
            hint = evidence.hint or None
            # The pending backoff is now running; a failure below schedules
            # the next one, a success clears the failure count.
            evidence.jev_next_retry_at = None
            evidence.jev_next_retry_delay = 0.0
            evidence.jev_last_attempt_fingerprint = request_fingerprint
            evidence.jev_last_attempt_source_ids = list(source_ids)
            evidence.jev_last_attempt_time = time.monotonic()
        started = time.monotonic()
        category = None
        retry_after = None
        try:
            outcome = self.jev_fn(lines, self.products, hint)
            answer = outcome.get("answer", {})
            choice = answer.get("choice")
            status = outcome.get("status", "needs_more_evidence")
            if choice not in {p["sku"] for p in self.products}:
                # Uncertain/unknown outcomes stay unresolved: no SKU is shown.
                choice, status = None, status if status in (
                    "needs_more_evidence", "unknown") else "needs_more_evidence"
                confidence, probabilities = None, None
            else:
                confidence, probabilities = answer.get("confidence"), answer.get("probabilities")
            result = {"status": status, "choice": choice,
                      "confidence": confidence, "probabilities": probabilities}
            error_text = ""
        except Exception as exc:
            # No result is accepted on failure; the track stays unresolved.
            category, retry_after = self._classify_jev_exception(exc)
            result = {"status": "needs_more_evidence", "choice": None,
                      "confidence": None, "probabilities": None}
            error_text = str(exc) or exc.__class__.__name__
            logger.warning("Jev request failed for track %s: %s", track_id, error_text)
        elapsed_ms = (time.monotonic() - started) * 1000
        with self.lock:
            current = self.tracks.get(track_id)
            if current is not evidence:
                # Replacement evidence (numeric ID reuse) or prune/reset won:
                # count the dispatch globally but never touch the new object
                # nor clear its in-flight reservation.
                if error_text:
                    self.jev_failures_total += 1
                else:
                    self.jev_successes += 1
                self.jev_stale_drops += 1
                return True
            self.jev_inflight.discard(track_id)
            if not error_text:
                self.jev_successes += 1
            else:
                self.jev_failures_total += 1
            for request_id in source_ids:
                diagnostic = self.diagnostics.get(request_id)
                if diagnostic is not None:
                    if "jev" in diagnostic:
                        diagnostic["jev_outcomes_replaced"] = diagnostic.get("jev_outcomes_replaced", 0) + 1
                    diagnostic["jev"] = {"source_request_ids": source_ids, "lines": lines,
                                         "result": result, "duration_ms": elapsed_ms,
                                         "source_ids_dropped": evidence.sources_dropped,
                                         "error": bool(error_text)}
            if not error_text:
                evidence.result = result
                confidence = result.get("confidence")
                evidence.complete = (
                    result.get("status") == "candidate" and bool(result.get("choice"))
                    and isinstance(confidence, (int, float))
                    and math.isfinite(confidence)
                    and self.config.stop_confidence <= confidence <= 1
                )
                if evidence.complete:
                    with self.slot_lock:
                        meta = self.slot_meta.pop(track_id, None)
                        if meta and meta[2] is not None: meta[2]["ocr_status"] = "complete"
                        self.slots.pop(track_id, None)
                        self.slot_order = deque(t for t in self.slot_order if t != track_id)
                evidence.jev_error = ""
                evidence.jev_failures = 0
                evidence.jev_next_retry_at = None
                evidence.jev_next_retry_delay = 0.0
                # OCR may have added evidence while this HTTP request was in
                # flight. Record only what the request actually contained so
                # the newer evidence can trigger a subsequent call.
                evidence.last_jev_fingerprint = request_fingerprint
                evidence.last_jev_time = time.monotonic()
                evidence.last_jev_ms = elapsed_ms
                if evidence.first_ident_at is None and result.get("choice"):
                    evidence.first_ident_at = evidence.last_jev_time
                self.jev_wake.set()
                return True
            # Failure path: never touch last_jev_fingerprint/last_jev_time so
            # the successful-result debounce is preserved and unchanged
            # evidence can still recover via backoff.
            # At most two retries per consecutive failure sequence: failures
            # counts consecutive retryable failures since the last success;
            # success resets it to 0. Changed evidence after a success may
            # later trigger a normal call after the successful-result debounce.
            if category == "terminal":
                evidence.result = result
                evidence.jev_error = error_text
                evidence.jev_terminal = True
                evidence.jev_terminal_error = error_text[:500]
                evidence.jev_next_retry_at = None
                evidence.jev_next_retry_delay = 0.0
                evidence.last_jev_ms = elapsed_ms
                self.jev_wake.set()
                return True
            # Retryable: bounded backoff, initial + at most two retries.
            evidence.result = result
            evidence.jev_failures += 1
            evidence.last_jev_ms = elapsed_ms
            if evidence.jev_failures > JEV_MAX_RETRIES:
                evidence.jev_exhausted = True
                evidence.jev_next_retry_at = None
                evidence.jev_next_retry_delay = 0.0
                evidence.jev_error = (
                    f"{error_text} (Jev retries exhausted after "
                    f"{evidence.jev_failures} attempts; no further retries for this evidence.)"
                )[:800]
                self.jev_wake.set()
                return True
            delays = tuple(self.config.jev_retry_delays)
            nominal = float(delays[min(evidence.jev_failures - 1, len(delays) - 1)])
            delay = nominal
            try:
                if retry_after is not None and math.isfinite(float(retry_after)):
                    delay = max(nominal, float(retry_after))
            except (TypeError, ValueError):
                pass
            # Earliest eligibility only: actual dispatch additionally depends
            # on worker availability and scheduler wake-up (1s poll).
            evidence.jev_next_retry_at = time.monotonic() + delay
            evidence.jev_next_retry_delay = delay
            evidence.jev_error = f"{error_text} (retry in {delay:.1f}s.)"[:800]
            self.jev_wake.set()
            return True

    def _ensure_ocr(self):
        if self.ocr is not None or self.ocr_gave_up:
            return
        try:
            self.ocr = self.ocr_factory()
        except Exception as error:
            self.ocr_gave_up = True
            self.ocr_error = str(error)
            logger.warning("OCR worker unavailable: %s", error)

    def _take_slot(self, with_meta=False):
        """Next pending crop; each track holds at most one, so turns rotate
        fairly and this is always the newest view of that bottle."""
        with self.slot_lock:
            while self.slot_order:
                track_id = self.slot_order.popleft()
                jpeg = self.slots.pop(track_id, None)
                if jpeg is not None:
                    meta = self.slot_meta.pop(track_id, (None, time.monotonic(), None, None))
                    return (track_id, jpeg, meta) if with_meta else (track_id, jpeg)
            return None

    def _loop(self):
        while not self.stop_event.is_set():
            self.wake.wait(timeout=1.0)
            self.wake.clear()
            if self.stop_event.is_set():
                break
            with self.slot_lock:
                pending = bool(self.slots)
            if pending:
                self._ensure_ocr()
            while not self.stop_event.is_set():
                item = self._take_slot(with_meta=True)
                if item is None:
                    break
                track_id, jpeg, (expected, queued, diagnostic, _frame) = item
                with self.lock:
                    evidence = self.tracks.get(track_id)
                    hint = evidence.hint if evidence is not None else ""
                if evidence is not expected or evidence is None or evidence.complete or self.ocr is None:
                    with self.lock:
                        if diagnostic is not None: diagnostic["ocr_status"] = "track_expired_or_unavailable"
                    continue
                if self.config.dedup:
                    # A held-still bottle yields effectively identical crops;
                    # re-running OCR on them only burns CPU for no new text.
                    digest = dhash(jpeg)
                    with self.lock:
                        last = evidence.last_ocr_hash
                    if digest is not None and last is not None and hamming(digest, last) <= DUPLICATE_HAMMING_MAX:
                        with self.lock:
                            if self.tracks.get(track_id) is evidence:
                                evidence.dedup_skips += 1
                                if diagnostic is not None: diagnostic["ocr_status"] = "duplicate"
                        continue
                else:
                    digest = None
                started = time.monotonic()
                with self.lock:
                    if diagnostic is not None:
                        diagnostic.update(ocr_status="running", queue_wait_ms=(started-queued)*1000)
                try:
                    lines = self.ocr.predict(jpeg)
                except Exception as error:
                    with self.lock:
                        if diagnostic is not None: diagnostic["ocr_status"] = "error"
                    logger.warning("OCR failed for track %s: %s", track_id, error)
                    continue
                ocr_ms = (time.monotonic() - started) * 1000
                with self.lock:
                    if self.tracks.get(track_id) is not evidence or evidence.complete:
                        # Expired while OCR ran: stale results must not
                        # attach to a reused ID.
                        continue
                    evidence.ocr_runs += 1
                    evidence.last_ocr_ms = ocr_ms
                    if digest is not None:
                        evidence.last_ocr_hash = digest
                    if evidence.first_ocr_done_at is None:
                        evidence.first_ocr_done_at = time.monotonic()
                with self.lock:
                    if diagnostic is not None:
                        diagnostic.update(ocr_status="done", ocr_ms=ocr_ms,
                                          lines=[_diagnostic_line(x) for x in (lines or [])[:100]],
                                          lines_dropped=max(0, len(lines or [])-100),
                                          text_truncated=any(len(str(x.get("text", ""))) > 1000 for x in (lines or [])[:100]))
                merged = self._merge(track_id, hint, lines or [], expected=evidence,
                                     request_id=diagnostic["request_id"] if diagnostic else None)
                if merged is not None and self._jev_due(merged):
                    if self.config.jev_inline:
                        self._identify(track_id, merged)
                    else:
                        self.jev_wake.set()

    def _jev_loop(self):
        """Pool worker: select with snapshot, reserve atomically, dispatch.

        Eligibility (_jev_due snapshot) is revalidated atomically with
        in-flight reservation and budget consumption inside _identify, so a
        worker holding a stale selection cannot dispatch after a peer has
        succeeded, scheduled backoff, exhausted retries, or marked terminal.
        When the process-wide budget is exhausted, _select_snapshot marks
        newly due tracks once and returns None so workers return to a
        blocking wait instead of spinning; counters stay stable.
        Backoff is per-track; other tracks stay schedulable while one waits.
        """
        while not self.stop_event.is_set():
            self.jev_wake.wait(timeout=1.0)
            self.jev_wake.clear()
            if self.stop_event.is_set():
                break
            while not self.stop_event.is_set():
                track_id, evidence, snapshot = self._select_snapshot()
                if track_id is None:
                    break
                self._identify(track_id, evidence, _snapshot=snapshot)
