"""Measurement server for the A/B gate replay comparison.

Same application code for every run; only the quality gate differs:
  A (old): full-resolution sharpness >= 15 (normalized value ignored).
  B (new): production quality_gate_accepts (full>=15 OR norm@1000>=30).

The gate is switched between runs via a mode file (no restarts, models stay
warm). Jev is disabled explicitly: TYPESAFE_API_KEY is forced empty AND
jev_catalog.classify is replaced by a counting guard that raises before any
network I/O. Zero external requests are asserted at the end of the session.

Logs (same monotonic clock as the driver):
  runs/server-events.jsonl  wrapped-method events with timings
  runs/server-cpu.jsonl     1 Hz process-tree CPU/memory samples
"""
import sys
import os
import json
import time
import threading
import functools
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)
GATE_FILE = HERE / "gate_mode.txt"

sys.path.insert(0, str(ROOT))
os.environ["LIGHTSTORE_DEVICE"] = "cpu"
os.environ["LIGHTSTORE_MODEL"] = "ppyoloe_objects365"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["TYPESAFE_API_KEY"] = ""
os.environ.pop("TYPESAFE_MODEL", None)

assert os.environ["TYPESAFE_API_KEY"] == "", "Jev key must be empty for this comparison"

lock = threading.Lock()
events = 0
log = (RUNS / "server-events.jsonl").open("a", buffering=1)
cpu_log = (RUNS / "server-cpu.jsonl").open("a", buffering=1)


def emit(kind, **data):
    global events
    with lock:
        events += 1
        if events > 100000:
            return
        log.write(json.dumps({"event": kind, "server_monotonic": time.monotonic(),
                              **data}, ensure_ascii=False, default=str) + "\n")


# ---- explicit Jev disable: counting guard, no network possible ----
import jev_catalog  # noqa: E402

jev_guard_calls = 0


def jev_disabled_guard(*args, **kwargs):
    global jev_guard_calls
    with lock:
        jev_guard_calls += 1
    emit("jev_blocked", lines=len(args[0]) if args else 0)
    raise RuntimeError("Jev disabled for gate comparison; no external call made.")


jev_catalog.classify = jev_disabled_guard

import identification  # noqa: E402
import vision  # noqa: E402
from vision import crop_sharpness as _full  # noqa: E402
from config import OCR_SHARPNESS_MIN  # noqa: E402

original_accepts = vision.quality_gate_accepts
_gate_cache = {"mode": None, "mtime": 0.0}


def current_mode():
    try:
        mtime = GATE_FILE.stat().st_mtime
    except OSError:
        return "B"
    if mtime != _gate_cache["mtime"]:
        try:
            _gate_cache["mode"] = GATE_FILE.read_text().strip().upper()
        except OSError:
            _gate_cache["mode"] = "B"
        _gate_cache["mtime"] = mtime
    return _gate_cache["mode"] or "B"


def gated_accepts(sharpness_full, sharpness_norm):
    mode = current_mode()
    if mode == "A":
        try:
            return float(sharpness_full) >= OCR_SHARPNESS_MIN
        except (TypeError, ValueError):
            return False
    return original_accepts(sharpness_full, sharpness_norm)


vision.quality_gate_accepts = gated_accepts


def wrap(cls, name, detail):
    original = getattr(cls, name)

    @functools.wraps(original)
    def measured(self, *args, **kwargs):
        data = detail(self, args, kwargs)
        try:
            result = original(self, *args, **kwargs)
            extra = {}
            if name == "process_detect":
                extra = {"frame_id": result["frame_id"],
                         "session_version": result["session_version"],
                         "tracks": result.get("tracks"),
                         "crop_request_ids": [r["request_id"] for r in result.get("crop_requests", [])],
                         "detect_ms": result.get("detect_ms")}
            elif name == "process_crop_response":
                ack = result
                extra = {"ack": ack, "gate": current_mode()}
            elif name == "_merge":
                extra = {"track_id": args[0] if args else None,
                         "merged": result is not None,
                         "lines": [str(l.get("text", ""))[:120] for l in (args[2] if len(args) > 2 else [])][:20]}
            elif name == "submit":
                extra = {"track_id": args[0] if args else None, "accepted": result,
                         "request_id": (kwargs.get("diagnostic") or {}).get("request_id")}
            elif name == "reset":
                extra = {}
            elif name == "_ensure_ocr":
                extra = {"already_loaded": self.ocr is not None}
            emit(name + "_end", **data, **extra)
            return result
        except Exception as error:
            emit(name + "_error", **data, error_type=type(error).__name__)
            raise

    setattr(cls, name, measured)


wrap(vision.FrameProcessor, "process_detect", lambda s, a, k: {"session_at_entry": s.session_version})
wrap(vision.FrameProcessor, "process_crop_response",
     lambda s, a, k: {"header_id": (a[0] or {}).get("request_id"), "header_track": (a[0] or {}).get("track_id")})
wrap(vision.FrameProcessor, "reset", lambda s, a, k: {"session_before": s.session_version})
wrap(identification.IdentificationService, "submit",
     lambda s, a, k: {"submit_track": a[0] if a else None})
wrap(identification.IdentificationService, "_merge",
     lambda s, a, k: {"merge_track": a[0] if a else None})
wrap(identification.IdentificationService, "_ensure_ocr", lambda s, a, k: {})


def cpu_sampler():
    try:
        import psutil
    except ImportError:
        return
    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    while True:
        time.sleep(1.0)
        try:
            with lock:
                kids = proc.children(recursive=True)
            total_cpu = proc.cpu_percent(interval=None)
            total_rss = proc.memory_info().rss
            for kid in kids:
                try:
                    total_cpu += kid.cpu_percent(interval=None)
                    total_rss += kid.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            cpu_log.write(json.dumps({"server_monotonic": time.monotonic(),
                                      "proc_tree_cpu_pct": round(total_cpu, 1),
                                      "proc_tree_rss_mb": round(total_rss / 1e6, 1),
                                      "n_children": len(kids)}) + "\n")
        except Exception:
            return


threading.Thread(target=cpu_sampler, name="cpu-sampler", daemon=True).start()

import app  # noqa: E402
import uvicorn  # noqa: E402

emit("server_start", gate_impl="file-switched A/B", jev="disabled-guard")
uvicorn.run(app.app, host="127.0.0.1", port=8002, log_level="warning")
