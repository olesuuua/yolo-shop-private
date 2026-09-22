"""Offline gate comparison on saved crops: run CPU OCR once per representative.

Keeps the OCR model, recognition settings and input preprocessing identical
to production (ocr_worker.py over the private socket). Measures the current
full-resolution Laplacian gate and normalized-scale alternatives per crop,
then records OCR text/polygons/timings so gate variants can be compared
without re-running recognition.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ppyoloe_protocol import receive, send  # noqa: E402
from identification import dhash  # noqa: E402

# Canonical heights for the scale-aware alternatives: crops in the dataset
# range from ~450x1000 to ~758x1650, so the raw Laplacian variance mixes
# text-pixel size with texture energy. INTER_AREA downscale to a canonical
# height keeps aspect and measures detail at a common text scale.
CANONICAL_HEIGHTS = (600, 1000, 1600)


def start_worker():
    python = ROOT / ".venv-ocr/bin/python"
    parent, child = socket.socketpair()
    child.set_inheritable(True)
    process = subprocess.Popen(
        [str(python), str(ROOT / "ocr_worker.py"), "--fd", str(child.fileno()),
         "--device", "cpu"],
        cwd=ROOT, pass_fds=(child.fileno(),),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
    )
    child.close()
    ready = json.loads(receive(parent).decode())
    assert ready["type"] == "ready", ready
    return parent, process


def ocr_one(connection, jpeg_bytes):
    started = time.monotonic()
    send(connection, jpeg_bytes)
    reply = json.loads(receive(connection).decode())
    elapsed = (time.monotonic() - started) * 1000
    assert reply["type"] == "lines", reply
    return {"lines": reply["lines"], "ocr_ms": round(elapsed, 1)}


def measurements(jpeg_bytes):
    """Sharpness variants: current full-resolution plus normalized-scale."""
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"full": None, "normalized": {}}
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = {}
    for canonical in CANONICAL_HEIGHTS:
        if height != canonical:
            scale = canonical / height
            resized = cv2.resize(image, (max(1, round(width * scale)), canonical),
                                 interpolation=cv2.INTER_AREA)
        else:
            resized = image
        small = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        normalized[canonical] = float(cv2.Laplacian(small, cv2.CV_64F).var())
    return {"full": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "normalized": normalized,
            "wh": [width, height], "dhash": dhash(jpeg_bytes)}


def main():
    manifest = json.loads((Path(__file__).parent / "crops-manifest.json").read_text())
    rows = manifest["rows"]
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    # One representative per group per kind: keep the highest-sharpness
    # rejected (blurry) and accepted member; both may coincide per group.
    representatives = {}
    for group, members in groups.items():
        for kind in ("blurry", "accepted"):
            pool = [m for m in members if m["reason"] == kind]
            if pool:
                best = max(pool, key=lambda r: r["sharpness"] or 0.0)
                representatives[best["request_id"]] = best

    connection, process = start_worker()
    out = {"schema": "quality-gate-ocr-1", "canonical_heights": CANONICAL_HEIGHTS,
           "crops": {}}
    try:
        for index, (request_id, row) in enumerate(sorted(representatives.items())):
            path = ROOT / "reports/video-comparison" / row["crop_path"]
            jpeg = path.read_bytes()
            entry = measurements(jpeg)
            entry.update(ocr=ocr_one(connection, jpeg))
            entry["crop_path"] = row["crop_path"]
            entry["run"] = row["run"]
            out["crops"][request_id] = entry
            if (index + 1) % 25 == 0:
                print(f"{index + 1}/{len(representatives)}", flush=True)
    finally:
        connection.close()
        process.terminate()
    destination = Path(__file__).parent / "ocr-representatives.json"
    destination.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print("representatives:", len(representatives))


if __name__ == "__main__":
    main()
