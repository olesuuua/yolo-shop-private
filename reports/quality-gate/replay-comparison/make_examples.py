"""Overlay OCR text boxes (crop-pixel polys from run diagnostics) on exact crops."""
import json
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE / "examples"
OUT.mkdir(exist_ok=True)

CASES = [("v1-b1", "111:2:156"), ("v1-b2", "172:2:232"),
         ("v1-a1", "68:8:115"), ("v3-b1", "365:5:550")]

for run, rid in CASES:
    diag = json.loads((RUNS / run / "diagnostics.json").read_text())
    reqs = (diag.get("diagnostics") or {}).get("requests", [])
    entry = next(r for r in reqs if r.get("request_id") == rid)
    img = cv2.imdecode(np.fromfile(
        str(RUNS / run / "crops" / (rid.replace(":", "_") + ".jpg")),
        dtype=np.uint8), cv2.IMREAD_COLOR)
    for line in entry.get("lines", []):
        txt = str(line.get("text", "")).strip()
        if not txt:
            continue
        poly = line.get("poly")
        if poly and len(poly) == 4:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), max(2, img.shape[1] // 300))
            x, y = int(min(p[0] for p in poly)), int(min(p[1] for p in poly))
            cv2.putText(img, f"{txt[:18]} {line.get('score', 0):.2f}", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, img.shape[1] / 700),
                        (0, 255, 0), 2)
    h, w = img.shape[:2]
    sc = min(1.0, 900 / max(h, w))
    if sc < 1:
        img = cv2.resize(img, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    out = OUT / f"{run}-{rid.replace(':', '_')}-ocr-boxes.jpg"
    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(run, rid, "->", out.name)
