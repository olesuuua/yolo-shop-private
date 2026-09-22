"""Draw OCR text boxes (crop-pixel polys) onto exact saved crops."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QD = Path(__file__).resolve().parent
OUT = QD / "examples"
OUT.mkdir(exist_ok=True)

reps = json.loads((QD / "ocr-representatives.json").read_text())["crops"]
targ = json.loads((QD / "ocr-targeted.json").read_text()) if (QD / "ocr-targeted.json").exists() else {}
pool = {**reps, **targ}

CASES = ["388:2:31", "437:4:122", "571:3:311", "463:7:160", "580:3:329", "404:2:61"]


def draw(rid):
    e = pool[rid]
    img = cv2.imdecode(np.fromfile(str(ROOT / "reports/video-comparison" / e["crop_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
    for line in e["ocr"]["lines"]:
        txt = str(line.get("text", "")).strip()
        if not txt:
            continue
        poly = line.get("poly")
        if poly and len(poly) == 4:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), max(2, img.shape[1] // 300))
            x, y = int(min(p[0] for p in poly)), int(min(p[1] for p in poly))
            cv2.putText(img, f"{txt[:18]} {line.get('score', 0):.2f}", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.6, img.shape[1] / 700), (0, 255, 0), 2)
        elif line.get("box"):
            x1, y1, x2, y2 = [int(v) for v in line["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Downscale long side to 900 for reviewable size, keep original untouched.
    h, w = img.shape[:2]
    sc = min(1.0, 900 / max(h, w))
    if sc < 1:
        img = cv2.resize(img, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    out = OUT / f"{rid.replace(':', '_')}-ocr-boxes.jpg"
    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(rid, "->", out.name, "lines:", len(e["ocr"]["lines"]))


for rid in CASES:
    draw(rid)
