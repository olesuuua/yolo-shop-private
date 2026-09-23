"""Evidence-based feasibility test: can the existing detector locate the bag opening?

Investigation-only harness for the dynamic-bag-zone question. It does NOT
change production behavior (fixed BAG_ROI in config.py stays untouched).

What it does:
  1. Extracts representative frames from a local bag video (stages:
     acquisition / stable-open / product-entering / obstruction / movement).
  2. Runs the Open Images YOLOv8n detector (the only configured profile
     whose FOOD_CLASSES contain "Plastic bag") on 640x480 frames, exactly as
     the server resizes them, at CONF_THRESHOLD=0.10.
  3. Compares each returned "Plastic bag" box against a human-marked bag
     OPENING box (not the bag body) and records IoU / center / edge error.
  4. Simulates a moving-zone packing tracker to show how a jumping/lost zone
     can manufacture a false "packed" event for a stationary product.
  5. Writes annotated frames + results.json under
     reports/bag-dynamic-zone/evidence/ and prints a markdown summary.

Usage (Jev stays disabled; bottle identification is out of scope):
  .venv/bin/python reports/bag-dynamic-zone/bag_feasibility.py \
      --video /path/to/vidoe_bag.mp4
  .venv/bin/python reports/bag-dynamic-zone/bag_feasibility.py \
      --video /path/to/vidoe_bag.mp4 --reproduce  # rerun + verify expectations

The raw video is never copied into the repo and never uploaded anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Stage table. Frame indices refer to the 60 fps source video (1866 frames).
# Opening boxes are human estimates in 640x480 detection pixels, marking the
# bag OPENING (the hole products pass through), not the bag body. None means
# no opening is definable in that frame (motion blur / gathered / occluded).
# ---------------------------------------------------------------------------
STAGES: list[dict] = [
    {"frame": 0, "stage": "empty-table (control)", "opening": None},
    {"frame": 150, "stage": "initial acquisition (bag entering, motion blur)",
     "opening": None},
    {"frame": 300, "stage": "placement (hand spreading the bag)",
     "opening": (100, 170, 380, 370)},
    {"frame": 600, "stage": "placement settling (hand at rim)",
     "opening": (125, 155, 375, 375)},
    {"frame": 750, "stage": "stable open bag, no hand",
     "opening": (115, 165, 415, 375)},
    {"frame": 900, "stage": "stable open bag, no hand",
     "opening": (120, 160, 420, 380)},
    {"frame": 1050, "stage": "product entering (hand + water bottle)",
     "opening": (125, 155, 475, 415)},
    {"frame": 1200, "stage": "product inside (water bottle lying in bag)",
     "opening": (90, 100, 490, 425)},
    {"frame": 1350, "stage": "product entering (second bottle, cola)",
     "opening": (125, 90, 475, 390)},
    {"frame": 1500, "stage": "stable, two bottles inside",
     "opening": (125, 85, 475, 385)},
    {"frame": 1650, "stage": "hand obstruction (gathering the bag)",
     "opening": None},
    {"frame": 1800, "stage": "bag movement (gathered, opening collapsed)",
     "opening": None},
]

EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
CONF = 0.10  # matches config.CONF_THRESHOLD
IMGSZ = 640


def box_iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if (aa + bb - inter) > 0 else 0.0


def box_center(b: tuple) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def load_openimages_model():
    """Same weights/revision pinning as config.py (openimages profile)."""
    import os

    os.environ.setdefault("LIGHTSTORE_MODEL", "openimages")
    import config  # noqa: F401  (validates LIGHTSTORE_IMGSZ etc.)

    from config import MODEL_HF_FILENAME, MODEL_HF_REPO, MODEL_HF_REVISION
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    ckpt = Path(hf_hub_download(
        repo_id=MODEL_HF_REPO, filename=MODEL_HF_FILENAME,
        revision=MODEL_HF_REVISION, cache_dir=ROOT / ".cache" / "huggingface",
        token=False,
    ))
    return YOLO(str(ckpt))


def run(video: Path, out_dir: Path) -> dict:
    import cv2

    model = load_openimages_model()
    bag_id = next(k for k, v in model.names.items() if v == "Plastic bag")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    from config import BAG_ROI

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    latencies: list[float] = []
    for entry in STAGES:
        idx = entry["frame"]
        if idx >= total:
            raise SystemExit(f"Frame {idx} beyond video length {total}.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, full = cap.read()
        if not ok:
            raise SystemExit(f"Cannot decode frame {idx}.")
        small = cv2.resize(full, (640, 480))

        start = time.perf_counter()
        result = model.predict(small, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(latency_ms)

        bags: list[dict] = []
        others: list[dict] = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            for xyxy, cls_id, conf in zip(boxes.xyxy.cpu().tolist(),
                                          boxes.cls.int().cpu().tolist(),
                                          boxes.conf.cpu().tolist()):
                name = result.names[cls_id]
                record = {"class": name, "conf": round(float(conf), 3),
                          "box": [round(float(v), 1) for v in xyxy]}
                (bags if cls_id == bag_id else others).append(record)

        opening = entry["opening"]
        best = max(bags, key=lambda b: b["conf"]) if bags else None
        iou = center_err = bottom_err = None
        if best is not None and opening is not None:
            iou = round(box_iou(tuple(best["box"]), tuple(opening)), 3)
            cx, cy = box_center(tuple(best["box"]))
            ox, oy = box_center(tuple(opening))
            center_err = round(((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5, 1)
            bottom_err = round(abs(best["box"][3] - opening[3]), 1)

        # Annotated frame: yellow = fixed BAG_ROI, green = human opening,
        # magenta = detector Plastic-bag box.
        annotated = small.copy()
        x1, y1, x2, y2 = (int(v) for v in BAG_ROI)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(annotated, "fixed BAG_ROI", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        if opening is not None:
            ox1, oy1, ox2, oy2 = opening
            cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (90, 220, 100), 2)
            cv2.putText(annotated, "human opening", (ox1, max(14, oy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 100), 1)
        if best is not None:
            bx1, by1, bx2, by2 = (int(v) for v in best["box"])
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 80, 255), 2)
            cv2.putText(annotated, f"Plastic bag {best['conf']}",
                        (bx1, min(470, by2 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 255), 1)
        img_path = out_dir / f"stage_f{idx:04d}.jpg"
        cv2.imwrite(str(img_path), annotated)

        rows.append({
            "frame": idx, "stage": entry["stage"],
            "opening_640x480": list(opening) if opening else None,
            "latency_ms": round(latency_ms, 1),
            "plastic_bag_detections": bags,
            "other_detections": others,
            "best_conf": best["conf"] if best else None,
            "best_box": best["box"] if best else None,
            "opening_iou": iou, "center_error_px": center_err,
            "bottom_edge_error_px": bottom_err,
            "annotated": img_path.name,
        })
    cap.release()
    lat = sorted(latencies)
    summary = {
        "video": video.name,
        "model": "yolov8n-oiv7.pt (openimages profile, Plastic bag id 394)",
        "conf_threshold": CONF, "inference_size": IMGSZ,
        "latency_ms": {"n": len(lat), "median": round(lat[len(lat) // 2], 1),
                       "min": round(lat[0], 1), "max": round(lat[-1], 1)},
        "frames": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def simulate_moving_zone_false_pack() -> dict:
    """Show how a jumping zone manufactures a packing event.

    A bottle sits stationary on the table. The dynamic zone is lost for a
    few frames (bag missed), then reappears shifted so it now covers the
    stationary bottle. PackingTracker (MIN_OUTSIDE_FRAMES=3,
    MIN_INSIDE_FRAMES=3) then counts the motionless bottle as packed.
    Geometry is synthetic; the mechanism is the production state machine.
    """
    from tracking import Detection, PackingTracker

    bottle = (500.0, 300.0, 580.0, 420.0)  # never moves
    tracker = PackingTracker(roi=(220, 140, 420, 380))
    events: list[dict] = []
    # Frames 1-3: zone near the bag (bottle outside) -> seen_outside=True.
    for _ in range(3):
        tracker.roi = (220, 140, 420, 380)
        events += tracker.update([Detection(7, "Bottle", bottle)])
    # Frames 4-5: bag lost -> zone frozen (or dropped); bottle unseen inside.
    for _ in range(2):
        events += tracker.update([Detection(7, "Bottle", bottle)])
    # Frames 6-8: zone jumps onto the stationary bottle (bag re-detected
    # shifted, or a false-positive box) -> inside counter accrues.
    for _ in range(3):
        tracker.roi = (480, 280, 600, 440)
        events += tracker.update([Detection(7, "Bottle", bottle)])
    return {"false_packed_events": events,
            "note": "Stationary bottle counted as packed after zone jump."}


def markdown(summary: dict, sim: dict) -> str:
    lines = ["| frame | stage | detected | conf | box | IoU vs opening |",
             "|---|---|---|---|---|---|"]
    for r in summary["frames"]:
        lines.append(
            f"| {r['frame']} | {r['stage']} | "
            f"{'yes' if r['best_box'] else 'no'} | {r['best_conf']} | "
            f"{r['best_box']} | {r['opening_iou']} |")
    lines.append(
        f"\nLatency (CPU, yolov8n-oiv7, 640px): {summary['latency_ms']}")
    lines.append(f"Moving-zone simulation: {sim['false_packed_events']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Local bag video path.")
    parser.add_argument("--reproduce", action="store_true",
                        help="Verify key expectations after the run.")
    args = parser.parse_args()

    summary = run(Path(args.video), EVIDENCE_DIR)
    sim = simulate_moving_zone_false_pack()
    print(markdown(summary, sim))

    if args.reproduce:
        by_frame = {r["frame"]: r for r in summary["frames"]}
        # Core feasibility expectations from the investigated recording.
        assert by_frame[750]["best_box"], "stable open bag should detect once"
        assert not by_frame[900]["best_box"], "stable frame missed (recall gap)"
        assert not by_frame[1350]["best_box"], "cola entry missed (recall gap)"
        assert not by_frame[1650]["best_box"], "obstructed bag must not detect"
        assert sim["false_packed_events"], "simulation must show false pack"
        print("\nREPRODUCE OK: expectations hold for this recording.")


if __name__ == "__main__":
    main()
