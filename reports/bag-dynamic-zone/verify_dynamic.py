"""Video verification for the dynamic whole-bag zone (investigation harness).

Runs the REAL bag localizer (local YOLOE-26n seg, "plastic bag" prompt) and
the REAL BagZoneTracker with production parameters over the recorded bag
video, then drives the REAL PackingTracker with scripted product boxes:

- track 1 (genuine transfer): outside the zone until it locks, then carried
  to the live zone centroid while stable -> must pack exactly once;
- track 2 (stationary bottle): never moves; the bag sweeps over it during
  the movement stage -> must never pack.

Nothing is uploaded anywhere; the raw video is never copied into the repo.
Annotated contour examples + metrics go to evidence/ (tracked, small).

Usage:
  .venv/bin/python reports/bag-dynamic-zone/verify_dynamic.py \
      --video /path/to/vidoe_bag.mp4 [--step 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EVIDENCE = Path(__file__).resolve().parent / "evidence"

# Fixed in 640x480 detection pixels; chosen after inspecting the locked zone
# geometry (see report): top-left table area, never covered by the bag.
GENUINE_OUTSIDE = (30.0, 30.0, 110.0, 110.0)
# Stationary bottle: table position the moving bag sweeps across late in the
# video (movement stage centroid path passes through it).
STATIONARY_BOTTLE = (420.0, 150.0, 500.0, 270.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--step", type=int, default=5,
                        help="Video-frame stride between processed frames.")
    args = parser.parse_args()

    import cv2
    import numpy as np

    from bag_zone import BagLocalizer, BagZoneTracker, load_bag_model
    from config import (
        BAG_ACQUIRE_STABLE, BAG_ADOPT_IOU, BAG_CONF_THRESHOLD,
        BAG_HEARTBEAT_FRAMES, BAG_MIN_FRAC, BAG_MISSES_TO_LOSE,
        BAG_MOTION_THRESHOLD, BAG_OVERLAP_THRESHOLD, BAG_RELOCK_IOU,
    )
    from tracking import Detection, PackingTracker
    from vision import annotate_frame

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = load_bag_model(ROOT)
    zone = BagZoneTracker(
        BagLocalizer(model, BAG_CONF_THRESHOLD, BAG_MIN_FRAC),
        overlap_threshold=BAG_OVERLAP_THRESHOLD,
        heartbeat_frames=BAG_HEARTBEAT_FRAMES,
        acquire_stable=BAG_ACQUIRE_STABLE, adopt_iou=BAG_ADOPT_IOU,
        relock_iou=BAG_RELOCK_IOU, motion_threshold=BAG_MOTION_THRESHOLD,
        misses_to_lose=BAG_MISSES_TO_LOSE)
    tracker = PackingTracker()
    tracker.zone_test = zone.zone_test

    status_hist = Counter()
    examples = {}
    want_examples = {0, 150, 600, 750, 1050, 1200, 1350, 1500, 1600, 1700, 1800}
    genuine_inside_seen = stationary_inside_seen = 0
    genuine_phase = [0]
    frame_id = 0
    started = time.perf_counter()
    for video_frame in range(0, total, args.step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
        ok, full = cap.read()
        if not ok:
            break
        small = cv2.resize(full, (640, 480))
        frame_id += 1
        state = zone.update(small, frame_id)
        status_hist[state["status"]] += 1
        tracker.zone_test = zone.zone_test
        tracker.zone_version = zone.zone_test.version
        tracker.set_packing_paused(zone.packing_paused)

        detections = []
        # Genuine transfer starts only after the zone locks: three outside
        # observations, then the box rides the live zone centroid.
        if state["status"] == "stable":
            genuine_phase[0] = min(genuine_phase[0] + 1, 4)
        if genuine_phase[0] >= 1:
            if genuine_phase[0] <= 3:
                detections.append(Detection(1, "Bottle", GENUINE_OUTSIDE))
            else:
                cx = (zone.zone_bbox[0] + zone.zone_bbox[2]) / 2
                cy = (zone.zone_bbox[1] + zone.zone_bbox[3]) / 2
                box = (cx - 20, cy - 20, cx + 20, cy + 20)
                detections.append(
                    Detection(1, "Bottle", tuple(float(v) for v in box)))
                genuine_inside_seen += 1
        detections.append(Detection(2, "Bottle", STATIONARY_BOTTLE))
        if bool(zone.zone_test(STATIONARY_BOTTLE)):
            stationary_inside_seen += 1
        tracker.update(detections)

        nearest = min(want_examples, key=lambda w: abs(w - video_frame))
        if abs(nearest - video_frame) < args.step and nearest not in examples:
            annotated = annotate_frame(small.copy(), [], tracker, {},
                                       zone, "dynamic")
            path = EVIDENCE / f"dynamic_f{nearest:04d}_{state['status']}.jpg"
            cv2.imwrite(str(path), annotated)
            examples[nearest] = {"file": path.name, "status": state["status"],
                                 "conf": state["conf"], "bbox": state["bbox"]}
    cap.release()
    wall_s = time.perf_counter() - started

    summary = {
        "video_frames": total, "stride": args.step,
        "processed_frames": frame_id,
        "status_histogram": dict(status_hist),
        "bag_inferences": zone.inference_count,
        "inference_ms_ema": zone.inference_ms_ema,
        "wall_s": round(wall_s, 1),
        "genuine_inside_frames": genuine_inside_seen,
        "stationary_inside_frames": stationary_inside_seen,
        "packed_counts": dict(tracker.packed_counts),
        "packed_events": tracker.packed_events,
        "examples": {str(k): v for k, v in sorted(examples.items())},
    }
    (EVIDENCE / "dynamic_results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    genuine = tracker.packed_counts.get("Bottle", 0)
    assert genuine == 1, f"genuine transfer must pack exactly once, got {genuine}"
    stationary_packs = [e for e in tracker.packed_events if e["track_id"] == 2]
    assert not stationary_packs, f"stationary bottle must never pack: {stationary_packs}"
    assert stationary_inside_seen > 0, "stationary bottle was never crossed: test vacuous"
    print("VERIFY OK: genuine packed once; stationary crossed but never packed.")


if __name__ == "__main__":
    main()
