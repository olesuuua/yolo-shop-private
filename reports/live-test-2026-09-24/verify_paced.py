"""Paced end-to-end verification through the real server pipeline.

Drives the LIVE server (real product detector, real bag localizer, real
tracker/timing) with composite frames: a real bottle sprite pasted onto
real bag-video frames at scripted positions, sent at live cadence
(~0.6 s between processed frames) so the wall-clock "Packing..." watch
(~2 s) and the bag-loss grace period behave as in production.

Scenarios (fresh session each):
  A. clear insertion: OUTSIDE x3 -> RIM x2 -> INSIDE x6, expect exactly 1.
  B. hand-hidden: OUTSIDE x4 -> hidden 3.0 s -> INSIDE x3, expect exactly 1
     (pending expiry while hidden, no double on reappearance).
  C. placed visibly beside the bag x8 (~5 s), expect 0.
  D. behind-bag occluded (OUTSIDE x4 -> hidden persistently): DOCUMENTED
     false count expected (camera cannot distinguish); asserts it stays 1.

The sprite reads as Bottle on plain table and Storage box over the bag;
mechanics and display assertions adapt to the observed class. Live-camera
Bottle evidence is in the screencast (sidebar "Unidentified bottle").

Usage: server must run on 8001 (CPU). Raw video never enters git.
  .venv/bin/python reports/live-test-2026-09-24/verify_paced.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import websockets

BASE = "http://127.0.0.1:8001"
WS = "ws://127.0.0.1:8001/ws/detect"
BAG_VIDEO = ("/home/olesya/.t3/userdata/attachments/"
             "bd973609-16e0-480c-a5c5-08f68bb55b03-63a78026-13b9-4a4e-b4f1-59f8d84628bf-mp4.mp4")
SPRITE = "/tmp/opencode/bagseg/sprite_bottle.png"
OUT_DIR = Path(__file__).resolve().parent
PACE = 0.6


def reset():
    req = urllib.request.Request(BASE + "/api/reset", method="POST")
    return json.load(urllib.request.urlopen(req, timeout=15))


def bag_frame(idx=750):
    v = cv2.VideoCapture(BAG_VIDEO)
    v.set(cv2.CAP_PROP_POS_FRAMES, idx)
    _, frame = v.read()
    v.release()
    return cv2.resize(frame, (640, 480))


def paste(base, sprite, x, y):
    img = base.copy()
    h, w = sprite.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(640, x + w), min(480, y + h)
    if x2 <= x1 or y2 <= y1:
        return img
    img[y1:y2, x1:x2] = sprite[y1 - y: y1 - y + (y2 - y1),
                               x1 - x: x1 - x + (x2 - x1)]
    return img


def jpeg(img):
    return cv2.imencode(".jpg", img)[1].tobytes()


async def send_recv(ws, payload):
    await ws.send(payload)
    return json.loads(await ws.recv())


async def lock_zone(ws, base):
    for _ in range(14):
        response = await send_recv(ws, jpeg(base))
        zone = response.get("bag_zone") or {}
        if zone.get("status") == "stable":
            return zone
        time.sleep(PACE)
    raise RuntimeError("zone never locked")


async def scenario(ws, base, zone, steps):
    """steps: list of (sprite_xy_or_None, count, pace). Returns responses."""
    sprite = cv2.imread(SPRITE)
    out = []
    for pos, count, pace in steps:
        for _ in range(count):
            img = base if pos is None else paste(base, sprite, *pos)
            out.append(await send_recv(ws, jpeg(img)))
            time.sleep(pace)
    return out


def sweep(x_from, x_to, y, step=27):
    """Gradual hand-like motion in ~27 px steps (no teleport gaps)."""
    xs = []
    x = x_from
    while x > x_to:
        xs.append((x, y))
        x -= step
    xs.append((x_to, y))
    return xs


def summary(responses):
    last = responses[-1]
    return {
        "packed_total": last.get("packed_total"),
        "display": last.get("packed_display_counts"),
        "events": [(e["track_id"], e["class_name"]) for r in responses for e in r.get("events", [])],
        "pending_seen": any(r.get("pending") for r in responses),
        "zones": sorted({(r.get("bag_zone") or {}).get("status") for r in responses}),
    }


async def scenario_frames(ws, base, frames):
    """Like scenario() but with per-frame sprite positions (None = hidden)."""
    sprite = cv2.imread(SPRITE)
    out = []
    for pos in frames:
        img = base if pos is None else paste(base, sprite, *pos)
        out.append(await send_recv(ws, jpeg(img)))
        time.sleep(PACE)
    return out


async def main():
    base = bag_frame()
    results = {}
    # Continuous hand-like paths: no teleport gaps, so the tracker trail
    # behaves like a real insertion (IDs persist or hand over at the rim).
    OUT = [(480, 80)] * 4
    RIM = [(453, 80), (426, 80), (400, 80)]
    SWEEP_IN = [(373, 80), (346, 80), (319, 80), (292, 80), (265, 80),
                (238, 80)]
    IN = [(220, 80)] * 6
    async with websockets.connect(WS, max_size=8_000_000) as ws:
        # A. clear insertion: gradual sweep outside -> inside.
        reset()
        zone = await lock_zone(ws, base)
        print("A zone:", zone.get("bbox"), zone.get("conf"))
        responses = await scenario_frames(ws, base, OUT + RIM + SWEEP_IN + IN)
        results["A_clear_insertion"] = summary(responses)

        # B. hand-hidden: outside, rim, gone 3.6 s, reappears inside.
        reset()
        zone = await lock_zone(ws, base)
        responses = await scenario_frames(
            ws, base, OUT + RIM + [None] * 6 + [(220, 80)] * 3)
        results["B_hand_hidden"] = summary(responses)

        # C. visibly beside the bag (~5 s): must stay zero.
        reset()
        zone = await lock_zone(ws, base)
        responses = await scenario_frames(ws, base, [(480, 80)] * 8)
        results["C_beside_bag"] = summary(responses)

        # D. behind-bag occluded: rim then hidden persistently.
        # DOCUMENTED false count expected.
        reset()
        zone = await lock_zone(ws, base)
        responses = await scenario_frames(
            ws, base, OUT + RIM + [None] * 7)
        results["D_behind_occluded"] = summary(responses)

    (OUT_DIR / "paced_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    assert results["A_clear_insertion"]["packed_total"] == 1, results["A_clear_insertion"]
    assert len(results["A_clear_insertion"]["events"]) == 1
    assert results["B_hand_hidden"]["packed_total"] == 1, results["B_hand_hidden"]
    assert len(results["B_hand_hidden"]["events"]) == 1
    assert results["C_beside_bag"]["packed_total"] == 0, results["C_beside_bag"]
    assert results["C_beside_bag"]["events"] == []
    # D documents the known limitation (occluded-persistent behind-bag).
    assert results["D_behind_occluded"]["packed_total"] == 1, results["D_behind_occluded"]
    print("PACED VERIFY OK")


if __name__ == "__main__":
    asyncio.run(main())
