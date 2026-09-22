"""Post-process the A/B replay comparison into runs/bottles CSV/JSON.

Joins (all on the shared monotonic clock):
  runs/<id>/run.json, collected.json, diagnostics.json, debug-after.json
  runs/server-events.jsonl, runs/server-cpu.jsonl, runs/driver-runs.jsonl

Brand rule (strict, same as offline eval): AQUA/AKBA; СЕНЕЖ*; СВЯТ/CBЯT/ИСТОЧ*;
ПРОСТОКВАШИНО; ДОМИК/ДЕРЕВНЕ/DOMIK. Lone AKB / DOM fragments, generic
МОЛОКО/fat/volume, neighboring-brand text and ambiguous fragments never count.
"""
import csv
import json
import re
import shutil
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

WINDOWS_V1 = {"A": (0.0, 8.7), "S": (8.7, 17.5), "W": (17.5, 26.2),
              "P": (26.2, 35.0), "D": (35.0, 1e9)}

# Manual review of every V1 W first hit + V3 S partials (exact crops viewed).
REVIEWED = {
    ("v1-b1", "W"): ("valid", "111:2:156 solo-window W close label; polys on brand; full 11.07/norm 30.82"),
    ("v1-b2", "W"): ("invalid", "172:2:232 shows Aqua+Senezhskaya, no W; W text is OCR misread; W unresolved here"),
    ("v1-a1", "W"): ("contaminated", "68:8:115 Domik-owned grouped crop; W ghost through bottle; neighbor text dominant"),
    ("v1-a2", "W"): ("contaminated", "274:7:398 same class as v1-a1 W hit; neighbor text dominant"),
    ("v3-a1", "S"): ("partial", "Senezhskaya label genuine; OCR keeps 7-char root only"),
    ("v3-b1", "S"): ("partial", "same as v3-a1 S"),
    ("v3-b2", "S"): ("partial", "same as v3-a1 S"),
}


def norm(text):
    return re.sub(r"\s+", "", str(text).upper())


def brand_of(lines):
    txt = " ".join(norm(l.get("text", "")) for l in lines)
    if "AQUA" in txt or "AKBA" in txt:
        return "A"
    if "СЕНЕЖ" in txt or "СEНЕЖ" in txt or "CEНЕЖ" in txt or "SENEZH" in txt:
        return "S"
    if "СВЯТ" in txt or "CBЯT" in txt or "ИСТОЧ" in txt or "UСТОЧ" in txt:
        return "W"
    if "ПРОСТОКВАШИНО" in txt or "PROSTOKVASHINO" in txt:
        return "P"
    if "ДОМИК" in txt or "ДЕРЕВНЕ" in txt or "DOMIK" in txt:
        return "D"
    return None


def pct(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def load_events():
    out = []
    for path in (RUNS / "server-events.jsonl",
                 RUNS / "server-events-invalid-submitbug.jsonl"):
        if path.exists() and path.name == "server-events.jsonl":
            for line in path.read_text().splitlines():
                if line.strip():
                    out.append(json.loads(line))
    return out


def main():
    events = load_events()
    cpu = []
    if (RUNS / "server-cpu.jsonl").exists():
        for line in (RUNS / "server-cpu.jsonl").read_text().splitlines():
            if line.strip():
                cpu.append(json.loads(line))
    # video_ended wall times bound active playback (same monotonic clock)
    ended = {}
    if (RUNS / "driver-runs.jsonl").exists():
        for line in (RUNS / "driver-runs.jsonl").read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                if e.get("event") == "video_ended":
                    ended[e["run"]] = e["wall"]
    run_dirs = sorted(p for p in RUNS.iterdir()
                      if p.is_dir() and (p / "run.json").exists()
                      and json.loads((p / "run.json").read_text()).get("measured"))
    runs_rows, bottle_rows = [], []
    for directory in run_dirs:
        meta = json.loads((directory / "run.json").read_text())
        run, gate = meta["run"], meta["gate"]
        collected = json.loads((directory / "collected.json").read_text())
        diagnostics = json.loads((directory / "diagnostics.json").read_text())
        debug_after = json.loads((directory / "debug-after.json").read_text())
        short = run.split("-")[0]
        w0, w1 = meta["wall_start"], meta["wall_end"]
        # Strict run window: post-roll already includes drain time, and every
        # run ended with an empty queue, so nothing belonging to this run can
        # complete after w1; wider windows bleed the next run's traffic in.
        ev = [e for e in events if w0 <= e["server_monotonic"] <= w1]
        gates_seen = {e.get("gate") for e in ev
                      if e["event"] == "process_crop_response_end"}
        assert gates_seen == {gate}, f"{run}: gate mismatch {gates_seen} != {gate}"
        detects = [e for e in ev if e["event"] == "process_detect_end"]
        # active playback: run start to video end (same monotonic clock)
        play_end = ended.get(run, detects[-1]["server_monotonic"] if detects else w1)
        active_s = max(0.0, play_end - w0)
        det_ms = [d.get("detect_ms", 0) for d in detects]
        # browser RTT for detections
        rtts = []
        blog = directory / "browser-events.jsonl"
        if blog.exists():
            pending = []
            for line in blog.read_text().splitlines():
                e = json.loads(line)
                if e.get("event") == "sent" and e.get("kind") == "detection":
                    pending.append(e["browser_ms"])
                elif e.get("event") == "received" and pending:
                    rtts.append(e["browser_ms"] - pending.pop(0))
        # gate outcomes: server acks are authoritative
        reasons = {}
        for e in ev:
            if e["event"] == "process_crop_response_end":
                r = (e.get("ack") or {}).get("reason", "?")
                reasons[r] = reasons.get(r, 0) + 1
        accepts = reasons.get("accepted", 0)
        # OCR per-request outcomes from end-of-run diagnostics
        diag_reqs = (diagnostics.get("diagnostics") or {}).get("requests", []) \
            if isinstance(diagnostics, dict) else []
        qw = [r["queue_wait_ms"] for r in diag_reqs if r.get("queue_wait_ms") is not None]
        om = [r["ocr_ms"] for r in diag_reqs if r.get("ocr_ms") is not None]
        statuses = {}
        for r in diag_reqs:
            s = r.get("ocr_status", "?")
            statuses[s] = statuses.get(s, 0) + 1
        done = statuses.get("done", 0)
        # OCR completions with text, in time order (server _merge log)
        merges = [e for e in ev if e["event"] == "_merge_end" and e.get("merged")]
        # per-bottle first brand hit
        frames = collected.get("frames", [])
        first_det_wall = detects[0]["server_monotonic"] if detects else None
        # attributed OCR executions: done diagnostics joined to request video ts
        req_vt = {r["request_id"]: r.get("video_timestamp")
                  for r in collected.get("requests", [])}
        done_by_bottle, brand_by_bottle = {}, {}
        for r in diag_reqs:
            if r.get("ocr_status") != "done":
                continue
            vt = req_vt.get(r.get("request_id"))
            b = brand_of(r.get("lines", []))
            if short == "v1" and vt is not None:
                for pid2, (lo, hi) in WINDOWS_V1.items():
                    if lo <= vt < hi:
                        done_by_bottle[pid2] = done_by_bottle.get(pid2, 0) + 1
                        if b == pid2:
                            brand_by_bottle[pid2] = brand_by_bottle.get(pid2, 0) + 1
                        break
        for pid in "ASWPD":
            # first detection attributed by V1 window, else run start
            if short == "v1":
                lo, hi = WINDOWS_V1[pid]
                fdet = [f for f in frames if lo <= (f.get("video_timestamp") or -1) < hi and f.get("tracks")]
                fdet_wall = None
                if fdet:
                    # map frame to server detect time via frame_id
                    fid = fdet[0]["frame_id"]
                    match = [d for d in detects if d.get("frame_id") == fid]
                    fdet_wall = match[0]["server_monotonic"] if match else None
                first_opp = None
                for r in collected.get("requests", []):
                    vt = r.get("video_timestamp")
                    if vt is not None and lo <= vt < hi:
                        first_opp = {"request_id": r["request_id"], "video_timestamp": round(vt, 2)}
                        break
            else:
                fdet_wall = first_det_wall
                first_opp = {"request_id": None, "video_timestamp": None,
                             "note": "all bottles continuously present"}
            # first correct brand OCR for this bottle (brand text attributes itself)
            hit = None
            for m in merges:
                if brand_of([{"text": t} for t in m.get("lines", [])]) == pid:
                    # find diagnostic text for supporting evidence
                    hit = {"wall": round(m["server_monotonic"], 2),
                           "track_id": m.get("merge_track"),
                           "lines": m.get("lines", [])[:4]}
                    break
            # need request-level link: match track+time to diagnostic request
            req_id, ocr_text, vt = None, (hit["lines"] if hit else []), None
            if hit:
                cands = [r for r in diag_reqs
                         if any(str(l.get("text", ""))[:120] in (hit["lines"] or [""])
                                for l in r.get("lines", [])) ] if hit else []
                if cands:
                    req_id = cands[0].get("request_id")
                    creq = next((r for r in collected.get("requests", [])
                                 if r["request_id"] == req_id), None)
                    vt = creq.get("video_timestamp") if creq else None
            delay = round(hit["wall"] - fdet_wall, 2) if hit and fdet_wall else None
            n_exec = sum(1 for r in diag_reqs if r.get("ocr_status") == "done")
            validity, note = REVIEWED.get((run, pid), ("unreviewed", ""))
            bottle_rows.append({
                "run": run, "gate": gate, "video": short, "bottle": pid,
                "first_detection_wall": round(fdet_wall, 2) if fdet_wall else None,
                "first_opportunity": first_opp,
                "first_brand_wall": hit["wall"] if hit else None,
                "first_brand_delay_s": delay,
                "first_brand_request": req_id,
                "first_brand_track": hit["track_id"] if hit else None,
                "first_brand_text": ocr_text,
                "first_brand_video_ts": vt,
                "brand_validity": validity if hit else "unresolved",
                "brand_note": note,
                "n_ocr_exec": done_by_bottle.get(pid),
                "n_brand_ocr": brand_by_bottle.get(pid, 0),
            })
        cpu_win = [c for c in cpu if w0 <= c["server_monotonic"] <= w1]
        runs_rows.append({
            "run": run, "gate": gate, "video": short, "complete": meta["complete"],
            "wall_s": round(w1 - w0, 1), "active_det_s": round(active_s, 1),
            "detections": len(detects),
            "det_per_s": round(len(detects) / active_s, 2) if active_s > 0 else None,
            "rtt_med_ms": round(statistics.median(rtts), 1) if rtts else None,
            "rtt_p95_ms": round(pct(rtts, 0.95), 1) if rtts else None,
            "detect_ms_med": round(statistics.median(det_ms), 1) if det_ms else None,
            "detect_ms_p95": round(pct(det_ms, 0.95), 1) if det_ms else None,
            "ocr_queue_med_ms": round(statistics.median(qw), 1) if qw else None,
            "ocr_queue_p95_ms": round(pct(qw, 0.95), 1) if qw else None,
            "ocr_exec_med_ms": round(statistics.median(om), 1) if om else None,
            "ocr_exec_p95_ms": round(pct(om, 0.95), 1) if om else None,
            "ocr_exec_max_ms": round(max(om), 1) if om else None,
            "ocr_done": done, "ocr_diag_n": len(diag_reqs),
            "accepts": accepts, "reject_reasons": json.dumps(reasons, sort_keys=True),
            "diag_statuses": json.dumps(statuses, sort_keys=True),
            "cpu_avg_pct": round(statistics.mean([c["proc_tree_cpu_pct"] for c in cpu_win]), 1) if cpu_win else None,
            "cpu_peak_pct": max([c["proc_tree_cpu_pct"] for c in cpu_win]) if cpu_win else None,
            "rss_peak_mb": max([c["proc_tree_rss_mb"] for c in cpu_win]) if cpu_win else None,
            "queue_depth_end": (debug_after.get("queue_depth")),
            "pending_end": sum(1 for r in diag_reqs if r.get("ocr_status") in ("queued", "running")),
        })
    with (HERE / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs_rows[0].keys()))
        w.writeheader(); w.writerows(runs_rows)
    with (HERE / "bottles.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bottle_rows[0].keys()))
        w.writeheader()
        for row in bottle_rows:
            row = dict(row)
            row["first_opportunity"] = json.dumps(row["first_opportunity"], ensure_ascii=False)
            row["first_brand_text"] = json.dumps(row["first_brand_text"], ensure_ascii=False)
            w.writerow(row)
    (HERE / "runs.json").write_text(json.dumps(runs_rows, indent=1) + "\n")
    (HERE / "bottles.json").write_text(json.dumps(bottle_rows, ensure_ascii=False, indent=1) + "\n")
    print("runs:", len(runs_rows), "bottles:", len(bottle_rows))


if __name__ == "__main__":
    main()
