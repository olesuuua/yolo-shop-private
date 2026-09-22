"""A/B gate replay driver (CDP + Chrome, one replay at a time).

Sequence per video: A-B-B-A (two runs per gate). Warmup replay first
(unmeasured). Fresh POST /api/reset per run with drain verification.
Jev stays disabled server-side; driver asserts jev_calls==0 and
jev_key_present==false before and after every run.

Writes (same monotonic clock as the server log):
  runs/driver-runs.jsonl   run boundaries + polls summaries
  runs/<run-id>/run.json, browser-events.jsonl, collected.json,
                 crops/<request>.jpg, debug-before/after.json,
                 readiness-before/after.json, diagnostics.json
"""
import asyncio
import base64
import json
import time
import urllib.request
from pathlib import Path

import websockets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = HERE / "runs"
COMPAT = ROOT / "reports/video-comparison" / "compatible"
GATE_FILE = HERE / "gate_mode.txt"
SERVER = "http://127.0.0.1:8002"
CDP_URL = "http://127.0.0.1:9223/json/list"

VIDEOS = [
    ("v1", "20260921_205640.mp4"),
    ("v2", "20260921_205735.mp4"),
    ("v3", "20260921_205754.mp4"),
]
POST_ROLL_S = 20.0


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self.seq = 0
        self.pending = {}
        self.events = []

    async def read(self):
        async for raw in self.ws:
            d = json.loads(raw)
            if "id" in d:
                f = self.pending.pop(d["id"], None)
                if f:
                    f.set_result(d)
            elif d.get("method") in ["Runtime.exceptionThrown", "Log.entryAdded"]:
                self.events.append(d)

    async def call(self, method, **params):
        self.seq += 1
        f = asyncio.get_running_loop().create_future()
        self.pending[self.seq] = f
        await self.ws.send(json.dumps({"id": self.seq, "method": method, "params": params}))
        d = await f
        if "error" in d:
            raise RuntimeError(d["error"])
        return d.get("result", {})

    async def evaluate(self, expression):
        r = await self.call("Runtime.evaluate", expression=expression,
                            returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r.get("result", {}).get("value")


OBSERVE = r'''(() => {
window.comparisonEvents=[];
const originalSend=WebSocket.prototype.send;
WebSocket.prototype.send=function(data) {
 if (!this.comparisonObserved) {
  this.comparisonObserved=true;
  this.addEventListener('message',e=>{try {
    const d=JSON.parse(e.data); delete d.image;
    comparisonEvents.push({event:'received',browser_ms:performance.now(),data:d});
  } catch {}});
 }
 const detect=data instanceof Blob && data.type==='image/jpeg';
 comparisonEvents.push({event:'sent',browser_ms:performance.now(),kind:detect?'detection':typeof data==='string'?'command':'crop',
   video_timestamp:detect&&pendingUpload?pendingUpload.video_timestamp:null,bytes:data.size||0});
 return originalSend.call(this,data);
};
return true;
})()'''


def http_json(url):
    return json.load(urllib.request.urlopen(url, timeout=30))


def http_post(url):
    req = urllib.request.Request(url, data=b"", method="POST")
    return json.load(urllib.request.urlopen(req, timeout=30))


def log(line):
    with (RUNS / "driver-runs.jsonl").open("a") as f:
        f.write(json.dumps(line) + "\n")


async def fetch_diagnostics():
    """Side WebSocket after the page socket is closed; read-only status."""
    last = None
    try:
        async with websockets.connect("ws://127.0.0.1:8002/ws/detect",
                                      max_size=64 * 1024 * 1024) as ws:
            await ws.send(json.dumps({"type": "status", "diagnostics": True}))
            async for raw in ws:
                d = json.loads(raw)
                if d.get("type") == "status":
                    return d
                last = d
    except Exception as error:
        return {"error": str(error), "last": last}
    return {"error": "no-status-reply", "last": last}


async def ensure_page(c):
    """Fresh page per run: no wedged promise chains, no stale sockets/URLs."""
    await c.call("Page.navigate", url=SERVER + "/?diag=1")
    for _ in range(100):
        try:
            if await c.evaluate('typeof sourceSelect !== "undefined"'):
                break
        except Exception:
            pass
        await asyncio.sleep(0.1)
    await c.evaluate(OBSERVE)


async def one_replay(c, video, run_id, gate, measured=True, post_roll=POST_ROLL_S):
    directory = RUNS / run_id
    (directory / "crops").mkdir(parents=True, exist_ok=True)
    await ensure_page(c)
    GATE_FILE.write_text(gate + "\n")
    await asyncio.sleep(0.5)
    reset = http_post(SERVER + "/api/reset")
    dbg = http_json(SERVER + "/api/ident-debug")
    assert dbg["queue_depth"] == 0 and not dbg["tracks"], f"session not drained: {dbg}"
    ready_before = http_json(SERVER + "/api/ident-readiness")
    assert ready_before["jev_calls"] == 0, ready_before
    assert not ready_before["jev_key_present"], ready_before

    await c.evaluate('sourceSelect.value="video"; sourceSelect.dispatchEvent(new Event("change"));')
    await c.evaluate("sourceChange")
    doc = await c.call("DOM.getDocument")
    q = await c.call("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#videoFile")
    await c.call("DOM.setFileInputFiles", nodeId=q["nodeId"],
                 files=[str(COMPAT / video)])
    await c.evaluate("sourceChange")
    vw, rs = 0, 0
    for i in range(300):
        vw = await c.evaluate("camera.videoWidth")
        rs = await c.evaluate("camera.readyState")
        if vw and vw > 0 and rs >= 2:
            break
        if i % 50 == 0:
            print(run_id, "waiting for decode", vw, rs, flush=True)
        await asyncio.sleep(0.2)
    assert vw > 0, f"video decoded zero-size frame after 60s (readyState {rs})"
    session = http_json(SERVER + "/api/session")

    wall_start = time.monotonic()
    log({"event": "run_begin", "run": run_id, "gate": gate,
         "video": video, "wall_start": wall_start,
         "session_version": session["session_version"]})
    await c.evaluate('document.getElementById("startButton").click()')
    blog = (directory / "browser-events.jsonl").open("w")
    records, frames, saved = {}, {}, set()
    det_sent = det_recv = 0
    post = None
    last_state = {}
    cap = 600.0
    while time.monotonic() - wall_start < cap:
        sample = await c.evaluate('({browser_ms:performance.now(),video_time:camera.currentTime,'
                                  'duration:camera.duration,ended:camera.ended,'
                                  'status:statusText.textContent,'
                                  'events:comparisonEvents.splice(0),'
                                  'requests:evidenceRequests,frames:evidenceFrames,'
                                  'dropped:evidenceDropped,backendDropped})')
        for event in sample["events"]:
            blog.write(json.dumps(event, ensure_ascii=False) + "\n")
            if event.get("event") == "sent" and event.get("kind") == "detection":
                det_sent += 1
            if event.get("event") == "received":
                det_recv += 1
        for record in sample["requests"]:
            rec = dict(record)
            data = rec.pop("crop_jpeg", None)
            rec.pop("original_png", None)
            if data and rec["request_id"] not in saved:
                path = directory / "crops" / (rec["request_id"].replace(":", "_") + ".jpg")
                path.write_bytes(base64.b64decode(data.split(",", 1)[1]))
                saved.add(rec["request_id"])
                rec["crop_path"] = "crops/" + path.name
            records[rec["request_id"]] = rec
        for frame in sample["frames"]:
            frames[frame["frame_id"]] = frame
        last_state = {k: v for k, v in sample.items()
                      if k not in ("requests", "frames", "events")}
        if sample["ended"] and post is None:
            post = time.monotonic()
            log({"event": "video_ended", "run": run_id, "wall": post,
                 "video_time": sample["video_time"], "records": len(records)})
            print(run_id, "ended", round(sample["video_time"], 2), flush=True)
        if post and time.monotonic() - post > post_roll:
            break
        if "Cannot" in sample["status"] or "Unsupported" in sample["status"]:
            raise RuntimeError(f"replay failed: {sample['status']}")
        await asyncio.sleep(1)
    wall_end = time.monotonic()
    complete = post is not None and wall_end - post >= post_roll
    # Reload first: kills the page socket (side channel needs it gone) and
    # avoids teardown races with in-flight reset promises. Evidence was
    # already copied incrementally; server diagnostics survive reload.
    try:
        await c.call("Page.navigate", url="about:blank")
    except Exception:
        pass
    await asyncio.sleep(1.0)
    diagnostics = await fetch_diagnostics()
    debug_after = http_json(SERVER + "/api/ident-debug")
    ready_after = http_json(SERVER + "/api/ident-readiness")
    assert ready_after["jev_calls"] == 0, ready_after
    assert not ready_after["jev_key_present"], ready_after
    blog.close()
    (directory / "collected.json").write_text(json.dumps(
        {"frames": list(frames.values()), "requests": list(records.values()),
         "det_sent": det_sent, "det_recv": det_recv, "last_state": last_state},
        ensure_ascii=False, indent=1))
    (directory / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=1))
    (directory / "debug-after.json").write_text(json.dumps(debug_after, ensure_ascii=False, indent=1))
    (directory / "readiness-after.json").write_text(json.dumps(ready_after, ensure_ascii=False, indent=1))
    (directory / "run.json").write_text(json.dumps(
        {"run": run_id, "gate": gate, "video": video, "measured": measured,
         "wall_start": wall_start, "wall_end": wall_end, "complete": complete,
         "post_roll_s": post_roll if post else 0.0,
         "video_width": vw, "session_version": session["session_version"],
         "det_sent": det_sent, "det_recv": det_recv,
         "n_frames": len(frames), "n_requests": len(records), "n_crops_saved": len(saved),
         "readiness_before": ready_before}, indent=1))
    log({"event": "run_end", "run": run_id, "wall_end": wall_end,
         "complete": complete, "det_sent": det_sent, "det_recv": det_recv,
         "n_requests": len(records), "jev_calls": ready_after["jev_calls"]})
    print(run_id, "saved", len(frames), "frames", len(records), "requests",
          "complete" if complete else "INCOMPLETE", flush=True)
    return directory


def run_done(run_id):
    try:
        meta = json.loads((RUNS / run_id / "run.json").read_text())
        return bool(meta.get("complete"))
    except OSError:
        return False


async def main():
    RUNS.mkdir(exist_ok=True)
    tabs = json.load(urllib.request.urlopen(CDP_URL, timeout=10))
    target = next(t for t in tabs if t["type"] == "page")
    async with websockets.connect(target["webSocketDebuggerUrl"],
                                  max_size=100 * 1024 * 1024) as ws:
        c = CDP(ws)
        reader = asyncio.create_task(c.read())
        await c.call("Page.enable")
        await c.call("Runtime.enable")
        # Warmup: unmeasured V3 replay, short post-roll; models stay warm.
        if not run_done("warmup"):
            await one_replay(c, VIDEOS[2][1], "warmup", "B", measured=False, post_roll=6.0)
        for short, filename in VIDEOS:
            for gate, rep in [("A", 1), ("B", 1), ("B", 2), ("A", 2)]:
                run_id = f"{short}-{gate.lower()}{rep}"
                if run_done(run_id):
                    print(run_id, "already complete, skipping", flush=True)
                    continue
                await one_replay(c, filename, run_id, gate)
        try:
            await c.evaluate("stop()")
        except Exception:
            pass
        (RUNS / "browser-errors.json").write_text(json.dumps(c.events, indent=1))
        reader.cancel()


if __name__ == "__main__":
    asyncio.run(main())
