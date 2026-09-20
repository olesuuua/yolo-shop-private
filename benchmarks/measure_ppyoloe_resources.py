"""Measure a separate, temporary LightStore server; leave the live session alone."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time

import cv2
import httpx
import psutil
import websockets

ROOT = Path(__file__).resolve().parents[1]


def cpu_seconds(process):
    values = process.cpu_times()
    return values.user + values.system


def percentile(values, q):
    return float(__import__("numpy").percentile(values, q))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, action="append", required=True,
                        help="Local image to repeat during the benchmark; may be supplied multiple times.")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/ppyoloe-resources.json")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to benchmark an existing server on this port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    paths = args.image
    packets = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.resize(image, (640, 480))
        packets.append(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())

    env = os.environ.copy()
    env.update({"LIGHTSTORE_MODEL": "ppyoloe_objects365", "LIGHTSTORE_IMGSZ": "640",
                "YOLO_CONFIG_DIR": str(ROOT / ".cache/ultralytics"),
                "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib"), "YOLO_AUTOINSTALL": "false"})
    samples = []
    processes = {}
    phase = "startup"
    stop = threading.Event()
    start = time.perf_counter()
    log_path = ROOT / ".cache/resource-benchmark-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
                                   "--port", str(args.port), "--log-level", "warning"],
                                  cwd=ROOT, env=env, stdout=log, stderr=log)
    parent = psutil.Process(server.pid)

    def monitor():
        while not stop.wait(.05):
            try:
                processes["server"] = parent
                for child in parent.children(recursive=True):
                    if any("ppyoloe_worker.py" in arg for arg in child.cmdline()):
                        processes["paddle"] = child
                row = {"t": time.perf_counter() - start, "phase": phase, "processes": {}}
                for name, process in tuple(processes.items()):
                    try:
                        row["processes"][name] = {"pid": process.pid,
                            "rss_bytes": process.memory_info().rss,
                            "cpu_seconds": cpu_seconds(process), "threads": process.num_threads()}
                    except psutil.NoSuchProcess:
                        continue
                samples.append(row)
            except psutil.NoSuchProcess:
                break

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    base = f"http://127.0.0.1:{args.port}"
    report = None
    try:
        with httpx.Client(timeout=1) as client:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Benchmark server stopped; inspect {log_path}")
                try:
                    session = client.get(base + "/api/session").raise_for_status().json()
                    break
                except (httpx.ConnectError, httpx.ReadTimeout):
                    time.sleep(.1)
            else:
                raise TimeoutError("Benchmark server did not become ready.")
        startup_seconds = time.perf_counter() - start
        assert session["model_profile"] == "ppyoloe_objects365" and session["inference_device"] == "cpu"
        phase = "idle"
        time.sleep(3)

        async def exercise():
            nonlocal phase
            async with websockets.connect(base.replace("http://", "ws://") + "/ws/detect",
                                          max_size=4_000_000) as ws:
                phase = "warmup"
                for _ in range(5):
                    await ws.send(packets[0])
                    reply = json.loads(await ws.recv())
                    if "error" in reply:
                        raise RuntimeError(reply["error"])
                phase = "load"
                initial_cpu = {name: cpu_seconds(p) for name, p in processes.items()}
                begin = time.perf_counter()
                durations = []
                request_bytes = response_bytes = 0
                count = 0
                while time.perf_counter() - begin < args.seconds:
                    # Hold each scene for several seconds to avoid a scene cut on every frame.
                    packet = packets[(count // 40) % len(packets)]
                    frame_start = time.perf_counter()
                    await ws.send(packet)
                    response = await ws.recv()
                    reply = json.loads(response)
                    durations.append((time.perf_counter() - frame_start) * 1000)
                    if "error" in reply:
                        raise RuntimeError(reply["error"])
                    request_bytes += len(packet)
                    response_bytes += len(response.encode())
                    count += 1
                elapsed = time.perf_counter() - begin
                used_cpu = {name: (cpu_seconds(p) - initial_cpu[name]) / elapsed * 100
                            for name, p in processes.items()}
                return {"frames": count, "seconds": elapsed, "fps": count / elapsed,
                        "latency_median_ms": statistics.median(durations),
                        "latency_p95_ms": percentile(durations, 95),
                        "cpu_percent_one_core_100": used_cpu,
                        "cpu_ms_per_frame": {name: percent / 100 * elapsed / count * 1000
                                             for name, percent in used_cpu.items()},
                        "request_kib_per_frame": request_bytes / count / 1024,
                        "response_kib_per_frame": response_bytes / count / 1024,
                        "payload_mbit_per_second": (request_bytes + response_bytes) * 8 / elapsed / 1e6}

        workload = asyncio.run(exercise())
        phase = "after_load"
        time.sleep(2)
        summary = {}
        for name in ["startup", "idle", "warmup", "load", "after_load"]:
            selected = [r for r in samples if r["phase"] == name]
            if not selected:
                continue
            roles = {role for row in selected for role in row["processes"]}
            totals = [sum(p["rss_bytes"] for p in row["processes"].values()) for row in selected]
            summary[name] = {"rss_sum_peak_mib": max(totals) / 2**20,
                             "rss_sum_median_mib": statistics.median(totals) / 2**20,
                             "processes": {}}
            for role in roles:
                values = [r["processes"][role] for r in selected if role in r["processes"]]
                rss = [p["rss_bytes"] for p in values]
                summary[name]["processes"][role] = {"rss_peak_mib": max(rss) / 2**20,
                    "rss_median_mib": statistics.median(rss) / 2**20,
                    "threads_peak": max(p["threads"] for p in values)}
        report = {"measured_at_utc": datetime.now(timezone.utc).isoformat(),
                  "hardware": {"chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                               "logical_cpus": psutil.cpu_count(), "ram_gib": psutil.virtual_memory().total / 2**30},
                  "configuration": {"profile": session["model_profile"], "classes": len(session["enabled_classes"]),
                                    "input_size": session["inference_size"], "frame_size": [640,480], "device": "cpu"},
                  "checkpoint_bytes": (ROOT / "weights/ppyoloe-objects365/ppyoloe_crn_s_obj365_pretrained.pdparams").stat().st_size,
                  "startup_seconds": startup_seconds, "memory": summary, "workload": workload,
                  "limitations": ["Mac measurements, not Raspberry Pi measurements.",
                    "RSS sums can double-count shared pages. USS could not be read on this Mac.",
                    "Excludes browser, camera capture, OS and benchmark driver memory.",
                    "Startup peak sampled every 50 ms; shorter peaks may be missed.",
                    f"Repeats {len(packets)} JPEG(s); throughput is not a detection-accuracy result."],
                  "samples": samples}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k:v for k,v in report.items() if k != "samples"}, indent=2), flush=True)
    finally:
        stop.set()
        monitor_thread.join(timeout=2)
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


if __name__ == "__main__":
    main()
