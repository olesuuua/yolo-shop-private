"""Start the published six-class model with paths resolved from this checkout."""

import argparse
import base64
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "models/grocery6-overhead/best.json"


def runtime_environment():
    environment = os.environ.copy()
    environment.update({
        "LIGHTSTORE_MODEL": "ppyoloe_custom",
        "LIGHTSTORE_PPYOLOE_MANIFEST": str(MANIFEST),
        "LIGHTSTORE_PADDLE_PYTHON": str(ROOT / ".venv-ppyolo-export/bin/python"),
        "LIGHTSTORE_IMGSZ": "416",
        "YOLO_CONFIG_DIR": str(ROOT / ".cache/ultralytics"),
        "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib"),
        "YOLO_AUTOINSTALL": "false",
        "PYTHONNOUSERSITE": "1",
    })
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def check_inference():
    # Use a generated frame: no training data or developer camera files needed.
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
    from app import app
    from ppyoloe_checkpoint import read_manifest

    manifest, _ = read_manifest(MANIFEST)
    ok, jpeg = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
    if not ok:
        raise RuntimeError("Could not encode the smoke-test frame.")
    started = time.perf_counter()
    with TestClient(app) as client:
        home = client.get("/")
        home.raise_for_status()
        response = client.get("/api/session")
        response.raise_for_status()
        state = response.json()
        if (state["enabled_classes"] != manifest["labels"] or state["inference_device"] != os.environ.get("LIGHTSTORE_DEVICE", "cpu")
                or state["inference_size"] != 416 or state["model_profile"] != "ppyoloe_custom"):
            raise RuntimeError("The loaded model differs from the published six-class configuration.")
        with client.websocket_connect("/ws/detect") as camera:
            for _ in range(3):
                camera.send_bytes(jpeg.tobytes())
                frame = camera.receive_json()
                if "error" in frame:
                    raise RuntimeError(frame["error"])
                decoded = cv2.imdecode(np.frombuffer(base64.b64decode(frame["image"]), np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape != (480, 640, 3) or frame["packed_total"] != 0:
                    raise RuntimeError("Camera pipeline returned an invalid smoke-test result.")
    report = {"platform": platform.platform(), "python": platform.python_version(),
              "checkpoint_sha256": manifest["checkpoint_sha256"],
              "labels": state["enabled_classes"], "device": state["inference_device"], "input_size": 416,
              "frames_processed": 3, "elapsed_seconds": round(time.perf_counter() - started, 3),
              "http_and_websocket": "passed"}
    (ROOT / ".cache/runtime-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--check", action="store_true", help="Verify real HTTP/WebSocket CPU inference, then exit")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    python = ROOT / ".venv/bin/python"
    if not python.is_file() or not (ROOT / ".venv-ppyolo-export/bin/python").is_file():
        parser.error("Run python3.14 setup_lightstore.py first.")
    environment = runtime_environment()
    # Re-exec once, even if invoked by a system Python or from another directory.
    if Path(sys.prefix) != ROOT / ".venv" or any(
        os.environ.get(key) != environment[key] for key in
        ("LIGHTSTORE_MODEL", "LIGHTSTORE_PPYOLOE_MANIFEST", "LIGHTSTORE_PADDLE_PYTHON", "LIGHTSTORE_IMGSZ")
    ) or os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    os.environ.update(environment)
    os.chdir(ROOT)
    from prepare_grocery6 import main as prepare
    prepare()
    if args.check:
        check_inference()
    else:
        print(f"LightStore six-class {os.environ.get('LIGHTSTORE_DEVICE', 'cpu')} model: http://127.0.0.1:{args.port}/", flush=True)
        os.execve(str(python), [str(python), "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
                              "--port", str(args.port)], environment)


if __name__ == "__main__":
    main()
