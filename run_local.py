"""Launch the original PP-YOLOE+ Small Objects365 model, loading credentials only in the backend."""
import argparse
import base64
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent


def base_environment():
    """Environment shared by the pre-exec hop and the final server process."""
    environment = os.environ.copy()
    environment.update({
        "LIGHTSTORE_MODEL": "ppyoloe_objects365",
        "LIGHTSTORE_PADDLE_PYTHON": str(ROOT / ".venv-ppyolo-gpu/bin/python"),
        "YOLO_CONFIG_DIR": str(ROOT / ".cache/ultralytics"),
        "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib"),
        "YOLO_AUTOINSTALL": "false",
        "PYTHONNOUSERSITE": "1",
    })
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    # Six-class leftovers must not leak into the original profile.
    environment.pop("LIGHTSTORE_PPYOLOE_MANIFEST", None)
    return environment


def runtime_environment():
    environment = base_environment()
    # Process env wins, then .env (loaded in main), then GPU by default.
    environment["LIGHTSTORE_DEVICE"] = os.environ.get("LIGHTSTORE_DEVICE", "gpu:0")
    return environment


def check_inference():
    # Use a generated frame: no training data or camera files needed.
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
    from app import app

    expected_device = os.environ.get("LIGHTSTORE_DEVICE", "gpu:0")
    expected_size = int(os.environ.get("LIGHTSTORE_IMGSZ", "640"))
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
        if (state["model_profile"] != "ppyoloe_objects365"
                or state["inference_device"] != expected_device
                or state["inference_size"] != expected_size):
            raise RuntimeError(f"Unexpected loaded model: {state}")
        with client.websocket_connect("/ws/detect") as camera:
            camera.send_bytes(jpeg.tobytes())
            frame = camera.receive_json()
            if "error" in frame:
                raise RuntimeError(frame["error"])
            decoded = cv2.imdecode(np.frombuffer(base64.b64decode(frame["image"]), np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape != (480, 640, 3):
                raise RuntimeError("Camera pipeline returned an invalid smoke-test result.")
    elapsed = round(time.perf_counter() - started, 3)
    print(f"Original Objects365 check passed: {state['model_label']} "
          f"on {expected_device} at {expected_size}px in {elapsed}s.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--check", action="store_true", help="Verify real HTTP/WebSocket GPU inference, then exit")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    python = ROOT / ".venv/bin/python"
    if not python.is_file() or not (ROOT / ".venv-ppyolo-gpu/bin/python").is_file():
        parser.error("Missing environments: .venv and .venv-ppyolo-gpu are required.")
    # Re-exec into .venv first; .env and the final device selection resolve there.
    if Path(sys.prefix) != ROOT / ".venv":
        environment = base_environment()
        os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    environment = runtime_environment()
    if any(os.environ.get(key) != environment[key] for key in
           ("LIGHTSTORE_MODEL", "LIGHTSTORE_DEVICE", "LIGHTSTORE_PADDLE_PYTHON")) \
            or os.environ.get("PYTHONPATH") or os.environ.get("PYTHONHOME"):
        os.execve(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    os.environ.update(environment)
    os.chdir(ROOT)
    from prepare_ppyoloe import main as prepare
    prepare()
    if args.check:
        check_inference()
    else:
        print(f"LightStore original Objects365 {os.environ.get('LIGHTSTORE_DEVICE', 'gpu:0')} model: "
              f"http://127.0.0.1:{args.port}/", flush=True)
        os.execve(str(python), [str(python), "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
                                "--port", str(args.port)], environment)


if __name__ == "__main__":
    main()
