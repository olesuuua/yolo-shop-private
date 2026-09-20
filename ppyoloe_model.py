"""Private Paddle worker feeding the existing Ultralytics ByteTrack adapter."""

from io import BytesIO
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import threading

import numpy as np

from ppyoloe_assets import CHECKPOINT_SHA256
from ppyoloe_postprocess import postprocess
from ppyoloe_protocol import receive, send

logger = logging.getLogger(__name__)


class PaddleWorker:
    def __init__(self, root, size, confidence, manifest=None):
        python = Path(os.environ.get("LIGHTSTORE_PADDLE_PYTHON", root / ".venv-ppyolo-export/bin/python"))
        if not python.is_file():
            raise FileNotFoundError("Paddle worker environment is missing; see the PP-YOLOE setup in README.md.")
        self.process = None
        self.lock = threading.Lock()
        self.connection, child = socket.socketpair()
        self.connection.settimeout(120)
        self.log_path = root / ".cache/ppyoloe-worker.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4",
                            "OPENBLAS_NUM_THREADS": "4", "VECLIB_MAXIMUM_THREADS": "4",
                            "GLOG_minloglevel": "2", "MPLCONFIGDIR": str(root / ".cache/matplotlib")})
        try:
            command = [str(python), str(root / "ppyoloe_worker.py"), "--fd", str(child.fileno()),
                       "--size", str(size), "--conf", str(confidence)]
            if manifest is not None:
                command.extend(["--manifest", str(manifest)])
            with self.log_path.open("a") as log:
                self.process = subprocess.Popen(
                    command,
                    cwd=root, env=environment, pass_fds=(child.fileno(),),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                )
            child.close()
            self.metadata = self._response("ready")
            self.connection.settimeout(30)
        except BaseException:
            child.close()
            self.close()
            raise

    def _response(self, expected_type):
        try:
            result = json.loads(receive(self.connection))
            if result.get("type") != expected_type:
                raise ValueError(result.get("message", "Unexpected Paddle worker response."))
            return result
        except (OSError, EOFError, ValueError) as error:
            raise RuntimeError(f"Paddle worker failed: {error}. Log: {self.log_path}") from error

    def predict(self, frame):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError(f"Paddle worker is not running. Log: {self.log_path}")
            payload = BytesIO()
            np.save(payload, frame, allow_pickle=False)
            try:
                send(self.connection, payload.getvalue())
                return np.asarray(self._response("detections")["rows"], dtype=np.float32).reshape(-1, 6)
            except Exception:
                # Discard the stream after failure so a delayed reply cannot become the next frame.
                self.close()
                raise

    def close(self):
        self.connection.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            self.process = None


class PPYOLOEModel:
    def __init__(self, worker, labels, size):
        self.worker = worker
        self.names = dict(enumerate(labels))
        self.size = size
        self.tracker = None
        self.tracker_config = None

    def close(self):
        self.worker.close()

    def track(self, frame, *, persist, tracker, classes, conf, imgsz, device,
              agnostic_nms, iou, nms, verbose=False):
        import torch
        from ultralytics.engine.results import Boxes, Results
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML

        if device not in {"cpu", "gpu:0"} or imgsz != self.size or not nms:
            raise ValueError("PP-YOLOE requires cpu or gpu:0, its configured input size and NMS.")
        if self.tracker is None or not persist or self.tracker_config != tracker:
            self.tracker = BYTETracker(IterableSimpleNamespace(**YAML.load(tracker)))
            self.tracker_config = tracker
        rows = self.worker.predict(frame)
        detections = postprocess(rows, frame.shape, self.size, classes, conf, iou, agnostic_nms)
        tracks = self.tracker.update(Boxes(detections, frame.shape[:2]), frame)
        tracked_boxes = np.asarray(tracks, dtype=np.float32).reshape(-1, 8)[:, :7]
        return [Results(frame, path="", names=self.names, boxes=torch.from_numpy(tracked_boxes))]


def load_ppyoloe(root: Path):
    from config import FOOD_CLASSES, INFERENCE_SIZE, CONF_THRESHOLD, PPYOLOE_MANIFEST, INFERENCE_DEVICE

    if PPYOLOE_MANIFEST:
        from ppyoloe_checkpoint import read_manifest
        manifest, _ = read_manifest(PPYOLOE_MANIFEST)
        labels, expected_sha = manifest["labels"], manifest["checkpoint_sha256"]
        if set(FOOD_CLASSES) != set(labels):
            raise ValueError("The custom class list does not match the checkpoint.")
        worker = PaddleWorker(root, INFERENCE_SIZE, CONF_THRESHOLD, manifest=PPYOLOE_MANIFEST)
    else:
        labels = json.loads((root / "objects365_classes.json").read_text())["labels"]
        if len(labels) != 365 or not FOOD_CLASSES.issubset(labels):
            raise ValueError("The Objects365 catalog does not match selected grocery classes.")
        expected_sha = CHECKPOINT_SHA256
        worker = PaddleWorker(root, INFERENCE_SIZE, CONF_THRESHOLD)
    metadata = worker.metadata
    if (metadata.get("labels") != labels or metadata.get("device") != INFERENCE_DEVICE
            or metadata.get("size") != INFERENCE_SIZE
            or metadata.get("checkpoint_sha256") != expected_sha):
        worker.close()
        raise ValueError("Paddle worker loaded the wrong model, class order, size or device.")
    logger.info("PP-YOLOE+ Small ready: Paddle %s, %s, %d enabled classes",
                metadata["paddle_version"], INFERENCE_DEVICE, len(FOOD_CLASSES))
    return PPYOLOEModel(worker, labels, INFERENCE_SIZE)
