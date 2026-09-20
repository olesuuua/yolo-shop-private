"""Paddle 3.3.1 CPU/CUDA inference in its own Python environment."""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

from ppyoloe_assets import CHECKPOINT_SHA256, LABELS_SHA256, PADDLEDETECTION_COMMIT, CONFIG_PATH
from ppyoloe_protocol import receive, send

ROOT = Path(__file__).resolve().parent


def checked_file(path, digest):
    with path.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {path.name}")
    return path


def load_detector(size, manifest_path=None):
    source = ROOT / ".cache/PaddleDetection"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != PADDLEDETECTION_COMMIT:
        raise ValueError("PaddleDetection revision does not match the pinned source.")
    sys.path.insert(0, str(source))
    import numpy as np
    import paddle
    from ppdet.core.workspace import load_config, merge_config, create
    from ppdet.modeling.bbox_utils import batch_distance2bbox
    from ppyoloe_postprocess import preprocess, select_candidates

    if paddle.__version__ != "3.3.1":
        raise RuntimeError("This worker requires the tested paddlepaddle==3.3.1 environment.")
    device = os.environ.get("LIGHTSTORE_DEVICE", "cpu")
    if device == "gpu:0" and not paddle.is_compiled_with_cuda():
        raise RuntimeError("GPU requested but Paddle was installed without CUDA.")
    paddle.set_device(device)
    if manifest_path:
        from ppyoloe_checkpoint import read_manifest
        manifest, checkpoint = read_manifest(manifest_path)
        labels, expected_sha = manifest["labels"], manifest["checkpoint_sha256"]
    else:
        assets = ROOT / "weights/ppyoloe-objects365"
        checkpoint = checked_file(assets / "ppyoloe_crn_s_obj365_pretrained.pdparams", CHECKPOINT_SHA256)
        label_file = checked_file(assets / "objects365_labels.txt", LABELS_SHA256)
        labels = [label.strip() for label in label_file.read_text().splitlines()]
        catalog = json.loads((ROOT / "objects365_classes.json").read_text())
        if len(labels) != 365 or labels != catalog["labels"]:
            raise ValueError("Objects365 class IDs differ from the verified checkpoint catalog.")
        expected_sha = CHECKPOINT_SHA256
    cfg = load_config(str(source / CONFIG_PATH))
    merge_config({"num_classes": len(labels), "eval_size": [size, size]})
    model = create(cfg.architecture)
    weights = paddle.load(str(checkpoint))
    state = model.state_dict()
    if set(state) != set(weights) or any(list(weights[k].shape) != list(v.shape) for k, v in state.items()):
        raise ValueError("Checkpoint parameters do not exactly match PP-YOLOE+ Small and its declared classes.")
    model.set_state_dict(weights)
    model.eval()
    if model.yolo_head.num_classes != len(labels):
        raise ValueError("The detector head does not match its checkpoint labels.")
    for layer in model.sublayers():
        if hasattr(layer, "convert_to_deploy"):
            layer.convert_to_deploy()

    @paddle.no_grad()
    def predict(frame, threshold):
        tensor = paddle.to_tensor(preprocess(frame, size))
        features = model.neck(model.backbone({"image": tensor}))
        scores, distances, anchors, strides = model.yolo_head(features)
        boxes = batch_distance2bbox(anchors, distances) * strides
        return select_candidates(boxes.numpy(), scores.numpy(), threshold, num_classes=len(labels))

    predict(np.zeros((480, 640, 3), dtype=np.uint8), 1.0)
    metadata = {"type": "ready", "labels": labels, "device": paddle.get_device(),
                "size": size, "paddle_version": paddle.__version__,
                "checkpoint_sha256": expected_sha, "checkpoint_tensors": len(weights)}
    return predict, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=.1)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    with socket.socket(fileno=args.fd) as connection:
        try:
            predict, metadata = load_detector(args.size, args.manifest)
            send(connection, json.dumps(metadata).encode())
            import numpy as np
            while True:
                try:
                    payload = receive(connection)
                except EOFError:
                    break
                frame = np.load(BytesIO(payload), allow_pickle=False)
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError("Expected a uint8 BGR frame.")
                detections = predict(frame, args.conf)
                send(connection, json.dumps({"type": "detections", "rows": detections.tolist()},
                                             allow_nan=False).encode())
        except Exception as error:
            traceback.print_exc()
            try:
                send(connection, json.dumps({"type": "error", "message": str(error)}).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
            raise


if __name__ == "__main__":
    main()
