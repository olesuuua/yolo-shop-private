"""Model construction, named class transfer and verified custom checkpoints."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from ppyoloe_assets import CHECKPOINT_SHA256, CONFIG_PATH, PADDLEDETECTION_COMMIT

SOURCE_CLASSES = {"apple": "Apple", "bottle": "Bottle", "can": "Canned", "pack of crisps": "Chips"}


def valid_labels(labels):
    return (isinstance(labels, list) and bool(labels)
            and all(isinstance(label, str) and label.strip() == label and bool(label) for label in labels)
            and len(set(labels)) == len(labels))


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_manifest(path, verify_weights=True):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    labels = manifest.get("labels", [])
    if (manifest.get("schema_version") != 1 or manifest.get("architecture") != "ppyoloe_plus_crn_s"
            or not valid_labels(labels)
            or manifest.get("source_checkpoint_sha256") != CHECKPOINT_SHA256
            or manifest.get("paddledetection_commit") != PADDLEDETECTION_COMMIT):
        raise ValueError("Unsupported custom PP-YOLOE manifest or source identity.")
    filename = manifest.get("weights", "")
    size = manifest.get("input_size")
    if (not filename or Path(filename).name != filename or not filename.endswith(".pdparams")
            or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("checkpoint_sha256", ""))
            or not isinstance(size, int) or size < 128 or size % 32):
        raise ValueError("Invalid checkpoint filename, checksum or input size.")
    checkpoint = path.parent / filename
    if verify_weights and file_digest(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Custom checkpoint checksum mismatch.")
    return manifest, checkpoint


def build_model(root, num_classes, size):
    source = Path(root) / ".cache/PaddleDetection"
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != PADDLEDETECTION_COMMIT:
        raise ValueError("PaddleDetection source does not match the pinned revision.")
    sys.path.insert(0, str(source))
    import paddle
    from ppdet.core.workspace import load_config, merge_config, create
    if paddle.__version__ != "3.3.1":
        raise RuntimeError("Use the prepared Paddle 3.3.1 Python environment.")
    paddle.set_device("cpu")
    cfg = load_config(str(source / CONFIG_PATH))
    merge_config({"num_classes": num_classes, "eval_size": [size, size], "norm_type": "bn",
                  "PPYOLOEHead": {"static_assigner_epoch": 0}})
    return create(cfg.architecture)


def transfer_state(source, target, class_ids):
    """Transfer Objects365 rows; None leaves that new classifier row initialized."""
    if set(source) != set(target):
        raise ValueError("Pretrained and target checkpoint keys do not match.")
    transferred, remapped = {}, []
    for key, expected in target.items():
        value = source[key]
        if re.fullmatch(r"yolo_head\.pred_cls\.[012]\.(weight|bias)", key):
            if value.shape[0] != 365:
                raise ValueError("Expected a 365-class pretrained classifier.")
            value = remap_classifier(value, expected, class_ids)
            remapped.append(key)
        if list(value.shape) != list(expected.shape):
            raise ValueError(f"Unexpected pretrained tensor shape: {key}")
        transferred[key] = value
    if len(remapped) != 6:
        raise ValueError("Expected all six classifier tensors to be remapped.")
    return transferred, remapped


def remap_classifier(source, target, indices):
    if target.shape[0] != len(indices) or list(source.shape[1:]) != list(target.shape[1:]):
        raise ValueError("Classifier dimensions do not match the class mapping.")
    result = target.clone() if hasattr(target, "clone") else target.copy()
    for destination, origin in enumerate(indices):
        if origin is not None:
            if not isinstance(origin, int) or not 0 <= origin < source.shape[0]:
                raise ValueError("Invalid source class index.")
            result[destination] = source[origin]
    return result


def continue_state(source, target, source_labels, target_labels):
    """Keep learned features and matching class rows; initialize genuinely new classes."""
    if not valid_labels(source_labels) or not valid_labels(target_labels):
        raise ValueError("Checkpoint classes must be unique nonempty names.")
    if set(source) != set(target):
        raise ValueError("Parent and target checkpoint keys do not match.")
    indices = [source_labels.index(label) if label in source_labels else None for label in target_labels]
    transferred, remapped = {}, []
    for key, expected in target.items():
        value = source[key]
        if re.fullmatch(r"yolo_head\.pred_cls\.[012]\.(weight|bias)", key):
            if value.shape[0] != len(source_labels):
                raise ValueError("Parent classifier dimensions differ from its manifest.")
            value = remap_classifier(value, expected, indices)
            remapped.append(key)
        if list(value.shape) != list(expected.shape):
            raise ValueError(f"Unexpected parent tensor shape: {key}")
        transferred[key] = value
    if len(remapped) != 6:
        raise ValueError("Expected all six classifier tensors to be remapped.")
    return transferred, remapped, indices
