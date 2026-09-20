"""CPU fine-tuning using PaddleDetection's PP-YOLOE head, assigner and losses."""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

import cv2
import numpy as np

from ppyoloe_assets import CHECKPOINT_SHA256, PADDLEDETECTION_COMMIT
from ppyoloe_checkpoint import (SOURCE_CLASSES, build_model, file_digest, read_manifest,
                               transfer_state, continue_state, valid_labels)
from ppyoloe_postprocess import preprocess, select_candidates

ROOT = Path(__file__).resolve().parent


class CocoImages:
    def __init__(self, root, split, prefix=None):
        self.root = Path(root) / split / "images"
        self.annotation = Path(root) / "annotations" / f"{split}.json"
        self.coco = json.loads(self.annotation.read_text())
        self.images = [row for row in self.coco["images"] if prefix is None or row["file_name"].startswith(prefix)]
        if not self.images:
            raise ValueError(f"No images in {split} for prefix {prefix!r}.")
        self.annotations = defaultdict(list)
        for row in self.coco["annotations"]:
            self.annotations[row["image_id"]].append(row)

    def read(self, index):
        row = self.images[index]
        image = cv2.imread(str(self.root / row["file_name"]))
        if image is None:
            raise ValueError(f"Cannot decode image: {row['file_name']}")
        return image, row, self.annotations[row["id"]]


def training_example(image, annotations, size, rng=None):
    height, width = image.shape[:2]
    boxes, classes = [], []
    for row in annotations:
        x, y, w, h = row["bbox"]
        boxes.append([x/width, y/height, (x+w)/width, (y+h)/height])
        classes.append(row["category_id"]-1)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if rng is not None:
        if rng.random() < .5:
            image = np.ascontiguousarray(image[:, ::-1])
            boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
        image = np.clip(image.astype(np.float32)*rng.uniform(.85, 1.15), 0, 255).astype(np.uint8)
    return preprocess(image, size)[0], boxes*size, np.asarray(classes, dtype=np.int64)


def make_batch(dataset, indices, size, rng, epoch):
    import paddle
    rows = [training_example(image, annotations, size, rng)
            for image, _, annotations in (dataset.read(i) for i in indices)]
    # Keep one padded slot even for an all-background batch.
    maximum = max(1, max(len(row[1]) for row in rows))
    boxes = np.zeros((len(rows), maximum, 4), np.float32)
    labels = np.zeros((len(rows), maximum, 1), np.int64)
    masks = np.zeros((len(rows), maximum, 1), np.float32)
    for i, (_, b, c) in enumerate(rows):
        boxes[i, :len(b)] = b; labels[i, :len(c), 0] = c; masks[i, :len(b)] = 1
    return {"image": paddle.to_tensor(np.stack([r[0] for r in rows])),
            "gt_bbox": paddle.to_tensor(boxes), "gt_class": paddle.to_tensor(labels),
            "pad_gt_mask": paddle.to_tensor(masks), "epoch_id": epoch}


def numpy_nms(rows, shape, size, iou=.45):
    """Class-agnostic NMS, same thresholds/top-k as the live ByteTrack adapter."""
    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6).copy()
    h, w = shape[:2]
    rows[:, [0, 2]] = np.clip(rows[:, [0, 2]]*w/size, 0, w)
    rows[:, [1, 3]] = np.clip(rows[:, [1, 3]]*h/size, 0, h)
    rows = rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]
    order = np.argsort(-rows[:, 4], kind="stable")[:1000]
    keep = []
    while len(order) and len(keep) < 300:
        first = order[0]; keep.append(first); order = order[1:]
        a, b = rows[first, :4], rows[order, :4]
        intersection = np.maximum(0, np.minimum(a[2:], b[:, 2:])-np.maximum(a[:2], b[:, :2])).prod(1)
        union = (a[2:]-a[:2]).prod()+(b[:, 2:]-b[:, :2]).prod(1)-intersection
        order = order[intersection/np.maximum(union, 1e-9) <= iou]
    return rows[keep]


def predict(model, image, size):
    import paddle
    from ppdet.modeling.bbox_utils import batch_distance2bbox
    model.eval()
    with paddle.no_grad():
        features = model.neck(model.backbone({"image": paddle.to_tensor(preprocess(image, size))}))
        scores, distances, anchors, strides = model.yolo_head(features)
        boxes = batch_distance2bbox(anchors, distances)*strides
        rows = select_candidates(boxes.numpy(), scores.numpy(), .001, num_classes=model.yolo_head.num_classes)
    return numpy_nms(rows, image.shape, size)


def evaluate(model, dataset, size, labels):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    detections = []
    for i in range(len(dataset.images)):
        image, info, _ = dataset.read(i)
        for x1, y1, x2, y2, score, cls in predict(model, image, size):
            detections.append({"image_id": info["id"], "category_id": int(cls)+1,
                               "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                               "score": float(score)})
    with redirect_stdout(StringIO()):
        ground_truth = COCO(str(dataset.annotation))
        if detections:
            result = ground_truth.loadRes(detections)
        else:
            result = COCO()
            result.dataset = {**dataset.coco, "annotations": []}; result.createIndex()
        evaluator = COCOeval(ground_truth, result, "bbox")
        evaluator.params.imgIds = [image["id"] for image in dataset.images]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    per_class = {}
    for index, label in enumerate(labels):
        precision = evaluator.eval["precision"][:, :, index, 0, -1]
        valid = precision[precision >= 0]
        per_class[label] = float(valid.mean()) if len(valid) else None
    return {"map50_95": float(evaluator.stats[0]), "map50": float(evaluator.stats[1]),
            "per_class_map50_95": per_class, "images": len(dataset.images),
            "prediction_boxes": len(detections)}


def set_training_mode(model, freeze):
    import paddle
    model.train()
    if freeze in {"head", "backbone"}:
        model.backbone.eval()
    if freeze == "head":
        model.neck.eval()
    # Tiny batches should not replace pretrained BatchNorm running statistics.
    for layer in model.sublayers():
        if isinstance(layer, (paddle.nn.BatchNorm2D, paddle.nn.SyncBatchNorm)):
            layer.eval()


def state_hashes(model):
    return {key: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for key, value in model.state_dict().items()}


def save_checkpoint(model, output, name, common, epoch, steps, metrics):
    import paddle
    path = output / f"{name}.pdparams"
    partial = path.with_suffix(".pdparams.tmp")
    paddle.save(model.state_dict(), str(partial)); partial.replace(path)
    manifest = {**common, "weights": path.name, "checkpoint_sha256": file_digest(path),
                "epoch": epoch, "optimizer_steps": steps, "validation": metrics}
    (output / f"{name}.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--init-manifest", type=Path)
    parser.add_argument("--eval-prefix", help="Also report metrics for filenames starting with this prefix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze", choices=("head", "backbone", "none"), default="head")
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.imgsz < 128 or args.imgsz % 32 or not 0 < args.lr < 1 or args.patience < 1:
        parser.error("Use positive epochs/batch/patience/lr, and imgsz >=128 divisible by 32.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.data / "dataset.json").read_text())
    labels = metadata["labels"]
    if not valid_labels(labels):
        raise ValueError("Dataset classes must be unique nonempty names.")
    for relative, digest in metadata["files"].items():
        if file_digest(args.data / relative) != digest:
            raise ValueError(f"Dataset changed after preparation: {relative}")
    source_path = ROOT / "weights/ppyoloe-objects365/ppyoloe_crn_s_obj365_pretrained.pdparams"
    if file_digest(source_path) != CHECKPOINT_SHA256:
        raise ValueError("Original checkpoint checksum mismatch.")
    model = build_model(ROOT, len(labels), args.imgsz)
    import paddle
    paddle.seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rng = random.Random(args.seed)
    catalog = json.loads((ROOT / "objects365_classes.json").read_text())["labels"]
    source_names = [SOURCE_CLASSES.get(label, label if label in catalog else None) for label in labels]
    class_ids = [catalog.index(name) if name is not None else None for name in source_names]
    parent = None
    if args.init_manifest:
        parent, parent_path = read_manifest(args.init_manifest)
        source = paddle.load(str(parent_path))
        state, remapped, parent_ids = continue_state(source, model.state_dict(), parent["labels"], labels)
        initialization = {label: {"method": "parent" if idx is not None else "new",
                                   "parent_class_id": idx} for label, idx in zip(labels, parent_ids)}
    else:
        source = paddle.load(str(source_path))
        state, remapped = transfer_state(source, model.state_dict(), class_ids)
        initialization = {label: {"method": "objects365" if idx is not None else "new",
                                   "source_class_id": idx} for label, idx in zip(labels, class_ids)}
    model.set_state_dict(state)
    # An explicit audit proves all non-classifier tensors and all selected rows were loaded.
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), state[key].numpy(), err_msg=key)
    initial_hashes = state_hashes(model)
    del source, state
    frozen_prefixes = {"head": ("backbone.", "neck."), "backbone": ("backbone.",), "none": ()}[args.freeze]
    for name, parameter in model.named_parameters():
        if name.startswith(frozen_prefixes):
            parameter.stop_gradient = True
    trainable = [p for p in model.parameters() if not p.stop_gradient]
    optimizer = paddle.optimizer.AdamW(learning_rate=args.lr, parameters=trainable, weight_decay=1e-4,
                                      grad_clip=paddle.nn.ClipGradByGlobalNorm(10.0))
    common = {"schema_version": 1, "architecture": "ppyoloe_plus_crn_s", "labels": labels,
              "source_checkpoint_sha256": CHECKPOINT_SHA256, "paddledetection_commit": PADDLEDETECTION_COMMIT,
              "paddle_version": paddle.__version__, "input_size": args.imgsz,
              "dataset_sha256": metadata["archive_sha256"], "dataset_warnings": metadata["warnings"],
              "source_class_ids": class_ids, "source_classes": source_names,
              "parent_checkpoint_sha256": parent["checkpoint_sha256"] if parent else None,
              "parent_labels": parent["labels"] if parent else None,
              "class_initialization": initialization,
              "evaluation_prefix": args.eval_prefix,
              "purpose": "smoke" if args.smoke else "finetune"}
    config = {**common, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
              "freeze": args.freeze, "batchnorm_running_stats": "frozen", "seed": args.seed,
              "patience": args.patience, "tensors_loaded": len(initial_hashes), "remapped_tensors": remapped,
              "trainable_parameters": sum(int(np.prod(p.shape)) for p in trainable),
              "dataset_splits": metadata["splits"], "validation_nms_iou": .45,
              "validation_confidence": .001, "test_used_for_selection": False}
    (output / "run.json").write_text(json.dumps(config, indent=2)+"\n")
    print(json.dumps({"event": "initialized", "tensors_loaded": len(initial_hashes),
                      "source_class_ids": class_ids, "class_initialization": initialization,
                      "parent_checkpoint_sha256": common["parent_checkpoint_sha256"],
                      "trainable_parameters": config["trainable_parameters"]}), flush=True)
    train, valid, test = [CocoImages(args.data, split) for split in ("train", "valid", "test")]
    start = time.perf_counter()
    baseline = evaluate(model, valid, args.imgsz, labels)
    view_valid = CocoImages(args.data, "valid", args.eval_prefix) if args.eval_prefix else None
    view_test = CocoImages(args.data, "test", args.eval_prefix) if args.eval_prefix else None
    baseline_view = evaluate(model, view_valid, args.imgsz, labels) if view_valid else None
    best_score, best_epoch, steps, stale = baseline["map50_95"], 0, 0, 0
    save_checkpoint(model, output, "best", common, 0, 0, baseline)
    history = [{"epoch": 0, "validation": baseline, "validation_view": baseline_view}]
    print(json.dumps({"event": "baseline", **baseline}), flush=True)
    steps_per_epoch = math.ceil(len(train.images)/args.batch_size)
    total_steps = (1 if args.smoke else args.epochs)*steps_per_epoch
    warmup = min(steps_per_epoch, 100)
    for epoch in range(1, (1 if args.smoke else args.epochs)+1):
        indices = list(range(len(train.images))); rng.shuffle(indices)
        set_training_mode(model, args.freeze)
        losses, step_times = [], []
        for offset in range(0, len(indices), args.batch_size):
            tick = time.perf_counter()
            rate = args.lr * min(1., (steps+1)/warmup) * (.1+.9*(1+math.cos(math.pi*steps/total_steps))/2)
            optimizer.set_lr(rate)
            batch = make_batch(train, indices[offset:offset+args.batch_size], args.imgsz, rng, epoch-1)
            result = model(batch)
            loss = result["loss"]
            if not bool(paddle.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite loss at step {steps+1}.")
            loss.backward()
            for parameter in trainable:
                if parameter.grad is not None and not bool(paddle.isfinite(parameter.grad).all().item()):
                    raise FloatingPointError(f"Non-finite gradient at step {steps+1}.")
            optimizer.step(); optimizer.clear_grad()
            steps += 1; losses.append(float(loss.item())); step_times.append(time.perf_counter()-tick)
            if steps == 1 or steps % 10 == 0 or args.smoke:
                print(json.dumps({"event": "step", "epoch": epoch, "step": steps,
                                  "loss": losses[-1], "seconds": step_times[-1], "lr": rate}), flush=True)
            if args.smoke and steps >= 3:
                break
        metrics = evaluate(model, valid, args.imgsz, labels)
        save_checkpoint(model, output, "last", common, epoch, steps, metrics)
        if metrics["map50_95"] > best_score:
            best_score, best_epoch, stale = metrics["map50_95"], epoch, 0
            save_checkpoint(model, output, "best", common, epoch, steps, metrics)
        else:
            stale += 1
        row = {"epoch": epoch, "steps": steps, "mean_loss": float(np.mean(losses)),
               "median_step_seconds": float(np.median(step_times)), "validation": metrics}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2)+"\n")
        print(json.dumps({"event": "epoch", **row}), flush=True)
        if stale >= args.patience:
            break
    final_hashes = state_hashes(model)
    changed = [k for k in initial_hashes if initial_hashes[k] != final_hashes[k]]
    frozen_changes = [k for k in changed if k.startswith(frozen_prefixes)]
    if not changed or frozen_changes:
        raise AssertionError(f"Expected trainable weights to change and frozen weights to stay fixed: {frozen_changes}")
    last_manifest, last_path = read_manifest(output / "last.json")
    restored = build_model(ROOT, len(labels), args.imgsz)
    restored.set_state_dict(paddle.load(str(last_path)))
    for key, digest in state_hashes(restored).items():
        if digest != final_hashes[key]:
            raise AssertionError(f"Checkpoint round-trip changed {key}.")
    sample = valid.read(0)[0]
    np.testing.assert_allclose(predict(model, sample, args.imgsz), predict(restored, sample, args.imgsz), atol=1e-5)
    # Test is evaluated once, using the validation-selected checkpoint, never for selection.
    best_manifest, best_path = read_manifest(output / "best.json")
    restored.set_state_dict(paddle.load(str(best_path)))
    test_metrics = evaluate(restored, test, args.imgsz, labels)
    best_view = evaluate(restored, view_valid, args.imgsz, labels) if view_valid else None
    test_view = evaluate(restored, view_test, args.imgsz, labels) if view_test else None
    summary = {"status": "smoke_passed" if args.smoke else "completed", "optimizer_steps": steps,
               "epochs_completed": len(history)-1, "best_epoch": best_epoch,
               "baseline_validation": baseline, "best_validation": best_manifest["validation"],
               "test": test_metrics, "elapsed_seconds": time.perf_counter()-start,
               "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
               "changed_tensors": len(changed), "frozen_tensors_changed": len(frozen_changes),
               "checkpoint_reload_verified": True, "dataset_warnings": metadata["warnings"],
               "parent_checkpoint_sha256": common["parent_checkpoint_sha256"],
               "evaluation_prefix": args.eval_prefix, "baseline_validation_view": baseline_view,
               "best_validation_view": best_view, "test_view": test_view,
               "note": f"The model has {len(labels)} dataset classes; it does not preserve all 78 original categories."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
