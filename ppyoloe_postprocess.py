"""Official PP-YOLOE input transform and grocery filtering before ByteTrack."""

import cv2
import numpy as np


def preprocess(frame, size):
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def select_candidates(boxes, scores, threshold, num_classes=365):
    """Pick the best checkpoint label, without relabelling excluded objects."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[2] != 4:
        raise ValueError("PP-YOLOE returned an invalid box tensor.")
    if scores.shape != (1, num_classes, boxes.shape[1]):
        raise ValueError(f"PP-YOLOE must return scores for all {num_classes} checkpoint classes.")
    labels = scores[0].argmax(axis=0)
    confidence = scores[0, labels, np.arange(len(labels))]
    valid = (np.isfinite(confidence) & (confidence >= threshold)
             & np.isfinite(boxes[0]).all(axis=1))
    return np.column_stack((boxes[0, valid], confidence[valid], labels[valid])).astype(np.float32)


def postprocess(rows, shape, size, classes, conf, iou, agnostic):
    import torch
    from torchvision.ops import batched_nms

    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    valid = (np.isfinite(rows).all(axis=1) & (rows[:, 4] >= conf)
             & np.isin(rows[:, 5], classes))
    rows = rows[valid].copy()
    if not len(rows):
        return rows
    height, width = shape[:2]
    rows[:, [0, 2]] = np.clip(rows[:, [0, 2]] * width / size, 0, width)
    rows[:, [1, 3]] = np.clip(rows[:, [1, 3]] * height / size, 0, height)
    rows = rows[(rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])]
    rows = rows[np.argsort(-rows[:, 4], kind="stable")[:1000]]
    if not len(rows):
        return rows
    labels = np.zeros(len(rows), dtype=np.int64) if agnostic else rows[:, 5].astype(np.int64)
    keep = batched_nms(torch.from_numpy(rows[:, :4].copy()),
                       torch.from_numpy(rows[:, 4].copy()), torch.from_numpy(labels), iou).numpy()[:300]
    return rows[keep]
