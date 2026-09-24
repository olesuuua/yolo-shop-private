"""Dynamic bag zone: whole-bag localization for packing decisions.

Separate from the production product detector (which is detection-only and
never sees the bag class). A YOLOE-26n *segmentation* checkpoint runs here
with a "plastic bag" text prompt and returns a whole-bag mask. This module
turns the mask into:

- ``contour``: a fitted outline drawn on the live image, and
- ``footprint``: a hole-filled polygon used by the packing rule
  (a raw plastic mask can leave the center empty, so the rule must use the
  filled area, not the raw mask pixels).

Design notes (MVP):

- ``BagLocalizer`` owns segmentation. ``BagZoneTracker`` locks its outline for
  ten seconds, then replaces it only with a credible whole-bag observation.
- A rejected replacement means "uncertain", not "gone": the tracker enters
  ``grace`` (outline retained for display, counting paused) and retries every
  processed frame. Only sustained absence (no usable candidate) expires to
  ``lost``; a rejected candidate that extends beyond the trusted outline and
  repeats consistently adopts early as a moved/changed bag. Measured live
  data justifies the strict adopt gate (ordinary jitter already costs ~3-8%
  missing area, a bottle notch ~3%, a true partial ~60%): jitter must not
  replace the outline, but it must not delete it either.
- The trusted whole-bag footprint is never carved by occlusion: a
  same-position subset (contained, little outside area, substantially
  smaller) is classified as occlusion by measured spatial evidence
  (containment, retained rim, translation) and can neither replace the
  outline nor sneak back through fresh acquisition while prior context is
  fresh. Consistency alone never proves completeness.
- Acquisition tolerates a couple of consecutive misses (confidence dip,
  transient occlusion) without restarting its consistency streak, within a
  bounded window; inconsistent shapes still restart it.
- During a lock, frame difference is diagnostic only and never changes
  packing geometry.
- Packing geometry is exposed as a ``ZoneTest`` callable so ``tracking.py``
  stays free of vision dependencies.
- Bounded per-inference diagnostics (no video) explain live failures in the
  Details panel and survive Stop; Reset preserves the finished log.
"""

import hashlib
import logging
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

GRID_WIDTH = 160
GRID_HEIGHT = 120


def verify_file(path: Path, expected: str) -> None:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Checksum mismatch for {path.name}; restore the pinned weights.")


def load_bag_model(root: Path):
    """Load the YOLOE-26n segmentation checkpoint with bag text prompts.

    Separate from ``yoloe_model.load_yoloe``: the product profile transfers
    detection-only weights (``yoloe-26n.yaml``) and drops the mask head,
    so it cannot provide the segmentation this component needs.
    """
    from config import (
        BAG_MODEL_PATH, BAG_MODEL_SHA256, BAG_PROMPTS,
    )
    from yoloe_model import TEXT_SHA256, download_verified

    from ultralytics import YOLOE

    path = download_verified(BAG_MODEL_PATH, root / BAG_MODEL_PATH, BAG_MODEL_SHA256)
    model = YOLOE(str(path))
    if getattr(model, "task", "") != "segment":
        raise ValueError("Bag model must be a segmentation checkpoint.")

    import ultralytics

    signature = ["bag-zone", BAG_MODEL_SHA256, TEXT_SHA256,
                 ultralytics.__version__, model.task, list(BAG_PROMPTS)]
    key = hashlib.sha256(repr(signature).encode()).hexdigest()
    cache = root / ".cache" / "bag-prompts" / f"{key}.npz"
    if cache.is_file():
        model.load_prompt_embeddings(cache)
    else:
        import torch
        from ultralytics.nn.text_model import MobileCLIPTS
        from yoloe_model import TEXT_URL

        text_path = download_verified(
            TEXT_URL, root / "weights/yoloe-26x/mobileclip2_b.ts", TEXT_SHA256)
        model.model.clip_model = MobileCLIPTS(
            device=torch.device("cpu"), weight=str(text_path))
        try:
            embeddings = model.model.get_text_pe(list(BAG_PROMPTS),
                                                 cache_clip_model=True)
            model.set_classes(list(BAG_PROMPTS), embeddings)
        finally:
            del model.model.clip_model
        cache.parent.mkdir(parents=True, exist_ok=True)
        staged = cache.with_suffix(".tmp.npz")
        try:
            model.save_prompt_embeddings(staged)
            staged.replace(cache)
        finally:
            staged.unlink(missing_ok=True)
    if list(model.names.values()) != list(BAG_PROMPTS):
        raise ValueError("Bag prompt labels do not match the configured prompts.")
    logger.info("Bag localizer ready: %s with prompts %s",
                Path(BAG_MODEL_PATH).name, ", ".join(BAG_PROMPTS))
    return model


class BagLocalizer:
    """Thin wrapper turning seg-model output into a whole-bag mask."""

    def __init__(self, model, conf: float, min_frac: float):
        self.model = model
        self.conf = conf
        self.min_frac = min_frac
        # Debug of the latest call for no_candidate diagnostics: how many
        # masks the model returned (>= conf) and the best score/area among
        # them. Distinguishes "model blind" (n_masks 0) from "masks too
        # small" without any extra inference or video.
        self.last_debug = {"n_masks": 0, "top_conf": 0.0, "top_frac": 0.0}

    def localize(self, frame_640x480):
        """Return ``{"mask", "conf", "inference_ms"}`` or None when no bag."""
        from config import INFERENCE_SIZE

        start = time.monotonic()
        result = self.model.predict(
            frame_640x480, imgsz=INFERENCE_SIZE, conf=self.conf,
            verbose=False)[0]
        inference_ms = (time.monotonic() - start) * 1000.0
        if result.masks is None or not len(result.masks.data):
            self.last_debug = {"n_masks": 0, "top_conf": 0.0, "top_frac": 0.0}
            return None
        scores = (result.boxes.conf.cpu().numpy()
                  if result.boxes is not None else np.ones(len(result.masks.data)))
        order = np.argsort(-scores)
        height, width = frame_640x480.shape[:2]
        seen = 0
        top_conf, top_frac = 0.0, 0.0
        for index in order:
            mask = result.masks.data[int(index)].cpu().numpy()
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height))
            binary = mask > 0.5
            frac = float(binary.mean())
            seen += 1
            top_conf = max(top_conf, float(scores[int(index)]))
            top_frac = max(top_frac, frac)
            if binary.mean() < self.min_frac:
                continue
            self.last_debug = {"n_masks": seen, "top_conf": top_conf,
                               "top_frac": top_frac}
            return {"mask": binary, "conf": float(scores[int(index)]),
                    "inference_ms": inference_ms}
        self.last_debug = {"n_masks": seen, "top_conf": top_conf,
                           "top_frac": top_frac}
        return None


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def fill_footprint(mask: np.ndarray) -> np.ndarray:
    """Close small gaps and fill interior holes: the packing rule uses the
    whole-bag area even where the plastic mask leaves the center empty.

    Exterior background is found by padding with a one-pixel border and
    flood-filling from (0, 0), which is then guaranteed background even when
    the mask touches the image edge or corner. Without the padding, a mask
    touching (0, 0) makes the seed foreground and the whole frame fills.
    """
    component = largest_component(mask)
    if not component.any():
        return component
    closed = cv2.morphologyEx(component.astype(np.uint8),
                              cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    height, width = closed.shape
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:height + 1, 1:width + 1] = closed
    flood = padded.copy()
    flood_mask = np.zeros((height + 4, width + 4), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 2)
    exterior = flood[1:height + 1, 1:width + 1] == 2
    holes = (closed == 0) & (~exterior)
    return (closed > 0) | holes


def footprint_contour(filled: np.ndarray):
    """Fitted outline polygon (Nx2 int, 640x480 pixels) or None."""
    contours, _ = cv2.findContours(filled.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) <= 0:
        return None
    approx = cv2.approxPolyDP(biggest, 0.005 * cv2.arcLength(biggest, True), True)
    poly = approx.reshape(-1, 2)
    if len(poly) < 3:
        return None
    return poly


def rasterize(poly: np.ndarray) -> np.ndarray:
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), np.uint8)
    if poly is None or len(poly) < 3:
        return grid
    scaled = (poly.astype(np.float32)
              * np.array([GRID_WIDTH / 640.0, GRID_HEIGHT / 480.0]))
    cv2.fillPoly(grid, [scaled.astype(np.int32)], 1)
    return grid


def translate_grid(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return cv2.warpAffine(grid, np.float32([[1, 0, dx], [0, 1, dy]]),
                          (GRID_WIDTH, GRID_HEIGHT),
                          flags=cv2.INTER_NEAREST, borderValue=0)


def grid_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(union) if union else 0.0


# Occlusion-vs-relocation verdict thresholds (see classify_candidate).
# Measured on grid footprints: a 67%-area side bite (live occlusion) scores
# containment 1.00 / extra 0.00 / rim 0.60; a moved bag 0.25 / 0.52 / 0.06;
# live jitter 0.95 / 0.05 / 0.79; a same-center ellipse 0.88 / 0.10 / 0.30.
# Consistency alone never decides: three agreeing partials are still partial.
OCCLUDE_CONTAINMENT = 0.90
OCCLUDE_MAX_EXTRA = 0.05
OCCLUDE_MAX_AREA = 0.88
RELOCATE_MIN_EXTRA = 0.05


def retained_boundary(old: np.ndarray, new: np.ndarray) -> float:
    """Fraction of the trusted rim still present in the candidate.

    Rim pixels of ``old`` within Chebyshev distance 2 of any ``new`` rim
    pixel count as retained. An occluded bag keeps most of its visible rim;
    a moved bag keeps almost none.
    """
    kernel = np.ones((3, 3), np.uint8)
    old_edge = (old - cv2.erode(old, kernel)) > 0
    new_edge = (new - cv2.erode(new, kernel)) > 0
    total = int(old_edge.sum())
    if not total or not bool(new_edge.any()):
        return 0.0
    near = cv2.dilate(new_edge.astype(np.uint8),
                      np.ones((5, 5), np.uint8)) > 0
    return float(np.logical_and(old_edge, near).sum()) / float(total)


def compare_footprints(old: np.ndarray, observed: np.ndarray):
    """Strict whole-shape comparison after a bounded translation.

    The missing-area and perimeter gates reject a bottle-shaped notch even
    when total area and IoU are deceptively close to the old bag. Gates stay
    strict on purpose (ordinary jitter already costs ~3-8% missing area vs
    the 0.8% gate): jitter must never *replace* the outline; the state
    machine rides it out as uncertainty instead.

    Returns ``(shift, reason, metrics)``; ``shift`` is None on rejection.
    """
    old_area = int(old.sum())
    new_area = int(observed.sum())
    metrics = {"area_ratio": round(new_area / old_area, 3) if old_area else 0.0,
               "iou": round(grid_iou(old, observed), 3)}
    if not old_area or not 0.88 <= new_area / old_area <= 1.12:
        return None, "area", metrics
    old_y, old_x = np.nonzero(old)
    new_y, new_x = np.nonzero(observed)
    dx = int(round(float(new_x.mean() - old_x.mean())))
    dy = int(round(float(new_y.mean() - old_y.mean())))
    metrics["shift_cells"] = [dx, dy]
    if max(abs(dx), abs(dy)) > 8:
        return None, "jump", metrics
    aligned = translate_grid(old, dx, dy)
    missing = int(np.logical_and(aligned > 0, observed == 0).sum())
    extra = int(np.logical_and(aligned == 0, observed > 0).sum())
    metrics["missing_frac"] = round(missing / old_area, 4)
    metrics["extra_frac"] = round(extra / old_area, 4)
    if missing > old_area * 0.008 or extra > old_area * 0.05:
        return None, "shape", metrics
    kernel = np.ones((3, 3), np.uint8)
    old_edge = int((aligned - cv2.erode(aligned, kernel)).sum())
    new_edge = int((observed - cv2.erode(observed, kernel)).sum())
    metrics["edge_ratio"] = round(new_edge / old_edge, 3) if old_edge else 0.0
    if new_edge > old_edge * 1.18:
        return None, "edge", metrics
    return (dx, dy), None, metrics


def classify_candidate(trusted: np.ndarray, candidate: np.ndarray) -> dict:
    """Occlusion vs relocation verdict from measured spatial evidence.

    - ``occlusion``: the candidate is essentially a same-position subset of
      the trusted outline (contained, little outside area, substantially
      smaller) — a hand or product covering part of the bag. Never proof of
      a new bag, however consistent.
    - ``relocation``: the candidate extends meaningfully beyond the trusted
      outline — the bag moved, spread, or was replaced elsewhere.
    - ``same``: same position and size within tolerance (rim flicker).
    - ``uncertain``: anything else; retry, adopt nothing.
    """
    trusted_area = int(trusted.sum())
    cand_area = int(candidate.sum())
    inter = int(np.logical_and(trusted > 0, candidate > 0).sum())
    containment = inter / cand_area if cand_area else 0.0
    extra = (cand_area - inter) / trusted_area if trusted_area else 0.0
    missing = (trusted_area - inter) / trusted_area if trusted_area else 0.0
    area_ratio = cand_area / trusted_area if trusted_area else 0.0
    trusted_y, trusted_x = np.nonzero(trusted)
    cand_y, cand_x = np.nonzero(candidate)
    if len(cand_x) and len(trusted_x):
        dx = int(round(float(cand_x.mean() - trusted_x.mean())))
        dy = int(round(float(cand_y.mean() - trusted_y.mean())))
    else:
        dx, dy = 0, 0
    spatial = {
        "area_ratio": round(area_ratio, 3),
        "containment": round(containment, 3),
        "extra_frac": round(extra, 4),
        "missing_frac": round(missing, 4),
        "shift_cells": [dx, dy],
        "retained_boundary": round(retained_boundary(trusted, candidate), 3),
        "iou": round(grid_iou(trusted, candidate), 3),
    }
    if (containment >= OCCLUDE_CONTAINMENT and extra < OCCLUDE_MAX_EXTRA
            and area_ratio < OCCLUDE_MAX_AREA):
        verdict = "occlusion"
    elif extra >= RELOCATE_MIN_EXTRA:
        verdict = "relocation"
    elif (0.88 <= area_ratio <= 1.12 and containment >= OCCLUDE_CONTAINMENT
            and extra < OCCLUDE_MAX_EXTRA):
        verdict = "same"
    else:
        verdict = "uncertain"
    return {"verdict": verdict, **spatial}


def box_footprint_fractions(bbox, grid: np.ndarray) -> tuple[float, float]:
    """(footprint_fraction, box_fraction) of the box∩footprint area.

    ``footprint_fraction`` is large only for bag-sized boxes sitting on the
    bag; contained products cover a small part of the footprint.
    ``box_fraction`` additionally requires the box itself to lie mostly
    inside, so big neighboring items are never mistaken for the bag.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return (0.0, 0.0)
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    foot_area = float(grid.sum()) * (640.0 / GRID_WIDTH) * (480.0 / GRID_HEIGHT)
    if box_area <= 0 or foot_area <= 0:
        return (0.0, 0.0)
    inter = box_overlap_fraction(bbox, grid) * box_area
    return (inter / foot_area, inter / box_area)


def is_bag_self(bbox, grid: np.ndarray,
                foot_frac: float = 0.55, box_frac: float = 0.5) -> bool:
    """True when a product box is really the bag itself.

    Calibrated on the live blue-bag scene: the false `Storage box` box
    covering the bag scores footprint≈0.9/box≈0.95, while genuine products
    beside it score ≈0 on at least one ratio.
    """
    f_foot, f_box = box_footprint_fractions(bbox, grid)
    return f_foot >= foot_frac and f_box >= box_frac


def _sample_grid(grid: np.ndarray, x: float, y: float) -> bool:
    gx = min(GRID_WIDTH - 1, max(0, int(x * GRID_WIDTH / 640.0)))
    gy = min(GRID_HEIGHT - 1, max(0, int(y * GRID_HEIGHT / 480.0)))
    return bool(grid[gy, gx])


def box_overlap_fraction(bbox, grid: np.ndarray) -> float:
    x1, y1, x2, y2 = (max(0.0, min(640.0, float(v))) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    # Sample the box on the coarse grid; exact for boxes >= ~2 cells.
    gx1, gy1 = int(x1 * GRID_WIDTH / 640.0), int(y1 * GRID_HEIGHT / 480.0)
    gx2, gy2 = int(np.ceil(x2 * GRID_WIDTH / 640.0)), int(np.ceil(y2 * GRID_HEIGHT / 480.0))
    gx1, gy1 = max(0, gx1), max(0, gy1)
    gx2, gy2 = min(GRID_WIDTH, gx2), min(GRID_HEIGHT, gy2)
    if gx2 <= gx1 or gy2 <= gy1:
        return 1.0 if _sample_grid(grid, (x1 + x2) / 2, (y1 + y2) / 2) else 0.0
    return float(grid[gy1:gy2, gx1:gx2].mean())


class ZoneTest:
    """Packing predicate for one locked footprint (overlap only).

    "Mostly inside" means intersection area ÷ product-box area >=
    ``overlap_threshold`` (0.70): the product box itself must lie mostly
    inside the filled footprint. Center-inside alone is insufficient, so a
    huge box whose center falls in the bag never counts.

    Rim hysteresis: the inside test uses the exact footprint, but outside
    evidence requires clearing a dilated rim (``margin_px``). A flickering
    mask edge must not manufacture outside->inside streaks for a product
    resting at the rim; genuine removals travel far beyond the margin.
    """

    def __init__(self, grid: np.ndarray, overlap_threshold: float,
                 version: int, margin_px: float = 16.0):
        self.grid = grid
        self.overlap_threshold = overlap_threshold
        self.version = version
        cells = max(1, int(round(margin_px * GRID_WIDTH / 640.0)))
        self.outer = cv2.dilate(grid, np.ones((cells * 2 + 1,) * 2, np.uint8))

    def __call__(self, bbox) -> bool:
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        return box_overlap_fraction(bbox, self.grid) >= self.overlap_threshold

    def overlap_fraction(self, bbox) -> float:
        """Product-box area share inside the footprint (diagnostics)."""
        try:
            return float(box_overlap_fraction(bbox, self.grid))
        except (TypeError, ValueError):
            return 0.0

    def clearly_outside(self, bbox) -> bool:
        """True only when the box is well clear of the footprint rim."""
        try:
            cx, cy = (float(bbox[0]) + float(bbox[2])) / 2.0, \
                     (float(bbox[1]) + float(bbox[3])) / 2.0
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        if _sample_grid(self.outer, cx, cy):
            return False
        return box_overlap_fraction(bbox, self.outer) < self.overlap_threshold


class NoZone:
    """Packing predicate before the first lock: nothing can be inside.

    New packing confirmations are impossible until a bag is acquired; this
    is explicit (see the LOCATING banner), never a silent fixed-ROI fallback.
    """
    version = -1

    def __call__(self, bbox) -> bool:
        return False

    def clearly_outside(self, bbox) -> bool:
        return True

    def overlap_fraction(self, bbox) -> float:
        return 0.0


def motion_energy(previous_small: np.ndarray, frame_640x480,
                  bbox, margin: int = 20) -> float:
    """Mean abs grayscale diff inside the expanded zone bbox (0-255).

    Computed on a 160x120 copy: cheap enough to run every processed frame.
    """
    small = cv2.resize(frame_640x480, (GRID_WIDTH, GRID_HEIGHT))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
    x1, y1, x2, y2 = bbox
    sx1 = max(0, int(x1 * GRID_WIDTH / 640.0) - margin)
    sy1 = max(0, int(y1 * GRID_HEIGHT / 480.0) - margin)
    sx2 = min(GRID_WIDTH, int(np.ceil(x2 * GRID_WIDTH / 640.0)) + margin)
    sy2 = min(GRID_HEIGHT, int(np.ceil(y2 * GRID_HEIGHT / 480.0)) + margin)
    region_prev = previous_small[sy1:sy2, sx1:sx2].astype(np.int16)
    region_now = gray[sy1:sy2, sx1:sx2]
    if region_now.size == 0:
        return 0.0
    return float(np.abs(region_now - region_prev).mean())


class BagZoneTracker:
    """State machine turning bag masks into a packing zone.

    States: ``locating`` (no zone yet), ``stable`` (zone locked, packing
    allowed), ``grace`` (refresh failed; outline retained for display but
    counting paused until the outline is confirmed again), ``lost`` (no
    zone, packing paused), ``unavailable`` (no model).
    ``packing_paused`` is True in every state except ``stable``.
    """

    # Bounded local diagnostics: per-inference records and state
    # transitions. No video, just numbers and reasons. Survives Stop
    # (server state is untouched by it); Reset stashes a copy first.
    DIAG_RECENT = 48
    DIAG_TRANSITIONS = 16

    def __init__(self, localizer=None, on_relocation=None,
                 overlap_threshold: float = 0.70,
                 acquire_stable: int = 3, relock_iou: float = 0.7,
                 acquire_tolerate_misses: int = 2,
                 prior_ttl_s: float = 30.0,
                 motion_threshold: float = 12.0,
                 rim_margin: float = 16.0, grace_period_s: float = 2.0,
                 max_footprint_frac: float = 0.70,
                 implausible_conf: float = 0.20,
                 implausible_frac: float = 0.45,
                 refresh_period_s: float = 10.0):
        self.localizer = localizer
        self.on_relocation = on_relocation
        self.overlap_threshold = overlap_threshold
        self.acquire_stable = acquire_stable
        self.relock_iou = relock_iou
        self.acquire_tolerate_misses = acquire_tolerate_misses
        self.prior_ttl_s = prior_ttl_s
        self.motion_threshold = motion_threshold
        self.rim_margin = rim_margin
        self.grace_period_s = grace_period_s
        self.max_footprint_frac = max_footprint_frac
        self.implausible_conf = implausible_conf
        self.implausible_frac = implausible_frac
        self.refresh_period_s = refresh_period_s
        self.last_session_log = None
        self.reset()

    def reset(self) -> None:
        # Preserve the finished diagnostic log before runtime state is
        # cleared, so a live failure can still be explained after Reset.
        try:
            previous = list(getattr(self, "diag_log", []))
            transitions = list(getattr(self, "transitions", []))
        except Exception:
            previous, transitions = [], []
        if previous or transitions:
            self.last_session_log = {
                "ended": time.strftime("%H:%M:%S"),
                "records": previous[-self.DIAG_RECENT:],
                "transitions": transitions[-self.DIAG_TRANSITIONS:],
            }
        self.status = "locating" if self.localizer is not None else "unavailable"
        self.footprint = None  # Nx2 int polygon in 640x480 pixels
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), np.uint8)
        # Reference geometry for refresh comparisons: cleared here and on
        # loss so reacquisition never compares against an old outline.
        self.last_locked_grid = None
        self.locked_at = None
        self.last_update_at = None
        self.move_warning = False
        self.refresh_rejections = 0
        self.last_rejection = None
        self.last_shift_px = [0, 0]
        self.zone_bbox = None
        self.zone_conf = 0.0
        self.zone_frame_id = -1
        self.zone_test = NoZone()
        self.zone_version = 0
        self.candidate_grid = None
        self.consecutive = 0
        self.last_streak_iou = None
        self.acquire_misses = 0
        self.rej_candidate = None
        self.rej_consecutive = 0
        # Prior trusted outline retained across loss (req: no
        # loss/reacquisition loophole). While fresh, same-position subsets
        # are recognized as likely occlusion and never acquired; genuine
        # moves (relocation verdict) and full-outline returns still lock.
        # Expires via prior_ttl_s so a truly changed bag is never blocked
        # permanently. Cleared here, on adopt, and never displayed.
        self.prior_grid = None
        self.prior_at = 0.0
        self.last_refresh = None
        self.reason = "locating: waiting for bag"
        self.misses = 0
        self.previous_small = None
        self.inference_count = 0
        self.inference_ms_ema = 0.0
        self.last_motion = 0.0
        self.grace_deadline = 0.0
        self.rejected_masks = 0
        self.diag_log = deque(maxlen=self.DIAG_RECENT)
        self.transitions = deque(maxlen=self.DIAG_TRANSITIONS)

    @property
    def packing_paused(self) -> bool:
        # Grace is uncertainty: the outline is retained for display, but
        # counting stays paused until a credible outline confirms it again.
        return self.status != "stable"

    def _relocation(self) -> None:
        self.misses = 0
        callback = self.on_relocation
        if callback is not None:
            callback()

    def _transition(self, new: str, cause: str, frame_id: int) -> None:
        old = self.status
        if old == new:
            return
        self.status = new
        try:
            self.transitions.append({
                "at": time.strftime("%H:%M:%S"),
                "frame": int(frame_id),
                "from": old,
                "to": new,
                "cause": str(cause),
            })
        except Exception:
            logger.exception("Bag diagnostic transition log failed.")

    def _note(self, frame_id: int, outcome: str, reason: str = "",
              conf=None, raw_frac=None, filled_frac=None,
              streak: str = "", iou=None, refresh: dict | None = None) -> None:
        """Append one bounded per-inference diagnostic record (no video)."""
        try:
            record = {"frame": int(frame_id), "outcome": outcome}
            if reason:
                record["reason"] = reason
            if conf is not None:
                record["conf"] = round(float(conf), 3)
            if raw_frac is not None:
                record["raw_frac"] = round(float(raw_frac), 3)
            if filled_frac is not None:
                record["filled_frac"] = round(float(filled_frac), 3)
            if streak:
                record["streak"] = streak
            if iou is not None:
                record["iou"] = round(float(iou), 3)
            if refresh:
                record["refresh"] = refresh
            self.diag_log.append(record)
        except Exception:
            logger.exception("Bag diagnostic record failed.")

    def _adopt(self, grid: np.ndarray, poly, bbox, conf: float,
               frame_id: int, now: float, force_relocation: bool = False,
               cause: str = "acquired") -> None:
        # Frame-id guard: an old prediction must never move the zone back.
        if frame_id <= self.zone_frame_id:
            return
        if self.status in ("locating", "lost") or force_relocation:
            # First lock, re-lock, or recovery: incomplete streaks and
            # pending watches restart, so visible items are evaluated
            # afresh against the new outline (consistent inside observations
            # can then count, even without a prior outside sighting).
            self._relocation()
            significant = True
        elif self.status == "grace":
            # Reappearance during grace: same-place refresh continues
            # normally; a real move still restarts product evidence.
            significant = not np.array_equal(grid, self.grid)
            if significant:
                self._relocation()
        else:
            # Stable refresh: any geometry change restarts product evidence
            # so visible items are evaluated afresh and pending watches tied
            # to the old outline cannot fire later.
            significant = not np.array_equal(grid, self.grid)
            if significant:
                self._relocation()
        if significant:
            self.zone_version += 1
            self.zone_test = ZoneTest(grid, self.overlap_threshold,
                                      self.zone_version,
                                      margin_px=self.rim_margin)
        else:
            np.copyto(self.grid, grid)
        self.grid = self.zone_test.grid
        self.last_locked_grid = self.grid.copy()
        self.footprint = poly
        self.zone_bbox = bbox
        self.zone_conf = conf
        self.zone_frame_id = frame_id
        self.locked_at = now
        self.move_warning = False
        self.last_update_at = now
        self._transition("stable", cause, frame_id)
        self.status = "stable"
        self.consecutive = 0
        self.candidate_grid = None
        self.last_streak_iou = None
        self.acquire_misses = 0
        self.rej_candidate = None
        self.rej_consecutive = 0
        # A newly trusted outline supersedes any retained prior context.
        self.prior_grid = None
        self.prior_at = 0.0
        self.misses = 0
        self.reason = "tracking (conf %.3f)" % float(conf)

    def _observe_candidate(self, grid: np.ndarray, frame_id: int) -> float:
        iou = None
        if self.candidate_grid is not None:
            iou = grid_iou(grid, self.candidate_grid)
            if iou >= self.relock_iou:
                self.consecutive += 1
            else:
                self.candidate_grid = grid
                self.consecutive = 1
        else:
            self.candidate_grid = grid
            self.consecutive = 1
        self.last_streak_iou = iou
        return iou

    def _observe_rejected(self, grid: np.ndarray) -> float:
        """Consistency among rejected replacements (moved-bag fast path).

        Returns the IoU against the previous rejected outline. A stably
        re-observed new shape adopts as the moved bag; jitter that agrees
        with nothing never reaches the adoption streak.
        """
        iou = None
        if self.rej_candidate is not None:
            iou = grid_iou(grid, self.rej_candidate)
            if iou >= self.relock_iou:
                self.rej_consecutive += 1
            else:
                self.rej_candidate = grid
                self.rej_consecutive = 1
        else:
            self.rej_candidate = grid
            self.rej_consecutive = 1
        return iou

    def update(self, frame_640x480, frame_id: int, now=None) -> dict:
        """Advance the zone with one processed frame; never raises."""
        if now is None:
            now = time.monotonic()
        try:
            return self._update(frame_640x480, frame_id, now)
        except Exception:
            logger.exception("Bag zone update failed; packing stays paused.")
            # Never fail silently: the panel must show that the bag path
            # itself errored instead of freezing on a stale "waiting" reason.
            self.reason = "bag update error — see server log"
            if self.status == "stable":
                self._enter_grace(now, self.zone_frame_id, "update_error")
            elif self.status == "grace" and now >= self.grace_deadline:
                self._enter_lost("update_error during grace", now, self.zone_frame_id)
            return self.snapshot()

    def _enter_grace(self, now: float, frame_id: int, cause: str) -> None:
        if self.status == "grace":
            return
        self.grace_deadline = now + self.grace_period_s
        # Uncertainty, not disappearance: keep the outline for display, pause
        # counting, keep product streaks frozen (no relocation) so a
        # transiently rejected refresh does not wipe live packing evidence.
        self._transition("grace", cause, frame_id)
        self.reason = "uncertain: %s — outline kept, counting paused" % cause

    def _register_miss(self, now: float, frame_id: int, cause: str) -> None:
        """Record an unusable bag observation.

        ``cause`` distinguishes "no_candidate" (the model returned nothing)
        from rejected candidates ("empty", "implausible", "no_contour").
        Only sustained absence expires to loss; a rejected-but-present bag
        stays uncertain with its outline retained.
        """
        self.misses += 1
        if self.status in ("locating", "lost"):
            # Tolerate occasional misses without restarting acquisition;
            # inconsistent shapes still restart it in _observe_candidate.
            self.acquire_misses += 1
            if self.acquire_misses > self.acquire_tolerate_misses:
                self.consecutive = 0
                self.candidate_grid = None
                self.last_streak_iou = None
                self.acquire_misses = 0
                self.reason = ("acquiring: reset after %d misses"
                               % (self.acquire_tolerate_misses + 1))
            else:
                self.reason = ("acquiring: miss %d/%d tolerated (streak %d/%d)"
                               % (self.acquire_misses, self.acquire_tolerate_misses,
                                  self.consecutive, self.acquire_stable))
            return
        if self.status == "stable":
            self._enter_grace(now, frame_id, cause)
        elif self.status == "grace":
            if now >= self.grace_deadline:
                self._enter_lost("sustained absence (%s)" % cause, now, frame_id)

    def _enter_lost(self, cause: str, now: float, frame_id: int = -1) -> None:
        # Grace expired: remove the outline and pause counting. The trusted
        # reference geometry is cleared so reacquisition never compares
        # against the old outline — but a copy is retained as prior context
        # so same-position partials are recognized as likely occlusion and
        # cannot slip back in through fresh acquisition.
        if self.last_locked_grid is not None:
            self.prior_grid = self.last_locked_grid.copy()
            self.prior_at = now
        self.footprint = None
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), np.uint8)
        self.last_locked_grid = None
        self.zone_bbox = None
        self.zone_conf = 0.0
        self.zone_test = NoZone()
        self.candidate_grid = None
        self.consecutive = 0
        self.last_streak_iou = None
        self.acquire_misses = 0
        self.rej_candidate = None
        self.rej_consecutive = 0
        self.misses = 0
        self._transition("lost", cause, frame_id)
        self.status = "lost"
        self.reason = "lost: %s — packing paused" % cause
        self._relocation()

    def _credible_refresh(self, observed: np.ndarray, old=None):
        """Strict refresh check of ``observed`` against the locked outline.

        Delegates to compare_footprints; ``old`` defaults to the locked
        reference (pass an explicit grid to check against prior context).
        Returns ``(shift, reason, metrics)``; ``shift`` is None on rejection.
        """
        if old is None:
            old = self.last_locked_grid
        return compare_footprints(old, observed)

    def _verdict_summary(self, verdict: dict) -> str:
        return ("%s (contained %.2f, rim %.2f, shift %s, area %.2f)" % (
            verdict["verdict"], verdict["containment"],
            verdict["retained_boundary"], verdict["shift_cells"],
            verdict["area_ratio"]))

    def _update(self, frame_640x480, frame_id: int, now: float) -> dict:
        self.last_update_at = now
        if self.localizer is None:
            self.status = "unavailable"
            return self.snapshot()
        if frame_id <= self.zone_frame_id:
            # Duplicate or reordered delivery: keep the newer zone.
            return self.snapshot()

        small = cv2.resize(frame_640x480, (GRID_WIDTH, GRID_HEIGHT))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if (self.status == "stable" and self.zone_bbox is not None
                and self.previous_small is not None):
            self.last_motion = motion_energy(
                self.previous_small, frame_640x480, self.zone_bbox)
            if self.last_motion >= self.motion_threshold:
                self.move_warning = True
        self.previous_small = gray
        if self.status == "stable" and now - self.locked_at < self.refresh_period_s:
            return self.snapshot()

        found = self.localizer.localize(frame_640x480)
        self.inference_count += 1
        conf = None
        raw_frac = None
        if found is not None:
            ms = found["inference_ms"]
            self.inference_ms_ema = (ms if not self.inference_ms_ema
                                     else 0.2 * ms + 0.8 * self.inference_ms_ema)
            conf = found["conf"]
            raw_frac = float(found["mask"].mean())

        if found is None:
            debug = getattr(self.localizer, "last_debug", None) or {}
            n_masks = int(debug.get("n_masks", 0) or 0)
            top_frac = float(debug.get("top_frac", 0.0) or 0.0)
            top_conf = float(debug.get("top_conf", 0.0) or 0.0)
            conf_threshold = getattr(self.localizer, "conf", None)
            local_min_frac = getattr(self.localizer, "min_frac", None)
            if n_masks > 0 and local_min_frac is not None:
                detail = ("masks too small (best frac %.3f < %.2f)"
                          % (top_frac, float(local_min_frac)))
            elif n_masks > 0:
                detail = ("masks rejected (best frac %.3f)" % top_frac)
            elif conf_threshold is not None:
                detail = ("no proposals >= conf %.2f" % float(conf_threshold))
            else:
                detail = "no proposals"
            self._note(frame_id, "no_candidate", detail,
                       top_conf or None, None, top_frac or None)
            self.reason = "acquiring: model returned no candidate (%s)" % detail \
                if self.status in ("locating", "lost") \
                else "uncertain: model returned no candidate (%s)" % detail
            self._register_miss(now, frame_id, "no_candidate")
            return self.snapshot()

        filled = fill_footprint(found["mask"])
        filled_frac = float(filled.mean())
        if not filled.any():
            self._note(frame_id, "rejected", "empty", conf, raw_frac, filled_frac)
            self._register_miss(now, frame_id, "empty")
            return self.snapshot()
        if (filled_frac > self.max_footprint_frac
                or (conf < self.implausible_conf
                    and filled_frac > self.implausible_frac)):
            # Implausible mask (live failure: 0.16-conf flood over the whole
            # screen): reject so the last good outline survives in grace
            # instead of adopting garbage geometry.
            self.rejected_masks += 1
            self._note(frame_id, "rejected", "implausible",
                       conf, raw_frac, filled_frac)
            self._register_miss(now, frame_id, "implausible")
            return self.snapshot()
        poly = footprint_contour(filled)
        if poly is None:
            self._note(frame_id, "rejected", "no_contour",
                       conf, raw_frac, filled_frac)
            self._register_miss(now, frame_id, "no_contour")
            return self.snapshot()
        grid = rasterize(poly)
        x1, y1, x2, y2 = (int(poly[:, 0].min()), int(poly[:, 1].min()),
                          int(poly[:, 0].max()), int(poly[:, 1].max()))

        if (self.last_locked_grid is not None
                and self.status in ("stable", "grace")):
            # Locked refresh: a new outline must resemble the old bag
            # (bounded translation, area, missing/extra, edge gates), so a
            # partial mask can never replace the full outline. A rejection
            # is uncertainty, not disappearance: the trusted outline is
            # retained, counting pauses, and every frame retries.
            shift, reason, metrics = self._credible_refresh(grid)
            self.last_refresh = {"reason": reason or "ok", **metrics}
            if shift is None:
                self.refresh_rejections += 1
                self.last_rejection = reason
                verdict = classify_candidate(self.last_locked_grid, grid)
                spatial = {key: verdict[key] for key in (
                    "verdict", "area_ratio", "containment", "extra_frac",
                    "missing_frac", "shift_cells", "retained_boundary",
                    "iou")}
                self.last_refresh = {"reason": reason or "ok", **metrics,
                                     **spatial}
                summary = self._verdict_summary(verdict)
                if self.status == "stable":
                    self._enter_grace(now, frame_id,
                                      "refresh rejected (%s): %s"
                                      % (reason, summary))
                    outcome = ("occlusion_kept" if verdict["verdict"] == "occlusion"
                               else "refresh_rejected")
                else:
                    self.reason = ("uncertain: refresh rejected (%s): %s — "
                                   "outline kept, counting paused"
                                   % (reason, summary))
                    outcome = ("occlusion_kept" if verdict["verdict"] == "occlusion"
                               else "refresh_rejected")
                self._note(frame_id, outcome, "%s + %s" % (reason, summary),
                           conf, raw_frac, filled_frac,
                           refresh=self.last_refresh)
                # Relocation fast path, spatially gated: only a candidate
                # that extends beyond the trusted outline (relocation
                # verdict) can adopt early — and only after mutual
                # consistency. Occluded/same-position subsets never adopt
                # here, however often they repeat.
                if verdict["verdict"] == "relocation":
                    rej_iou = self._observe_rejected(grid)
                    if self.rej_consecutive >= self.acquire_stable:
                        self._note(frame_id, "adopted", "relocated: " + summary,
                                   conf, raw_frac, filled_frac,
                                   streak="%d/%d" % (self.rej_consecutive,
                                                     self.acquire_stable),
                                   iou=rej_iou,
                                   refresh=self.last_refresh)
                        self._adopt(grid, poly, (x1, y1, x2, y2), conf,
                                    frame_id, now, cause="relocated")
                else:
                    self.rej_candidate = None
                    self.rej_consecutive = 0
                if (self.status == "grace" and now >= self.grace_deadline
                        and self.rej_consecutive < self.acquire_stable):
                    self._enter_lost("refresh rejected 2s (%s): %s"
                                     % (reason, summary), now, frame_id)
                return self.snapshot()
            dx, dy = shift
            self.last_rejection = None
            self.last_shift_px = [dx * 4, dy * 4]
            moved = max(abs(dx), abs(dy)) >= 2
            self._note(frame_id, "refresh_ok", "",
                       conf, raw_frac, filled_frac, refresh=metrics)
            # Geometry changes invalidate outside evidence; packed IDs stay
            # with the product tracker and cannot count again on refresh.
            self._adopt(grid, poly, (x1, y1, x2, y2), conf, frame_id, now,
                        force_relocation=moved,
                        cause="refresh" if not moved else "refresh_moved")
        else:
            # Locating or lost reacquisition. A fresh prior outline (kept
            # across loss, never displayed) guards the loophole where a
            # partial mask causes loss and is immediately re-accepted:
            # same-position subsets are recognized as likely occlusion and
            # held, while a strict full-outline match or a relocation
            # verdict still locks. The prior expires via prior_ttl_s so a
            # genuinely changed bag is never blocked permanently.
            prior_fresh = (self.prior_grid is not None
                           and (now - self.prior_at) <= self.prior_ttl_s)
            if prior_fresh:
                strict_shift, _, strict_metrics = self._credible_refresh(
                    grid, old=self.prior_grid)
                if strict_shift is not None:
                    self.acquire_misses = 0
                    iou = self._observe_candidate(grid, frame_id)
                    streak = "%d/%d" % (self.consecutive, self.acquire_stable)
                    self._note(frame_id, "candidate", "recovery streak " + streak,
                               conf, raw_frac, filled_frac, streak=streak,
                               iou=iou, refresh=strict_metrics)
                else:
                    verdict = classify_candidate(self.prior_grid, grid)
                    spatial = {key: verdict[key] for key in (
                        "verdict", "area_ratio", "containment", "extra_frac",
                        "missing_frac", "shift_cells", "retained_boundary",
                        "iou")}
                    if verdict["verdict"] == "relocation":
                        self.acquire_misses = 0
                        iou = self._observe_candidate(grid, frame_id)
                        streak = "%d/%d" % (self.consecutive, self.acquire_stable)
                        self._note(frame_id, "candidate",
                                   "relocation streak " + streak + ": "
                                   + self._verdict_summary(verdict),
                                   conf, raw_frac, filled_frac, streak=streak,
                                   iou=iou, refresh=spatial)
                    else:
                        # Likely same-position occlusion (or flicker): hold,
                        # keep presence evidence, advance nothing.
                        self.acquire_misses = 0
                        self.reason = ("lost: likely occlusion of prior bag "
                                       "(%s) — awaiting full outline"
                                       % self._verdict_summary(verdict))
                        self._note(frame_id, "prior_hold",
                                   self._verdict_summary(verdict),
                                   conf, raw_frac, filled_frac,
                                   refresh=spatial)
                        return self.snapshot()
            else:
                # No (or expired) prior context: accept a different shape,
                # size, or location after consistent new observations,
                # without comparing against any old outline. Occasional
                # misses do not restart the streak (bounded tolerance);
                # inconsistent shapes do.
                if self.prior_grid is not None:
                    self.prior_grid = None
                    self.prior_at = 0.0
                self.acquire_misses = 0
                iou = self._observe_candidate(grid, frame_id)
                streak = "%d/%d" % (self.consecutive, self.acquire_stable)
            if self.consecutive >= self.acquire_stable:
                self._note(frame_id, "adopted", "acquired",
                           conf, raw_frac, filled_frac, streak=streak, iou=iou)
                self._adopt(grid, poly, (x1, y1, x2, y2), conf, frame_id,
                            now, cause="acquired")
            else:
                self.reason = "acquiring: streak %s · iou %s · conf %.3f" % (
                    streak, "—" if iou is None else "%.2f" % iou, float(conf))
                self._note(frame_id, "candidate", "", conf, raw_frac,
                           filled_frac, streak=streak, iou=iou)
        return self.snapshot()

    def snapshot(self) -> dict:
        poly = self.footprint
        return {
            "status": self.status,
            "packing_paused": self.packing_paused,
            "conf": round(float(self.zone_conf), 3),
            "bbox": list(self.zone_bbox) if self.zone_bbox else None,
            "zone_frame_id": self.zone_frame_id,
            "zone_version": self.zone_version,
            "contour": [[int(x), int(y)] for x, y in poly[:64]]
            if poly is not None else [],
            "inference_count": self.inference_count,
            "inference_ms_ema": round(self.inference_ms_ema, 1),
            "last_motion": round(self.last_motion, 2),
            "possible_bag_move_during_lock": self.move_warning,
            "last_refresh_shift_px": self.last_shift_px,
            "refresh_rejections": self.refresh_rejections,
            "last_refresh_rejection": self.last_rejection,
            "refresh_due_in_s": round(max(0.0, self.refresh_period_s -
                (self.last_update_at - self.locked_at)), 1) if self.locked_at is not None else None,
            "rejected_masks": self.rejected_masks,
            "bag_diag": {
                "reason": self.reason,
                "streak": "%d/%d" % (self.consecutive, self.acquire_stable),
                "streak_iou": (round(float(self.last_streak_iou), 3)
                               if self.last_streak_iou is not None else None),
                "acquire_misses": self.acquire_misses,
                "last_refresh": self.last_refresh,
                "prior": {"active": self.prior_grid is not None},
                "recent": list(self.diag_log)[-8:],
                "transitions": list(self.transitions),
                "last_log": self.last_session_log,
            },
        }
