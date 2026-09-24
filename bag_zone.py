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
- Acquisition and failed refreshes retry on processed frames. During a lock,
  frame difference is diagnostic only and never changes packing geometry.
- Packing geometry is exposed as a ``ZoneTest`` callable so ``tracking.py``
  stays free of vision dependencies.
"""

import hashlib
import logging
import time
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

    def localize(self, frame_640x480):
        """Return ``{"mask", "conf", "inference_ms"}`` or None when no bag."""
        from config import INFERENCE_SIZE

        start = time.monotonic()
        result = self.model.predict(
            frame_640x480, imgsz=INFERENCE_SIZE, conf=self.conf,
            verbose=False)[0]
        inference_ms = (time.monotonic() - start) * 1000.0
        if result.masks is None or not len(result.masks.data):
            return None
        scores = (result.boxes.conf.cpu().numpy()
                  if result.boxes is not None else np.ones(len(result.masks.data)))
        order = np.argsort(-scores)
        height, width = frame_640x480.shape[:2]
        for index in order:
            mask = result.masks.data[int(index)].cpu().numpy()
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height))
            binary = mask > 0.5
            if binary.mean() < self.min_frac:
                continue
            return {"mask": binary, "conf": float(scores[int(index)]),
                    "inference_ms": inference_ms}
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
    whole-bag area even where the plastic mask leaves the center empty."""
    component = largest_component(mask)
    if not component.any():
        return component
    closed = cv2.morphologyEx(component.astype(np.uint8),
                              cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    flood = closed.copy()
    hole_mask = np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, hole_mask, (0, 0), 2)
    holes = flood != 2
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
    """Packing predicate for one locked footprint (center-in OR overlap).

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
            cx, cy = (float(bbox[0]) + float(bbox[2])) / 2.0, \
                     (float(bbox[1]) + float(bbox[3])) / 2.0
        except (TypeError, ValueError):
            return False
        if _sample_grid(self.grid, cx, cy):
            return True
        return box_overlap_fraction(bbox, self.grid) >= self.overlap_threshold

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
    allowed), ``grace`` (refresh failed; outline retained for two seconds),
    ``lost`` (no zone, packing paused), ``unavailable`` (no model).
    ``packing_paused`` is True in every state except ``stable`` and ``grace``.
    """

    def __init__(self, localizer=None, on_relocation=None,
                 overlap_threshold: float = 0.30,
                 acquire_stable: int = 3, relock_iou: float = 0.7,
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
        self.motion_threshold = motion_threshold
        self.rim_margin = rim_margin
        self.grace_period_s = grace_period_s
        self.max_footprint_frac = max_footprint_frac
        self.implausible_conf = implausible_conf
        self.implausible_frac = implausible_frac
        self.refresh_period_s = refresh_period_s
        self.reset()

    def reset(self) -> None:
        self.status = "locating" if self.localizer is not None else "unavailable"
        self.footprint = None  # Nx2 int polygon in 640x480 pixels
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), np.uint8)
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
        self.misses = 0
        self.previous_small = None
        self.inference_count = 0
        self.inference_ms_ema = 0.0
        self.last_motion = 0.0
        self.grace_deadline = 0.0
        self.rejected_masks = 0

    @property
    def packing_paused(self) -> bool:
        return self.status not in ("stable", "grace")

    def _relocation(self) -> None:
        self.misses = 0
        callback = self.on_relocation
        if callback is not None:
            callback()

    def _adopt(self, grid: np.ndarray, poly, bbox, conf: float,
               frame_id: int, now: float, force_relocation: bool = False) -> None:
        # Frame-id guard: an old prediction must never move the zone back.
        if frame_id <= self.zone_frame_id:
            return
        if self.status in ("locating", "lost") or force_relocation:
            # First lock, re-lock, or recovery: products that looked
            # "outside" under the old (or absent) zone must re-prove
            # themselves, so a bag placed over a stationary product cannot
            # count it as packed.
            self._relocation()
            significant = True
        elif self.status == "grace":
            # Reappearance during grace: same-place refresh continues
            # normally; a real move still restarts product evidence.
            significant = not np.array_equal(grid, self.grid)
            if significant:
                self._relocation()
        else:
            significant = not np.array_equal(grid, self.grid)
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
        self.status = "stable"
        self.consecutive = 0
        self.candidate_grid = None
        self.misses = 0

    def _observe_candidate(self, grid: np.ndarray, frame_id: int) -> None:
        if (self.candidate_grid is not None
                and grid_iou(grid, self.candidate_grid) >= self.relock_iou):
            self.consecutive += 1
        else:
            self.candidate_grid = grid
            self.consecutive = 1

    def _try_lock(self, grid, poly, bbox, conf: float, frame_id: int,
                  now: float) -> None:
        if self.consecutive >= self.acquire_stable:
            self._adopt(grid, poly, bbox, conf, frame_id, now)

    def update(self, frame_640x480, frame_id: int, now=None) -> dict:
        """Advance the zone with one processed frame; never raises."""
        if now is None:
            now = time.monotonic()
        try:
            return self._update(frame_640x480, frame_id, now)
        except Exception:
            logger.exception("Bag zone update failed; packing stays paused.")
            if self.status == "stable":
                self._enter_grace(now)
            elif self.status == "grace" and now >= self.grace_deadline:
                self._enter_lost()
            return self.snapshot()

    def _enter_grace(self, now: float) -> None:
        if self.status == "grace":
            return
        self.grace_deadline = now + self.grace_period_s
        self.status = "grace"
        self._relocation()

    def _register_miss(self, now: float) -> None:
        self.misses += 1
        self.consecutive = 0
        self.candidate_grid = None
        if self.status == "stable":
            self._enter_grace(now)
        elif self.status == "grace":
            if now >= self.grace_deadline:
                self._enter_lost()
        elif self.status != "stable":
            self.status = "lost" if self.zone_frame_id >= 0 else "locating"

    def _enter_lost(self) -> None:
        # Grace expired: remove the outline and pause counting.
        self.footprint = None
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), np.uint8)
        self.zone_bbox = None
        self.zone_conf = 0.0
        self.zone_test = NoZone()
        self.candidate_grid = None
        self.consecutive = 0
        self.misses = 0
        self.status = "lost"
        self._relocation()

    def _credible_refresh(self, observed: np.ndarray):
        """Compare whole shapes after a bounded translation, not raw overlap.

        The missing-area and perimeter gates reject a bottle-shaped notch
        even when total area and IoU are deceptively close to the old bag.
        """
        old = self.last_locked_grid
        old_area = int(old.sum())
        new_area = int(observed.sum())
        if not old_area or not 0.88 <= new_area / old_area <= 1.12:
            return None, "area"
        old_y, old_x = np.nonzero(old)
        new_y, new_x = np.nonzero(observed)
        dx = int(round(float(new_x.mean() - old_x.mean())))
        dy = int(round(float(new_y.mean() - old_y.mean())))
        if max(abs(dx), abs(dy)) > 8:
            return None, "jump"
        aligned = translate_grid(old, dx, dy)
        missing = int(np.logical_and(aligned > 0, observed == 0).sum())
        extra = int(np.logical_and(aligned == 0, observed > 0).sum())
        if missing > old_area * 0.008 or extra > old_area * 0.05:
            return None, "shape"
        kernel = np.ones((3, 3), np.uint8)
        old_edge = int((aligned - cv2.erode(aligned, kernel)).sum())
        new_edge = int((observed - cv2.erode(observed, kernel)).sum())
        if new_edge > old_edge * 1.18:
            return None, "edge"
        return (dx, dy), None

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
        if found is not None:
            ms = found["inference_ms"]
            self.inference_ms_ema = (ms if not self.inference_ms_ema
                                     else 0.2 * ms + 0.8 * self.inference_ms_ema)

        if found is None:
            self._register_miss(now)
            return self.snapshot()

        filled = fill_footprint(found["mask"])
        if not filled.any():
            self._register_miss(now)
            return self.snapshot()
        conf = found["conf"]
        if (float(filled.mean()) > self.max_footprint_frac
                or (conf < self.implausible_conf
                    and float(filled.mean()) > self.implausible_frac)):
            # Implausible mask (live failure: 0.16-conf flood over the whole
            # screen): reject as a miss so the last good outline survives in
            # grace instead of adopting garbage geometry.
            self.rejected_masks += 1
            self._register_miss(now)
            return self.snapshot()
        poly = footprint_contour(filled)
        if poly is None:
            self._register_miss(now)
            return self.snapshot()
        grid = rasterize(poly)
        x1, y1, x2, y2 = (int(poly[:, 0].min()), int(poly[:, 1].min()),
                          int(poly[:, 0].max()), int(poly[:, 1].max()))

        if self.last_locked_grid is not None:
            shift, reason = self._credible_refresh(grid)
            if shift is None:
                self.refresh_rejections += 1
                self.last_rejection = reason
                if self.status in ("stable", "grace"):
                    self._enter_grace(now)
                    if now >= self.grace_deadline:
                        self._enter_lost()
                return self.snapshot()
            dx, dy = shift
            self.last_rejection = None
            self.last_shift_px = [dx * 4, dy * 4]
            moved = max(abs(dx), abs(dy)) >= 2
            # Geometry changes invalidate outside evidence; packed IDs stay
            # with the product tracker and cannot count again on refresh.
            self._adopt(grid, poly, (x1, y1, x2, y2), conf, frame_id, now,
                        force_relocation=moved)
        else:
            self._observe_candidate(grid, frame_id)
            self._try_lock(grid, poly, (x1, y1, x2, y2), conf, frame_id, now)
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
        }
