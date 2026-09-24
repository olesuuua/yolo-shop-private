"""Geometry and packing events, independent of YOLO, OpenCV and FastAPI."""

import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Iterable

from config import (
    BAG_OVERLAP_THRESHOLD, BAG_ROI, FOOD_CLASSES,
    MIN_INSIDE_FRAMES, MIN_OUTSIDE_FRAMES, PENDING_PACK_SECONDS,
    TRACK_TTL_FRAMES,
)

BBox = tuple[float, float, float, float]


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _center_distance(a: BBox, b: BBox) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def intersection_area(bbox: BBox, roi: BBox = BAG_ROI) -> float:
    return bbox_area((
        max(bbox[0], roi[0]), max(bbox[1], roi[1]),
        min(bbox[2], roi[2]), min(bbox[3], roi[3]),
    ))


def intersection_ratio(bbox: BBox, roi: BBox = BAG_ROI) -> float:
    """Fraction of the object box in the ROI (not IoU)."""
    area = bbox_area(bbox)
    return intersection_area(bbox, roi) / area if area > 0 else 0.0


def is_inside_bag(
    bbox: BBox, roi: BBox = BAG_ROI,
    overlap_threshold: float = BAG_OVERLAP_THRESHOLD,
) -> bool:
    if not all(isfinite(value) for value in bbox) or bbox_area(bbox) <= 0:
        return False
    cx, cy = bbox_center(bbox)
    return (
        roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]
    ) or intersection_ratio(bbox, roi) >= overlap_threshold


@dataclass(frozen=True)
class Detection:
    track_id: int
    class_name: str
    bbox: BBox


@dataclass
class TrackState:
    track_id: int
    class_name: str
    last_bbox: BBox
    was_inside_roi: bool
    packed: bool
    last_seen_frame: int
    seen_outside: bool = False
    inside_frames: int = 0
    outside_frames: int = 0
    # Zone version under which the outside evidence was collected. A zone
    # geometry change invalidates it: the product must be seen outside the
    # NEW zone before any inside streak can count.
    outside_zone_version: int = -1
    # Version under which the current consistent-inside streak was
    # collected (dynamic zone). A new outline restarts the streak, so items
    # are always evaluated afresh against the current footprint.
    inside_zone_version: int = -1
    # "Packing..." watch: monotonic deadline (seconds) while a
    # reliably-outside track stays hidden after vanishing at the bag
    # boundary (None = not pending).
    pending_deadline: float | None = None
    # Zone version when the pending watch started. Expiry under a different
    # version never counts: disappearance alone, or under obsolete geometry,
    # must not manufacture a packing event.
    pending_zone_version: int = -1


class PackingTracker:
    def __init__(
        self, roi: BBox = BAG_ROI,
        overlap_threshold: float = BAG_OVERLAP_THRESHOLD,
        min_inside_frames: int = MIN_INSIDE_FRAMES,
        track_ttl_frames: int = TRACK_TTL_FRAMES,
        min_outside_frames: int = MIN_OUTSIDE_FRAMES,
        zone_test=None,
        pending_pack_seconds: float = PENDING_PACK_SECONDS,
    ):
        if not all(isfinite(v) for v in roi) or bbox_area(roi) <= 0:
            raise ValueError("ROI must be a finite rectangle with positive area.")
        if not 0 < overlap_threshold <= 1:
            raise ValueError("Overlap threshold must be in (0, 1].")
        if min_inside_frames < 1 or min_outside_frames < 1 or track_ttl_frames < 1:
            raise ValueError("Frame thresholds must be positive.")
        self.roi = roi
        self.overlap_threshold = overlap_threshold
        self.min_inside_frames = min_inside_frames
        self.min_outside_frames = min_outside_frames
        self.track_ttl_frames = track_ttl_frames
        # Optional dynamic-zone predicate: zone_test(bbox) -> bool. When set
        # it replaces the fixed-ROI test; when None the legacy ROI applies.
        self.zone_test = zone_test
        # Monotonic geometry version of the current zone_test (-1 means
        # unversioned: legacy fixed-ROI behavior with no version checks).
        # Outside evidence only counts toward packing when it was collected
        # under the current version.
        self.zone_version = -1
        self.packing_paused = False
        if pending_pack_seconds <= 0:
            raise ValueError("Pending-pack wait must be positive.")
        self.pending_pack_seconds = pending_pack_seconds
        self.reset()

    def set_packing_paused(self, paused: bool) -> None:
        """Freeze zone-geometry confirmations (bag moving/lost/locating).

        While paused, sightings refresh geometry/recency but streak counters
        are frozen. The "Packing..." watch is product evidence, not zone
        geometry, so it still starts, cancels, and resolves on real time —
        that is what counts insertions that shift or lose the bag.
        """
        self.packing_paused = bool(paused)

    def note_zone_relocation(self) -> None:
        """Discard incomplete outside-to-inside streaks after the bag moved.

        A new outline restarts all product evidence: streak counters reset
        and pending watches clear, so visible items are evaluated afresh and
        disappearance alone can never count. Already confirmed packed
        counts and IDs are left intact.
        """
        for state in self.tracks.values():
            if state.packed:
                continue
            state.inside_frames = 0
            state.outside_frames = 0
            state.seen_outside = False
            state.outside_zone_version = -1
            state.inside_zone_version = -1
            state.pending_deadline = None
            state.pending_zone_version = -1

    def _pack_once(self, state: TrackState, events: list) -> None:
        """Count one packed item for a track (normal, pending, or merged)."""
        state.packed = True
        state.pending_deadline = None
        self.packed_ids.add(state.track_id)
        self.packed_counts[state.class_name] += 1
        event = {
            "track_id": state.track_id,
            "class_name": state.class_name,
            "event": "packed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_number": self.frame_number,
        }
        self.packed_events.append(event)
        events.append(event.copy())

    def _clearly_outside(self, bbox: BBox) -> bool:
        """True only when the box is well clear of the packing zone."""
        if self.zone_test is not None:
            try:
                clearly_outside = getattr(self.zone_test, "clearly_outside", None)
                if clearly_outside is not None:
                    return bool(clearly_outside(bbox))
                return not bool(self.zone_test(bbox))
            except Exception:
                return False
        return not is_inside_bag(bbox, self.roi, self.overlap_threshold)

    def reset(self) -> None:
        self.frame_number = 0
        self.tracks: dict[int, TrackState] = {}
        self.packed_counts: Counter[str] = Counter()
        self.packed_events: list[dict] = []
        # Preserve packed IDs after stale geometry is removed, until reset.
        self.packed_ids: set[int] = set()

    def update(self, detections: Iterable[Detection], now: float | None = None) -> list[dict]:
        """Advance exactly one processed frame, including empty frames.

        ``now`` is monotonic seconds for the "Packing..." watch (≈2 s of
        real time, independent of processed-frame rate); defaults to the
        current time.
        """
        if now is None:
            now = time.monotonic()
        self.frame_number += 1
        for track_id, state in list(self.tracks.items()):
            if self.frame_number - state.last_seen_frame > self.track_ttl_frames:
                del self.tracks[track_id]

        events = []
        seen_ids = set()
        for detection in detections:
            if (
                detection.class_name not in FOOD_CLASSES
                or detection.track_id in seen_ids
                or not all(isfinite(v) for v in detection.bbox)
                or bbox_area(detection.bbox) <= 0
            ):
                continue
            seen_ids.add(detection.track_id)
            if self.zone_test is not None:
                try:
                    inside = bool(self.zone_test(detection.bbox))
                except Exception:
                    inside = False
            else:
                inside = is_inside_bag(detection.bbox, self.roi, self.overlap_threshold)
            outside = self._clearly_outside(detection.bbox)
            state = self.tracks.get(detection.track_id)
            if state is None:
                state = TrackState(
                    track_id=detection.track_id,
                    class_name=detection.class_name,
                    last_bbox=detection.bbox,
                    was_inside_roi=inside,
                    packed=detection.track_id in self.packed_ids,
                    last_seen_frame=self.frame_number,
                )
                self.tracks[detection.track_id] = state
                if inside and not state.packed:
                    # Tracking-ID change mid-insertion: a new ID appearing
                    # inside next to a pending same-class track is the same
                    # physical item — count it once under the new ID.
                    for pending in self.tracks.values():
                        if (pending.pending_deadline is not None
                                and pending.class_name == state.class_name
                                and not pending.packed
                                and (self.zone_version < 0
                                     or pending.pending_zone_version == self.zone_version)
                                and _center_distance(pending.last_bbox, state.last_bbox) < 120.0):
                            pending.pending_deadline = None
                            pending.pending_zone_version = -1
                            self._pack_once(state, events)
                            break

            # Gaps and boundary jitter cannot count as stable transfer evidence.
            if self.frame_number - state.last_seen_frame > 1:
                state.inside_frames = 0
                state.outside_frames = 0
                state.inside_zone_version = -1
            if state.pending_deadline is not None and not state.packed:
                # Product evidence resolves on real time even while the zone
                # geometry is frozen: this counts insertions that move the bag.
                # A pending watch tied to an old outline never resolves: the
                # item must re-prove itself against the current geometry.
                if (self.zone_version >= 0
                        and state.pending_zone_version != self.zone_version):
                    state.pending_deadline = None
                    state.pending_zone_version = -1
                elif outside:
                    # Reappeared clearly outside: it never went in; cancel.
                    state.pending_deadline = None
                    state.pending_zone_version = -1
                elif inside:
                    # Reappeared inside after vanishing at the boundary:
                    # the hidden transfer completed while unseen.
                    self._pack_once(state, events)
                    state.last_bbox = detection.bbox
                    state.was_inside_roi = True
                    state.last_seen_frame = self.frame_number
                    continue
                # Rim band: keep waiting, accrue nothing meanwhile.
            if self.packing_paused:
                # Frozen zone state: the sighting refreshes geometry/recency
                # but streak counters stay frozen, so a zone jump alone can
                # never manufacture a streak for a stationary product.
                state.last_bbox = detection.bbox
                state.was_inside_roi = inside
                state.last_seen_frame = self.frame_number
                continue
            if outside:
                state.outside_frames += 1
                if state.outside_frames >= self.min_outside_frames:
                    state.seen_outside = True
                    state.outside_zone_version = self.zone_version
                state.inside_frames = 0
                state.inside_zone_version = -1
            elif inside:
                state.outside_frames = 0
                if self.zone_test is not None:
                    # Dynamic whole-bag zone: "mostly inside" (overlap with
                    # the filled footprint) for consistent frames counts,
                    # even when the camera never saw the item enter. The
                    # streak is versioned, so a new outline restarts it.
                    if state.inside_zone_version != self.zone_version:
                        state.inside_zone_version = self.zone_version
                        state.inside_frames = 1
                    else:
                        state.inside_frames += 1
                    if state.inside_frames >= self.min_inside_frames and not state.packed:
                        self._pack_once(state, events)
                # Legacy fixed-ROI path keeps the outside->inside transfer
                # below (versioned outside evidence required).
            else:
                # Rim hysteresis band (neither inside nor clearly outside):
                # hold position, accrue nothing either way.
                pass
            outside_current = (
                self.zone_version < 0
                or state.outside_zone_version == self.zone_version
            )
            if (self.zone_test is None and inside and state.seen_outside
                    and outside_current and not state.packed):
                state.inside_frames += 1
                if state.inside_frames >= self.min_inside_frames:
                    self._pack_once(state, events)

            state.last_bbox = detection.bbox
            state.was_inside_roi = inside
            state.last_seen_frame = self.frame_number

        for track_id, state in self.tracks.items():
            if track_id in seen_ids:
                continue
            if state.packed:
                continue
            if state.pending_deadline is not None:
                if now >= state.pending_deadline:
                    # Hidden the whole ~2 s wait: the boundary disappearance
                    # was a completed transfer. A watch tied to an old
                    # outline never fires on expiry: disappearance alone, or
                    # under obsolete geometry, must not count.
                    if (self.zone_version < 0
                            or state.pending_zone_version == self.zone_version):
                        self._pack_once(state, events)
                    else:
                        state.pending_deadline = None
                        state.pending_zone_version = -1
                continue
            # The watch starts on product evidence alone, even while the
            # zone is frozen: insertions routinely move the bag.
            if state.seen_outside and not self._clearly_outside(state.last_bbox):
                # Reliably outside, vanished at the boundary (hand
                # occlusion): watch ~2 s of real time instead of dropping
                # the trail. Vanishing while clearly outside starts no watch.
                state.pending_deadline = now + self.pending_pack_seconds
                state.pending_zone_version = self.zone_version
                continue
            state.inside_frames = 0
            state.outside_frames = 0
            state.inside_zone_version = -1
        return events

    def _zone_inside(self, bbox: BBox) -> bool:
        """Mirror of the per-detection inside test in update()."""
        if self.zone_test is not None:
            try:
                return bool(self.zone_test(bbox))
            except Exception:
                return False
        return is_inside_bag(bbox, self.roi, self.overlap_threshold)

    def _zone_overlap(self, bbox: BBox) -> float:
        """Share of the product box inside the footprint (diagnostics)."""
        overlap = getattr(self.zone_test, "overlap_fraction", None)
        if callable(overlap):
            try:
                return max(0.0, min(1.0, float(overlap(bbox))))
            except Exception:
                return 0.0
        try:
            return max(0.0, min(1.0, float(intersection_ratio(bbox, self.roi))))
        except Exception:
            return 0.0

    def track_diagnostics(self) -> list:
        """Per-track packing state for visible uncounted products.

        Each entry reports the footprint-overlap share, the inside streak,
        and the reason counting is currently blocked. Bounded by the live
        track count; no video.
        """
        out = []
        for state in self.tracks.values():
            pending = state.pending_deadline is not None
            if state.packed:
                blocked = "already counted"
            elif self.packing_paused:
                blocked = "packing paused (bag uncertain)"
            elif pending:
                blocked = "hidden at boundary, awaiting reappearance"
            elif self._zone_inside(state.last_bbox):
                blocked = ("inside %d/%d — still confirming"
                           % (state.inside_frames, self.min_inside_frames))
            elif self._clearly_outside(state.last_bbox):
                blocked = "outside the footprint"
            else:
                blocked = "at footprint rim"
            out.append({
                "track_id": state.track_id,
                "class_name": state.class_name,
                "overlap": round(self._zone_overlap(state.last_bbox), 3),
                "inside": "%d/%d" % (state.inside_frames,
                                     self.min_inside_frames),
                "seen_outside": bool(state.seen_outside),
                "pending": bool(pending),
                "packed": bool(state.packed),
                "frames_since_seen": self.frame_number - state.last_seen_frame,
                "blocked": blocked,
            })
        return out

    def snapshot(self) -> dict:
        return {
            "packed_counts": dict(self.packed_counts),
            "packed_total": sum(self.packed_counts.values()),
            "event_count": len(self.packed_events),
            "last_event": self.packed_events[-1].copy() if self.packed_events else None,
            "track_diag": self.track_diagnostics(),
        }
