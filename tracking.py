"""Geometry and packing events, independent of YOLO, OpenCV and FastAPI."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Iterable

from config import (
    BAG_OVERLAP_THRESHOLD, BAG_ROI, FOOD_CLASSES,
    MIN_INSIDE_FRAMES, MIN_OUTSIDE_FRAMES, TRACK_TTL_FRAMES,
)

BBox = tuple[float, float, float, float]


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


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


class PackingTracker:
    def __init__(
        self, roi: BBox = BAG_ROI,
        overlap_threshold: float = BAG_OVERLAP_THRESHOLD,
        min_inside_frames: int = MIN_INSIDE_FRAMES,
        track_ttl_frames: int = TRACK_TTL_FRAMES,
        min_outside_frames: int = MIN_OUTSIDE_FRAMES,
        zone_test=None,
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
        self.reset()

    def set_packing_paused(self, paused: bool) -> None:
        """Freeze new packing confirmations (bag moving/lost/locating).

        Outside observations still accrue; inside streaks cannot advance
        while paused, so a zone jump alone can never manufacture an event.
        """
        self.packing_paused = bool(paused)

    def note_zone_relocation(self) -> None:
        """Discard incomplete outside-to-inside streaks after the bag moved.

        A stationary product under a jumping zone must re-prove itself with
        fresh observations once the bag stabilizes. Already confirmed packed
        counts and IDs are left intact.
        """
        for state in self.tracks.values():
            if state.packed:
                continue
            state.inside_frames = 0
            state.outside_frames = 0
            state.seen_outside = False
            state.outside_zone_version = -1

    def reset(self) -> None:
        self.frame_number = 0
        self.tracks: dict[int, TrackState] = {}
        self.packed_counts: Counter[str] = Counter()
        self.packed_events: list[dict] = []
        # Preserve packed IDs after stale geometry is removed, until reset.
        self.packed_ids: set[int] = set()

    def update(self, detections: Iterable[Detection]) -> list[dict]:
        """Advance exactly one processed frame, including empty frames."""
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
                try:
                    clearly_outside = getattr(self.zone_test, "clearly_outside", None)
                    outside = bool(clearly_outside(detection.bbox)) \
                        if clearly_outside is not None else not inside
                except Exception:
                    outside = not inside
            else:
                inside = is_inside_bag(detection.bbox, self.roi, self.overlap_threshold)
                outside = not inside
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

            # Gaps and boundary jitter cannot count as stable transfer evidence.
            if self.frame_number - state.last_seen_frame > 1:
                state.inside_frames = 0
                state.outside_frames = 0
            if self.packing_paused:
                # Frozen zone state: the sighting refreshes geometry/recency
                # but contributes zero transfer evidence either way, so a
                # zone jump alone can never manufacture (or preserve) a
                # streak for a stationary product.
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
            elif inside:
                state.outside_frames = 0
            else:
                # Rim hysteresis band (neither inside nor clearly outside):
                # hold position, accrue nothing either way.
                pass
            outside_current = (
                self.zone_version < 0
                or state.outside_zone_version == self.zone_version
            )
            if inside and state.seen_outside and outside_current and not state.packed:
                state.inside_frames += 1
                if state.inside_frames >= self.min_inside_frames:
                    state.packed = True
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

            state.last_bbox = detection.bbox
            state.was_inside_roi = inside
            state.last_seen_frame = self.frame_number

        for track_id, state in self.tracks.items():
            if track_id not in seen_ids and not self.packing_paused:
                state.inside_frames = 0
                state.outside_frames = 0
        return events

    def snapshot(self) -> dict:
        return {
            "packed_counts": dict(self.packed_counts),
            "packed_total": sum(self.packed_counts.values()),
            "event_count": len(self.packed_events),
            "last_event": self.packed_events[-1].copy() if self.packed_events else None,
        }
