# Dynamic whole-bag zone — MVP implementation report

Target is the **whole bag** (not the opening). The camera page now shows a
fitted green contour that follows the bag, and packing decisions use the
filled bag footprint. No frames were uploaded anywhere; the raw video
(`vidoe_bag.mp4`, 1866 frames / 60 fps / 1280x720) stayed local and is not
committed. Bottle, fruit, OCR, Jev, and replay behavior are preserved
(full suite: **188 tests OK**, 2 pre-existing skips).

## 1. Local YOLOE-26n segmentation test (raw frames, no cloud)

Checkpoint `weights/yoloe-26n/yoloe-26n-seg.pt` (sha verified against the
pinned `yoloe26n` digest), loaded as a **segmentation** model with a text
prompt — separate from the product profile, whose `yoloe-26n.yaml`
detection-only adaptation has no mask head. Production product detector
(ppyoloe) is unchanged.

Prompt comparison, 11 stage frames, conf >= 0.10, 640 px, CPU:

| prompt | spread-bag stages detected | empty f0 | notes |
|---|---|---|---|
| `plastic bag` | 7/8 (misses: motion-blur entry f150; gathered f1650/f1800) | clean | one box/frame, conf 0.14–0.45 |
| `plastic bag, shopping bag, bag` | 8/8 + fragments | clean | higher conf (to 0.80) but duplicate boxes |
| `transparent plastic bag` | 8/8 + fragments | clean | conf to 0.89, 2–3 boxes/frame |

Single prompt `plastic bag` selected: cleanest output. Masks cover the whole
bag (fill 0.79–0.83 of bbox on stable frames; holes where hand/bottles
intrude are filled into the footprint). Dense scan (every 15th frame, 125
samples): 73 raw detections; tiny false fragments (frac 0.01–0.08, e.g.
table edge) killed by the `BAG_MIN_FRAC = 0.08` area gate. Seg inference:
**median ~26 ms CPU** (max 34 ms) — cheap vs the ~167 ms production
detector and the ~0.57 s end-to-end processed-frame cadence (~1.75 fps).

Movement tracking: centroid path follows the bag through the spread-bag
move (f1500→1650); once gathered into a ball the prompt no longer fires —
**known limitation** (see §5), handled by an explicit LOST + paused state.

## 2. What was built

- **`bag_zone.py`** (new): `BagLocalizer` (seg model wrapper),
  `BagZoneTracker` state machine (`locating → stable → moving / lost`,
  plus `unavailable`), footprint geometry (largest component → close →
  hole-fill → fitted contour), `ZoneTest` packing predicate (center-in OR
  >= 0.30 overlap, same semantics as the old ROI rule) with rim hysteresis.
- **`config.py`**: `BAG_ZONE_MODE` (`dynamic` default, `fixed` opt-out),
  model/prompt pins, `BAG_HEARTBEAT_FRAMES = 10` (~6 s),
  `BAG_ACQUIRE_STABLE = 3`, `BAG_ADOPT_IOU = 0.50`,
  `BAG_RELOCK_IOU = 0.70`, `BAG_MOTION_THRESHOLD = 12.0` (stable pairs
  3.5–5.2 vs moving pairs 20–41), `BAG_MISSES_TO_LOSE = 2`,
  `BAG_RIM_MARGIN_PX = 16.0`.
- **`tracking.py`**: `PackingTracker` accepts `zone_test`; `set_packing_paused`
  freezes **all** transfer evidence while paused (a pause can neither create
  nor preserve a streak); `note_zone_relocation` discards incomplete streaks
  (packed counts/IDs intact); outside evidence is versioned
  (`outside_zone_version`) so a zone-geometry change invalidates it.
  Legacy fixed-ROI path is byte-identical when no zone is attached.
- **`vision.py`**: `FrameProcessor` builds the zone (load failure →
  `unavailable`, packing paused, banner — never a silent fixed fallback),
  updates it before the packing tracker each frame, draws the contour +
  status banners, exposes `bag_zone` in detection responses and
  `/api/session`. The bag class never enters product tracks, OCR crops, or
  counts (separate model, separate path).
- **Page** (`static/index.html`, `static/app.js`): bag status line
  (`tracking ✓ / moving — packing paused / lost — packing paused /
  locating…`) next to the live view.
- **Zone discipline**: acquisition runs inference every processed frame
  until 3 mutually consistent masks lock; stable runs heartbeat (every 10th
  frame) + motion trigger; loss/motion re-checks every frame until relock.
  Every adopt carries its frame id — an older prediction can never move the
  zone backward (unit-tested). Ordinary jitter (footprint IoU >= 0.9)
  refreshes the contour in place without invalidating product evidence;
  larger changes bump the zone version.

## 3. Verification on the recorded video

Harness: `reports/bag-dynamic-zone/verify_dynamic.py` (committed) — real
localizer + real zone + real tracker; scripted products: track 1 genuine
transfer (outside ×3 → live centroid after lock), track 2 stationary bottle
at a table spot the moving bag sweeps across.

| run | processed | inferences (share) | locating / stable / moving / lost | genuine packs | stationary packs |
|---|---|---|---|---|---|
| stride 5 (dense) | 374 | 41 (11%) | 87 / 216 / 5 / 66 | 1 (@frame 93) | **0** (266 crossed frames) |
| stride 34 (~live cadence) | 55 | 13 (24%) | 20 / 22 / 5 / 8 | 1 (@frame 26) | **0** (35 crossed frames) |

Contour examples: `evidence/dynamic_f0750_stable.jpg` (fitted outline),
`dynamic_f1600_lost.jpg` (`BAG LOST - packing paused`, packed count kept),
plus locating/stable frames across all stages and `dynamic_results.json`.

Three real failure mechanisms were found by this harness and fixed (each
reproduced before the fix, gone after — unit tests pin all three):

1. **Initial-lock swallow**: bag placed over a stationary product packed it
   (NoZone "outside" evidence + first lock). Fix: every non-steady adopt
   fires relocation. Test: `test_outside_evidence_goes_stale…` + verify.
2. **Gradual creep**: slow zone growth onto a stationary product packed it
   with no relocation signal. Fix: versioned outside evidence (stale after
   any significant geometry change).
3. **Rim flicker**: a reaching hand carved ~20 px off the mask rim for ~10
   frames, flipping a rim-resting bottle outside→inside. Fix: 16 px rim
   hysteresis — entries use the exact footprint, outside evidence must clear
   the dilated rim.

No missed genuine packing (transfer completes ~3.5 s after entering a
stable zone) and no false packing in any run after the fixes.

## 4. Reproduce

```bash
.venv/bin/python reports/bag-dynamic-zone/verify_dynamic.py --video /path/to/vidoe_bag.mp4 [--step 5]
.venv/bin/python -m unittest discover -s tests -p 'test_bag_zone.py' -v
.venv/bin/python -m unittest discover -s tests  # full suite
LIGHTSTORE_BAG_ZONE=fixed .venv/bin/python -m unittest discover -s tests -p 'test_app.py'  # legacy path
```

## 5. Known limitation + smallest next option

A **gathered/crumpled moving bag is not followed**: the `plastic bag` prompt
stops firing, the zone goes LOST, packing pauses with a banner, and already
confirmed counts are kept. The outline is therefore never presented as
working when it is not. Smallest viable next step if following the gathered
bag matters: add 1–2 visual-prompt embeddings (or a few dozen local
fine-tune masks) of the gathered state to the same local seg checkpoint —
no architecture change; this harness re-measures in seconds.

## 6. Live test

Server + page are left running (see handoff message for URL/commit).
Procedure: open the page → Start → `LOCATING BAG…` → place and spread the
bag (≈2 s) → green fitted contour + `Bag: tracking ✓` → move a bottle from
outside into the bag → `PACKED` after ~3–4 s inside → drag the bag while a
bottle sits inside → amber `BAG MOVING - packing paused`, no new packs,
counts kept → gather the bag → `BAG LOST - packing paused` → spread it
again → re-locks and resumes.
