# Dynamic whole-bag zone — MVP implementation report

Target is the **whole bag** (not the opening). The camera page now shows a
fitted green contour that follows the bag, and packing decisions use the
filled bag footprint. No frames were uploaded anywhere; the raw video
(`vidoe_bag.mp4`, 1866 frames / 60 fps / 1280x720) stayed local and is not
committed. Bottle, fruit, OCR, Jev, and replay behavior are preserved
(full suite: **191 tests OK**, 2 pre-existing skips).

## 0. Urgent fix first: invisible products (GPU detector silently blind)

Symptom: bottles/cans fully visible for seconds, no product boxes on the
page. Traced stage by stage on the same frames (raw worker rows →
ByteTrack tracks → page payload), with the dynamic bag enabled:

- Worker rows (GPU): **0 rows on every frame**, including the repo's own
  `outputs/test-bottles*.jpg`. Same checkpoint on CPU: `Bottle 0.12–0.63`
  plus desk/keyboard rows. The Paddle GPU path (driver 580 / CUDA 13 vs the
  pinned PaddlePaddle 3.3.1 build) runs without errors and returns nothing.
- ByteTrack (`bytetrack-ppyoloe.yaml`: new tracks need conf >= 0.50) and
  `extract_food_detections` (drops ID-less boxes) then have nothing to work
  with — the page can show nothing. **Bag pause was never the cause**:
  it gates packing counts only; the display path is independent.
- Fix: run the production detector on **CPU** (`LIGHTSTORE_DEVICE=cpu`,
  the historically validated Sep-21 configuration, ~145 ms/frame).
  Verified live: `test-bottles1.jpg` → 2 Bottle tracks + `visible_counts`
  immediately, and bottles display while the bag zone is `stable`.
- Residual layer (not changed): sub-0.50 detections never initiate tracks
  (uncalibrated starting values per the yaml comment). Transparent top-down
  bottles in the bag video score ≤ 0.19 — a genuine detector limitation,
  pre-existing. Next step if weak-view products still hide: measure live
  bottle/can confs, then calibrate `new_track_thresh` with evidence.

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
  `BagZoneTracker` state machine (`locating → stable → grace → moving /
  lost`, plus `unavailable`), footprint geometry (largest component → close →
  hole-fill → fitted contour), `ZoneTest` packing predicate (center-in OR
  >= 0.30 overlap, same semantics as the old ROI rule) with rim hysteresis.
- **`config.py`**: `BAG_ZONE_MODE` (`dynamic` default, `fixed` opt-out),
  model/prompt pins, `BAG_HEARTBEAT_FRAMES = 10` (~6 s),
  `BAG_ACQUIRE_STABLE = 3`, `BAG_ADOPT_IOU = 0.50`,
  `BAG_RELOCK_IOU = 0.70`, `BAG_MOTION_THRESHOLD = 12.0` (stable pairs
  3.5–5.2 vs moving pairs 20–41), `BAG_MISSES_TO_LOSE = 2`,
  `BAG_GRACE_PERIOD_S = 2.0` (wall clock), `BAG_RIM_MARGIN_PX = 16.0`.
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

## 2b. Bag-loss grace period (2 s)

When a locked bag misses twice, the zone enters `grace` instead of going
lost: the last outline stays visible (gray `BAG (reacquiring…)` ghost),
packing **continues** in that frozen area, and every processed frame tries
to reacquire (camera movement is covered: the motion trigger already forces
per-frame inference when the scene shifts). Streaks restart at grace entry,
so only fresh observations under the ghost can complete. Same-place
reappearance (footprint IoU >= 0.5, no geometry change) updates the outline
and continues normally with no extra wipe; reappearance elsewhere follows
the moving path (freeze + pause + relock, evidence restarted). After 2 s
wall-clock without reacquisition the outline is removed, packing pauses,
and the page shows `BAG LOST - packing paused`. A bag that slides onto a
stationary product mid-grace cannot pack it: relocation + versioned outside
evidence + rim hysteresis all still apply (unit test
`test_bag_move_during_grace_creates_no_false_pack`, plus the video check
below).

Harness: `reports/bag-dynamic-zone/verify_dynamic.py` (committed) — real
localizer + real zone + real tracker; scripted products: track 1 genuine
transfer (outside ×3 → live centroid after lock), track 2 stationary bottle
at a table spot the moving bag sweeps across.

| run | processed | inferences (share) | locating / stable / grace / moving / lost | genuine packs | stationary packs |
|---|---|---|---|---|---|
| stride 5 (dense) | 374 | 41 (11%) | 87 / 216 / 20 / 5 / 46 | 1 (@frame 93) | **0** (266 crossed frames) |
| stride 34 (~live cadence) | 55 | 13 (24%) | 20 / 22 / 8 / 5 / 0 | 1 (@frame 26) | **0** (35 crossed frames) |

Contour examples: `evidence/dynamic_f0750_stable.jpg` (fitted outline),
`evidence/dynamic_f1600_grace.jpg` (gray ghost + reacquiring label, counts
kept), `evidence/dynamic_f1700_lost.jpg` (`BAG LOST - packing paused`),
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

Server + page run on port 8001 with `LIGHTSTORE_DEVICE=cpu` (GPU Paddle
path is blind — see §0; revisit only after a Paddle/CUDA environment fix).
Procedure: open the page → Start → `LOCATING BAG…` → place and spread the
bag (≈2 s) → green fitted contour + `Bag: tracking ✓` → move a bottle from
outside into the bag → `PACKED` after ~3–4 s inside → briefly cover the bag
(< 2 s) → gray ghost outline, counting continues, `Bag: reacquiring…` →
uncover → contour updates, continues → drag the bag while a bottle sits
inside → amber `BAG MOVING - packing paused`, no new packs, counts kept →
gather the bag (> 2 s) → `BAG LOST - packing paused` → spread it again →
re-locks and resumes.
