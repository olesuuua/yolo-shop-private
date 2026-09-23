# Dynamic bag zone: evidence-based feasibility test (investigation only)

**Status:** investigation. No production behavior changed. Fixed `BAG_ROI`
(`config.py:143`) remains the packing zone; `tracking.py` is untouched.
Jev stayed disabled; bottle identification is out of scope for this test.
The raw video (`vidoe_bag.mp4`, 1866 frames, 60 fps, 1280x720, ~31 s) was
**not** committed and **not** uploaded anywhere — all runs used the local file.

## 1. Pipeline review (as found)

- Packing zone: fixed `BAG_ROI = (220, 140, 420, 380)` in `config.py:143`,
  in 640x480 detection pixels. `tracking.py:40` (`is_inside_bag`) counts a
  box as inside when its center is in the ROI **or** its box/ROI overlap
  >= `BAG_OVERLAP_THRESHOLD` (0.30). `PackingTracker.update` (`tracking.py:101`)
  requires `MIN_OUTSIDE_FRAMES=3` outside then `MIN_INSIDE_FRAMES=3` inside
  before emitting `packed`.
- Premise correction: the **default production detector
  (`ppyoloe_objects365`) has no `Plastic bag` class.** The 365-label catalog
  (`objects365_classes.json`) contains no such label, and its grocery
  `Packaging` group is only `{Bottle, Storage box}`. Verified at runtime:
  `'Plastic bag' in FOOD_CLASSES` is `False` under the default profile.
  The claim holds only for the `openimages` profile (`yolov8n-oiv7.pt`,
  601 classes, `Plastic bag` id 394), where `'Plastic bag' in FOOD_CLASSES`
  is `True` — i.e. there it **would currently be counted as a product**
  (`extract_food_detections` in `vision.py:45`, `PackingTracker.update`
  filter in `tracking.py:111`, `visible_counts`/`packed_counts`). Any
  dynamic-zone design must exclude the bag class from product counting
  explicitly; reusing the openimages profile as-is would count the bag
  itself (and its contents jitter) as packed items.
- Therefore the existing-detector test below uses the openimages profile —
  the only configured detector that can output `Plastic bag` at all.

## 2. Method

Harness: `reports/bag-dynamic-zone/bag_feasibility.py` (reusable, committed).
It resizes frames to 640x480 exactly like `FrameProcessor.process_detect`,
runs `yolov8n-oiv7.pt` (same HF repo/revision pin as `config.py`) at
`CONF_THRESHOLD=0.10`, and compares each `Plastic bag` box against a
human-marked **bag opening** box (the hole products pass through, not the
bag body; `None` where no opening is definable). It also runs a synthetic
moving-zone simulation through the real `PackingTracker`.

Reproduce (weights download from Hugging Face on first run; video stays local):

```bash
.venv/bin/python reports/bag-dynamic-zone/bag_feasibility.py \
    --video /path/to/vidoe_bag.mp4
.venv/bin/python reports/bag-dynamic-zone/bag_feasibility.py \
    --video /path/to/vidoe_bag.mp4 --reproduce
```

Full per-frame output: `evidence/results.json`. Annotated frames
(yellow = fixed `BAG_ROI`, green = human opening, magenta = detector box):
`evidence/stage_f*.jpg`.

## 3. Results per stage (conf >= 0.10, CPU)

| frame | stage | bag detected | conf | box (640x480) | IoU vs opening | center err |
|---|---|---|---|---|---|---|
| 0 | empty table (control) | no | — | — | — | — |
| 150 | initial acquisition, motion blur | **no (miss)** | — | — | — | — |
| 300 | placement, hand spreading bag | **no (miss)** | — | — | — | — |
| 600 | placement settling | yes | 0.133 | [47, 154, 398, 479] | 0.48 | ~65 px |
| 750 | stable open, no hand | yes | 0.161 | [73, 144, 486, 457] | 0.49 | ~55 px |
| 900 | stable open, no hand | **no (miss)** | — | — | — | — |
| 1050 | product entering (water bottle) | yes | 0.186 | [122, 146, 557, 479] | 0.63 | — |
| 1200 | bottle lying inside | yes | 0.392 | [70, 100, 578, 460] | 0.71 | — |
| 1350 | product entering (cola bottle) | **no (miss)** | — | — | — | — |
| 1500 | stable, two bottles inside | **no (miss)** | — | — | — | — |
| 1650 | hand obstruction / gathering | no (correct-ish) | — | — | — | — |
| 1800 | bag moved, opening collapsed | no (correct) | — | — | — | — |

Latency (yolov8n-oiv7, CPU, 640 px, warm): median **~30 ms** (min 28, max 443
cold first frame). Production reference for interval planning:
`reports/video-comparison/comparison-summary.json` (v1-ocr-cold) measured
**~0.57 s median between processed frames (~1.75 fps)** with PP-YOLOE
`detect_ms` median ~167 ms on CPU — the end-to-end cadence, not just inference.

## 4. Failure analysis

- **Missed detections (recall):** 5 of 9 bag-present frames with a definable
  opening missed at the production threshold (f150, f300, f900, f1350,
  f1500 — 56% miss). Lowering the threshold does not rescue them: at
  conf=0.01 the same frames still produce no `Plastic bag` box (only a
  0.071 `Handbag` false label at f1500). The white translucent T-shirt bag
  on a white table, top-down, is near-invisible to this class.
- **False detections (precision):** the same frames fire `Person`, `Clothing`
  (0.25–0.29 on the bag itself), `Footwear`, `Bed`, `Computer keyboard` at
  conf >= 0.10 (see `results.json` `other_detections`). The bag region is
  simultaneously labeled `Clothing` with nearly the same box as `Plastic bag`
  when the latter fires — tracking a bag id would compete with these.
- **Opening-location error:** when detected, the box covers the **whole bag
  body plus margin** (e.g. f750 box is 412x313 px vs a ~300x210 px opening),
  IoU vs the opening only 0.48–0.71, bottom edge up to ~100 px below the
  opening rim. A whole-body box is not a usable opening proxy: shrinking it
  by a fixed inset would be guesswork that breaks as the bag deforms.
- **Bottles invisible to this profile:** no `Bottle` detection in any test
  frame at conf 0.10 despite two clearly visible bottles — the openimages
  profile would degrade product detection too. It is a bag-measurement
  instrument only, never a replacement detector.
- **False packing events with a moving zone:** demonstrated, not just
  theorized. Simulation through the unmodified `PackingTracker`: a
  **stationary** bottle box is seen outside the zone for 3 frames
  (`seen_outside=True`), the zone is then lost and reappears shifted onto
  the motionless bottle for 3 frames → a `packed` event fires at processed
  frame 8 for a product that never moved. In this recording the concrete
  trigger moments are f1350 (cola entering exactly while the bag is
  undetected — a re-acquired shifted zone would swallow the stationary
  water bottle) and f1650–f1800 (zone jumps as the hand gathers/moves the
  bag). **Verdict: a moving zone gated only by MIN_INSIDE_FRAMES is unsafe;
  loss/motion must suspend packing decisions.**

## 5. Roboflow candidate (exact link, no substitution)

Link investigated (page reachable, no upload performed):
`https://universe.roboflow.com/skenj-a/plastic-bag-g03mw-mijil` —
**`skenj-a / plastic bag`**, task **Instance Segmentation**, model tag
`yolov11` / **`yolov11n-seg`** (YOLOv11 Nano seg), 1 class
(`plastic bag - v2 2023-06-27 3:00pm`), dataset version v1 (1051 images;
train 733 / valid 213 / test 105; Auto-Orient + Stretch 640x640, no
augmentations), checkpoint lineage COCO-seg, reported **mAP@50 71.3%,
precision 99.9%, recall 49.8%**, license **CC BY 4.0** (attribution
required). Listed inference method is **hosted only** (Roboflow serverless
`InferenceHTTPClient`, `model_id="plastic-bag-g03mw-mijil/1"`, API key
required) — no direct `.pt`/weights download is offered on the page, and
using it would upload frames to a cloud service, which is **blocked by the
no-upload constraint** unless you explicitly approve. Annotations, from the
preview overlays and CCTV-style filenames (`15_X001_C…`), read as
**whole-bag instance masks**, not bag-opening polygons — an opening-level
zone would still need derivation/validation. Not tested on this recording
for exactly these reasons. Next step if you want it evaluated: approve (a)
a cloud-inference trial or (b) forking/downloading the dataset for local
training, then rerun this harness against it.

## 6. Recommendation — next implementation plan (not implemented)

Do not track the bag every frame. Proposed state machine, separate from
production until proven:

1. **Acquisition:** on session start / bag-lost, run a bag localizer (not
   the openimages class — see §4; candidate: Roboflow model locally, or a
   small custom opening-segmentation head) each processed frame until the
   opening polygon is stable (IoU > 0.85 across 3 consecutive processed
   frames, ~2 s at the measured ~1.75 fps cadence). Only then lock the zone.
2. **Stable tracking without inference:** while locked, run **zero** bag
   inference; instead do cheap frame-difference motion inside an expanded
   zone bbox each processed frame. Re-run the localizer at a slow heartbeat
   (every ~10 processed frames ≈ ~6 s) to correct drift.
3. **Motion/loss gating:** if inter-check motion exceeds threshold or the
   heartbeat localizer finds no opening (IoU < 0.5 vs locked zone, or
   confidence below calibrated cut), mark zone `LOST/MOVING`, **freeze the
   `PackingTracker` (queue but do not commit inside-counts)** and show
   "bag moving — packing paused". Require re-acquisition (step 1) before
   resuming; never count transitions that occurred while gated.
4. **Counting hygiene:** the bag class must never enter `FOOD_CLASSES`;
   keep a dedicated single-purpose bag model/session whose boxes feed only
   the zone, never `extract_food_detections` / `packed_counts`.
5. **Acceptance test:** rerun this harness: require >=90% opening recall on
   definable frames, IoU vs opening >= 0.8, zero false packs in the f1350
   and f1650–f1800 windows, and heartbeat cost within the ~1.75 fps budget.

## 7. Artifacts

- `bag_feasibility.py` — harness (this dir).
- `evidence/results.json` — machine-readable per-frame results.
- `evidence/stage_f*.jpg` — 12 annotated frames. Key examples: `stage_f0750.jpg`
  (detected but whole-body box, IoU 0.49), `stage_f0900.jpg` (stable bag,
  missed), `stage_f1200.jpg` (best case, conf 0.39, still body box),
  `stage_f1350.jpg` (cola entering, missed — false-pack risk window).
