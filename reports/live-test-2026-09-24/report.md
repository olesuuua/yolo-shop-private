# Live test 2026-09-24 — review, fixes, verification

Captures: `screenshots/` (8 page shots, 08:45–08:50) and the user screencast
(kept out of git; ~11 s at 09:33–09:34). Server: port 8001, CPU detector.

## 1. Packing miss trace (orange bottle in bag, count 0)

Timeline from the shots: both sodas outside + bag lost (08:45) → upright
bag tracking 0.564, bottles beside it (08:46) → orange overlapping the bag
while the bag reports moving/paused (08:47) → full-screen green outline at
conf 0.161 (08:48) → bag lost, orange standing in the bag mouth, cola
Recognized, count 0 (08:49–08:50). The video case counted via "Packing…".

Four stacked mechanisms, all fixed:
1. **Pause freeze**: insertion shifts the bag → moving/lost → streaks and
   the old frame-count watch froze. Fix: the watch is now **~2 s of real
   time** (`PENDING_PACK_SECONDS`, monotonic clock passed per processed
   frame) and resolves on product evidence **even while the zone is
   moving/lost**. Bag grace (counting, unpaused) is honored as before.
2. **Version invalidation on mask wobble**: the entering hand/bottle shifts
   the seg mask → heartbeat adopts bump the zone version → outside evidence
   goes stale. Pending (pause-independent) is now the primary insertion
   path when geometry churns.
3. **Full-screen mask flood**: rejected (see §3) → grace keeps the last
   good outline instead of adopting garbage.
4. **ID churn at occlusion**: the new-ID-inside merge (same class, 120 px)
   is retained; continuous hand motion is what the paced replay validates.

Behind-bag discipline kept: watches start only on vanish-at-boundary with
reliable outside history (never for clearly-outside disappearances);
visible rim-touching placements hold in the hysteresis band (unit-tested,
paced scenario C). An **occluded-persistent** behind-bag placement still
counts falsely — top-down geometry cannot separate it from a hidden
insertion (documented, tested, reconfirmed live below).

## 2. Persistent packed display (verified live, no change needed)

The screencast's Details panel proves the requested behavior already works:
after the flash, Counters keeps **Packed 1 · "Unidentified bottle" ·
"Last packed: Unidentified bottle · 9:34:12 AM"** while the bottle leaves
Recognition. The banner now also names the item (`PACKED: <name>`) via the
packed-label cache instead of `Bottle #3`. Late OCR renames the row without
recounting (unit-tested, incl. post-prune stickiness).

## 3. Implausible bag masks (rejected)

08-48-43's screen-filling outline at conf 0.161 is now rejected as a miss:
footprint fraction > 0.70, or conf < 0.20 with fraction > 0.45
(white-bag stable masks sit ~0.3–0.45 at 0.14+, unaffected). Rejections
surface as `rejected_masks` in the zone snapshot. Unit-tested (flood
rejected, generous bar still locks).

## 4. Stretch fix (letterbox, all sides aligned)

16:9 uploads were plain-stretched into 640×480. Now both server
(`letterbox_frame`: centered 640×360 + 60 px bars; identity for 4:3) and
browser (same `letterboxRect` math on the detection canvas) preserve
aspect, and OCR mapping inverts the letterbox on both ends
(`detect_to_upload` / `mapCropRect`), so boxes, footprints, and crops stay
in one consistent 640×480 space. Viewer CSS already used `object-fit:
contain` (no change needed). JPEG qualities untouched (0.75 detect,
0.85 crops). Tests: rewritten widescreen/portrait expectations + exact
1920×1080 mapping cases, both suites green.

## 5. OCR: actual crops, raw results, why orange stalled

- Panel thumbnails ARE the OCR input (`/api/ident-crop/{id}` bytes,
  CSS-sized): live track 4's crop is the whole 641×696 object box, so the
  tiny thumbnail does not hide pixels from OCR.
- Raw evidence (screencast Details): orange [#43] after 4 runs —
  "Добрый/апельсин" mixed with "Добрыд/анельсу"; cola [#1] similar
  fragments. Sharpness 232–257 (gate 15/30) is excellent — **sharpness and
  JPEG quality are not the limiters; keep q85.**
- Stall mechanism: Добрый has 0.5 л/1 л brand siblings, so completion
  requires volume text ("1 л"), which never reads on the curved specular
  1 л label → "Likely match" indefinitely; cola completed once its volume
  read ("seen on label"). Jev itself is healthy (52 attempts/47 ok).
- No crop/confidence rules changed. Smallest next option (explicit user
  call): brand-text-only completion for sizes with no readable volume, or
  holding the bottle straight-on longer.

## 6. Verification (observed live vs automated, distinguished)

Observed live (user captures): clear insertion counted via "Packing…"
(09:34); sidebar keeps "Unidentified bottle" persistently; Camera readout
shows the 1080p preference working (**Camera 1920×1080**); cola Recognized.
Paced end-to-end replay through the real server (real detector + zone +
timing, composite bottle sprite, ~0.6 s cadence):

| run | result |
|---|---|
| A clear insertion (gradual sweep) | **1 × "Unidentified bottle"**, 1 event, zone stable |
| B hand-hidden (rim → 3.6 s hidden → inside) | **1 × "Unidentified bottle"** via pending (seen in payload), zone moving at times — pause-independent path proven |
| C visibly beside bag (~5 s) | **0**, no events, no watch |
| D behind-bag occluded (rim → hidden 4.2 s) | **1 (documented false)**, pending seen |

Harness: `verify_paced.py` (+ `paced_results.json`). Automated: 204
backend + 34 frontend tests green. Teleport-style jumps were found to
defeat history unrealistically (no watch starts on clearly-outside
vanishings — correct); the harness moves the sprite in 27 px hand-like
steps. Still needing the live camera: blue-bag suppression re-confirm,
apple pack, 1080p OCR hit rate, real behind-bag run.

## 7. Remaining limitations

Occluded-persistent behind-bag false count; gathered-bag following;
sub-0.50 views never initiate tracks; single cold-start frame can show no
tracks; sibling-volume rule stalls curved-label recognition.
