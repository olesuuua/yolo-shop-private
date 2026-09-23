# Clear-view baseline: installed OCR CAN read both milk brands frontally

No production, environment, model, threshold, or Jev-patch change. CPU
only, zero Jev calls. Images decoded EXIF-transposed (as viewed); raw
bytes/hashes in `manifest_inputs.json`. Normalization frozen upfront:
uppercase + whitespace-stripped exact match; strict brand roots
(ПРОСТОКВАШИНО, ДОМИК/ДЕРЕВНЕ) in correct script; partials scored
separately by correct-script prefix/root presence. No catalog matching.

## Selected images (max two per brand; rest marked unsuitable)

| File (3000×4000) | Visible text | Verdict |
|---|---|---|
| P-front-081946 | Frontal cat label, white-on-blue arc ПРОСТОКВАШИНО, МОЛОКО 2,5%, sharp, single bottle | **Suitable primary** |
| P-side-082007 | Side/back, curved band + back-label block, sharp | Suitable secondary |
| P-back-081955 | Back legal text only, no brand lettering | **Unsuitable (no brand)** |
| D-front-2288 | Frontal red script Домик в деревне + МОЛОКО 2,5%, sharp | **Suitable primary** |
| D-side-2289 | Green side panel, small vertical brand strip | Suitable secondary |
| D-back-2290 / D-side-2291 | Barcode/legal, brand slivers only | **Unsuitable** |

## Results (installed pipeline, defaults: box_thresh 0.6, rec_thresh 0.0)

Auto full-image (warmed, ~7.8–8.5 s per 12 MP image):
- P-side: **ПРОСТОКВАШИНО 0.929 exact** + back-label lines. Brand recovered automatically.
- P-front: 8 lines (МОЛОКО×2, 2,5% 0.979, fragments); **arc brand absent** — localization miss on the frontal curve.
- D-front: `Doмиk` 0.435 + `вдеревне` 0.548 (exact modulo space/case) + small print. Brand localized; first word mixed-script.
- D-side: same partials + dense small print.

Manual brand lines via installed crop op + direct rec (warmed ~33–63 ms):
- P-arc (corrected quad, verified sub shows full band): `ПРОСТОВ` 0.42 — correct-script prefix, full word fails on curvature.
- D-up: `Домик` 0.29 — **exact**; D-lo: `Ддере` partial. (First misplaced P quad caught the cat and read `""` — discarded as annotation error, preserved in log.)

## Comparison with video failures

Video 404/437 (curved brand, small/foreshortened): brand absent among
survivors. Clear P-side recovers it exactly; clear P-front still misses
it in auto but yields a correct-script prefix isolated. Video 571/455
(Domik total miss/empty): clear D-front localizes both script lines with
one exact word. Direction is consistent — resolution/framing/view convert
misses into localized (partly exact) text — but cameras, compression, and
lighting differ, so no single causal factor is claimed. Curvature alone
defeats full-word recognition (P-arc) and auto-localization (P-front)
even at 12 MP.

## Recommendation (one): repeatable laptop-browser capture/view experiment

Clear views succeed (P exact on side view; D localized with exact
second word and exact first word via oracle), so run the actual
laptop→browser→backend pipeline: present each milk bottle frontally at
~30–40 cm, rotate slowly to show a flatter label segment, hold 3 s per
pose under diffuse light, single bottle in frame. Record per-request
crops, sharpness, OCR lines, and Jev outcomes through the existing
diagnostics; success = strict brand root accumulated for both brands in
one session. If the frontal arc still never localizes through the real
pipeline, that isolates a localization experiment (curved-band
proposals) with a concrete failing input already in hand (`subs/P-arc.jpg`).
