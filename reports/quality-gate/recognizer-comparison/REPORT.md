# Recognizer comparison: installed vs alternative (OFFLINE, no evaluation run)

No production, dependency, model-selection, or Jev-patch change was made.
No alternative was acquired. CPU-only context; zero Jev calls.

## 1. Local inventory (no suitable alternative on disk)

- Installed/production: `cyrillic_PP-OCRv5_mobile_rec` (8.0 MB) +
  `PP-OCRv5_mobile_det` (4.8 MB) under `.cache/paddlex/official_models`;
  PaddlePaddle 3.3.1 / PaddleOCR-PaddleX 3.7.x in `.venv-ocr`.
- HF hub cache: only the two Paddle models above.
- Main `.venv`: torch/torchvision CPU (detector side), no OCR libraries.
- No tesseract binary, no EasyOCR/RapidOCR/TrOCR weights anywhere on disk
  (searched `site-packages`, caches, `/` for `*.onnx`, tesseract).

Conclusion: **no locally available compatible alternative recognizer
exists**. Per instructions no acquisition was performed; the proposal
below is reported before any download.

## 2. Proposed candidate (not acquired)

**EasyOCR, languages `ru` + `en`, CPU (`gpu=False`)**, run recognizer-only
via `Reader.recognize()` on the exact saved oracle subs
(`reports/quality-gate/recognizer-oracle/subs/*.jpg`, byte-identical).

- Why this one: EasyOCR documents Russian (Cyrillic) among its supported
  recognition languages; CPU execution via PyTorch is a documented mode.
  Latin-only models were excluded by this criterion.
- Requirements before any run (isolated scratch venv — never
  `.venv-ocr`/production): network to PyPI and the EasyOCR model hub;
  `pip install easyocr`; first-call download of the `ru` (+`en`)
  recognition weights; disk/RAM headroom for package + weights (exact
  sizes to be recorded at acquisition time — not stated here as verified
  fact); warmed per-line CPU latency to be measured on this machine.
- Native preprocessing differs (EasyOCR recognizer resizes/normalizes per
  its own contract); inputs stay byte-identical, preprocessing documented
  per model rather than forced equal.

## 3. Frozen scoring plan (fixed before any alternative output exists)

- Primary denominator (clearly readable only), grouped by physical brand
  region — variants count once: **G571** Domik script (571-M1/M2/B2/B3),
  **G404** curved ПРОСТОКВАШИНО (404-M1).
- Exploratory, outside primary denominator: 437-M1 (small/partial),
  455-M1 (rotated/small, highly uncertain).
- Positive controls, must stay correct (3 lines): 388-C1 СВЯТОЙ, 388-C2
  ИСТОЧНИК, 437-C AKBA.
- Negative controls (saved detector-recall rec inputs; any catalog-brand
  text here is a false-brand hit): 404 left-edge `60` box, 437:3:121
  cap-top empty box, 388 mirrored-Latin empty boxes, 463 neighbor `2,50`
  box, 580 top-edge `2,5%` box, 172 AKB/CKAA boxes.
- Strict brand recovery = visually verified target-root text
  (ДОМИК/ДЕРЕВНЕ, ПРОСТОКВАШИНО, СВЯТ/ИСТОЧ, AKBA in correct script);
  Latin-transliterated fragments (e.g. `DegepeB`, `OMUKBHE`) do **not**
  count. Report exact + normalized (uppercase/whitespace-stripped,
  difflib ratio) accuracy per line and per crop/brand, warmed ms and peak
  RSS per model.
- Advance to detector-plus-recognizer integration only if: BOTH milk
  brands recovered on ≥1 source crop each; 3/3 controls correct; zero new
  false-brand on negatives; CPU ms/RSS measured and practical. Passing
  would only qualify a broader held-out study, never a model swap.

## 4. 177:1:237 classification (preserved crop, erratum intact)

Crop `runs/v1-b2/crops/177_1_237.jpg` shows a frontal Saint Spring label;
raw record (track 1, frame 177, sharp full 12.44/norm 36.22) reads
`СВЯТОЙ` (0.65) + `ИСТОЧНИК` (0.75) plus `3hs`/`""`.
Classification: **correct target text for the visible bottle**
(OCR matches the crop's front label; mirrored Latin correctly empty).
Track-level attribution (track 1's earlier A/S history) stays uncertain
and is an ownership question, not an OCR error. The 172 erratum
(`ERRATUM_172_2_232.md`, disputed attribution) is untouched; nothing is
transferred without visual evidence.

## 5. Verdict: NO-GO (comparison impossible locally — failure reported)

No alternative met the criteria because **no alternative could be run**:
nothing on disk qualifies, and acquisition is explicitly out of scope for
this step. No further models or preprocessing experiments were added.
Lines still lacking automatic detector localization at production
settings (box_thresh 0.6): all milk brand lines — 571 script (only
partial boxes at 0.4), 404 curved band, 437 small curve, 455 rotated
label. The installed recognizer's oracle record stands: 0/7 milk oracle
inputs recovered vs 3/3 controls (`recognizer-oracle/REPORT.md`).
