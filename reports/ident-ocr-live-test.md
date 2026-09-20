# Live OCR identification test — work report

Scope: connect the existing standalone OCR (`ocr_local.py`) and Jev
(`jev_catalog.py`) utilities to the live webcam pipeline for product
identification only. Detector (PP-YOLOE Objects365 on CPU), OCR model
(PP-OCRv5 Cyrillic) and Jev model unchanged throughout. Order matching and
bag-counting untouched.

## 1. Two-bottle catalog and live pipeline

- Located reference photos + store-description screenshots under
  `data/local/products/bottle-1` (Aqua Minerale) and `bottle-2` (Сенежская).
- Created `data/local/catalog.json` in the `jev_catalog.py` format (2 SKUs:
  `aqua-minerale-0-5l`, `senezhskaya-0-5l`). Store descriptions preserved
  verbatim; `verified_label_text` holds only clearly legible packaging words;
  unknowns omitted. Bottle-2 contradictions flagged in `cautions` (shelf life
  12 vs 18 мес., storage +25 °C vs +30 °C, mineral-water wording).
- New `identification.py`: per-track OCR accumulation keyed strictly by
  ByteTrack ID, one background worker owning CPU OCR (persistent `.venv-ocr`
  subprocess, `ocr_worker.py`) plus Jev requests; detector labels passed only
  as unreliable hints; uncertain/unknown outcomes stay unresolved;
  `exact_sku_verified` never set; key stays server-side (`/api/ident-readiness`
  reports presence only).
- `vision.py`: uploads stay high-res (≤1280 wide) for OCR crops while
  detection stays 640×480; sharp crops per track; preview never blocks on
  OCR/Jev; per-box status overlay. `app.py`: readiness endpoint. Page:
  1280×960 capture, per-track identification panel.

## 2. CPU OCR fix

`ocr_local.py`/`ocr_worker.py` failed on CPU with a oneDNN
`NotImplementedError` in PP-OCRv5 detection. Fixed with the documented
PaddleX flag `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False` for CPU. Validated on
all 6 reference photos: Senezhskaya front reads cleanly; Aqua front reads
(`АКВА` often OCRs as Latin `AKBA` — recorded in catalog clues); rear small
print is mostly noise but brand/volume/EAC tokens survive. Evidence:
`outputs/ident-validation-bottles.json`.

## 3. Sticky-identity fix

Symptom (camera test): after removing Aqua, the next bottle inherited its
identity — ByteTrack reuses IDs, and evidence lived 30 s. Fix: a track absent
more than `IDENT_ABSENT_FRAMES = 3` consecutive processed frames loses all
evidence/identity; reappearing IDs restart at "need evidence". Unit-tested.

## 4. Overlay readability

Replaced outline-shadow labels with solid pill backgrounds, larger type,
stacked above each box (class on top, `ID:` below). Verified on a webcam shot
before deploying.

## 5. Two milk products

Added `milk-1` (Простоквашино) and `milk-2` (Домик в деревне) to the catalog
(4 SKUs; no packaging photos supplied, so `verified_label_text` is minimal
and the near-pair warning is in `cautions`). Live Jev probes with the project
key (3 calls): brand visible → correct milk at confidence 1.00 both ways;
brand absent (only shared `МОЛОКО/2,5%/930мл`) → `insufficient_evidence`,
no guessing between the twins.

## 6. Camera-to-OCR responsiveness investigation

Measured on CPU before changing anything: detection 0.16 s/frame; OCR
0.12–0.9 s by crop size; downscaling crops to 170 px destroys OCR (kept
native); sharpness calibration showed the gate of 60 rejected perfectly
readable crops (54.8 reads flawlessly, text survives to ~7).
Changed: gate 60 → 15; first retry 2.0 s → 0.75 s until evidence flows;
shared FIFO → per-track newest-crop slots with round-robin fairness; dHash
dedup of still frames; Jev moved to its own thread (OCR continues
mid-request). Diagnostics: `/api/ident-debug` (upload size, fps, detection
ms, per-track crop dims/sharpness/timings/recognized text) and
`/api/ident-crop/<id>` thumbnails shown in the sidebar.
Before/after harness (identical scripted load, 0.9 s OCR / 0.8 s Jev):
second bottle first OCR 2.6 → 1.8 s, identification 3.4 → 2.6 s; redundant
OCR bounded. Evidence: `outputs/ident-pipeline-measurements.json`.

## 7. Простоквашино catalog refresh

Updated `prostokvashino-2-5-930ml` from the revised
`data/local/products/milk-1/catalog-reference.md`: added
`Всегда свежее и 100% натуральное` to packaging text and identification
clues, with the Markdown reference recorded as a source. The caution notes
that this is user-supplied packaging text, not independently verified from
a packaging photo; generic freshness/naturalness wording alone does not
confirm an exact SKU. The other three catalog entries were preserved.

Validated catalog loading and Jev request construction for all four products
without an API call. Restarted the existing server on CPU and verified
`/api/session` and `/api/ident-readiness`: Objects365 on CPU, catalog healthy,
all four SKUs loaded, and API key present. OCR initializes on first demand.

The catalog and supplied product references are included in this commit
explicitly despite the `data/local/` ignore rule. Credentials remain excluded.

## Validation and state

- 89 unit tests pass (`discover -s tests`, 2 pre-existing optional skips),
  including scheduling, isolation, no-transfer, dedup, and endpoint tests.
- Pre-commit verification repeated the Python suite successfully. All six
  frontend controller tests also pass after updating their browser mocks for
  the camera-start timeout, readiness polling timer, and readiness endpoint.
- `run_local.py --check` passes on CPU; server running on :8001 with the
  4-product catalog (key present, never exposed).
- Open camera observations: Senezhskaya identifies reliably; Aqua sometimes
  accumulates no evidence (diagnostics added, root cause still under test);
  Prostokvashino less reliable than Domik (contrast/size hypothesis, needs
  the crop-thumbnail comparison from live testing).

## Files changed/added

Changed: `app.py`, `config.py`, `ocr_local.py`, `static/app.js`,
`static/index.html`, `tests/test_app.py`, `vision.py`
(`run_local.py` modification predates this work). Added:
`identification.py`, `ocr_worker.py`, `tests/test_identification.py`,
`data/local/catalog.json`, `.env` (key placeholder, git-ignored),
`outputs/ident-validation-bottles.json`,
`outputs/ident-pipeline-measurements.json`.
