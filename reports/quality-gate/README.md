# Pre-OCR quality gate: evaluation and narrowly scoped improvement

Inference stayed on CPU throughout. No external Jev calls, no camera access,
no Jev acceptance/retry changes. OCR model, recognition settings and input
preprocessing were identical for every gate variant (PP-OCRv5 Cyrillic CPU via
`ocr_worker.py`); gates only decide accept/reject.

## Evaluation set

- `crops-manifest.json`: 329 saved native crops, 154 near-duplicate groups
  (dhash hamming<=6, union-find). Group-aware split: tuning 198 / validation 131.
- `ocr-representatives.json`: 156 group-deduped CPU OCR runs (best blurry +
  best accepted per group) with full + normalized (600/1000/1600) sharpness,
  text, recognition scores, crop-pixel polys/boxes, timings.
- `ocr-targeted.json`: 8 extra CPU OCRs — headlines 404:2:61 (readable P,
  generic-only) and 444:2:133 (accepted-empty, still empty) + 6 P blurry
  confirming the curved-brand pattern.
- `labels.json`: manual review subset with target text transcribed separately
  from neighboring text; 9 crops visually inspected, ambiguous cases marked.
  OCR failure alone never marks readability (e.g. 571:3:311 is human-readable
  but returns zero lines/polys — a detection failure).
- `sharpness-all.json`: full + normalized sharpness for all 329 crops.

## Offline comparison (tuning vs held-out validation)

Brand-hit = strict target roots (AQUA/AKBA; СЕНЕЖ*; СВЯТ/CBЯT/ИСТОЧ*;
ПРОСТОКВАШИНО; ДОМИК/ДЕРЕВНЕ/DOMIK). Lone AKB and DOM fragments do not count.

| split | gate | acc | target-hit | empty | neighbor-brand | ocr_ms | missed |
|---|---|---|---|---|---|---|---|
| tuning | current_full15 | 32 | 8 | 3 | 1 | 21084.0 | 15 |
| tuning | bypass | 76 | 23 | 10 | 1 | 62630.4 | 0 |
| tuning | norm1000_gte30 | 30 | 11 | 0 | 1 | 33788.2 | 12 |
| tuning | union_full15_or_n1000g30 | 46 | 15 | 3 | 1 | 39532.5 | 8 |
| validation | current_full15 | 30 | 5 | 6 | 1 | 17103.6 | 14 |
| validation | bypass | 79 | 19 | 12 | 2 | 62386.2 | 0 |
| validation | norm1000_gte30 | 23 | 6 | 0 | 1 | 27851.0 | 13 |
| validation | union_full15_or_n1000g30 | 46 | 9 | 6 | 1 | 37722.0 | 10 |

Per-product: all recovered hits are Aqua/Senezhskaya/Saint Spring waters.
Prostokvashino brand = 0 everywhere (6/6 extra P blurry → МОЛОКО/2,5% only);
Domik brand ≈ 0. Bypass admits +7/+6 empty OCRs and an extra neighbor-brand
case on validation; the union adds zero extra empties and zero extra
neighbor-brand on either split.

Full-329 gate-level admits (upper bound; downstream dedup/throttle suppress
duplicates, so real extra OCR ≈ reps-level +44%/+53%):
- tuning: current 56 → union 125 (+69, 2.23x)
- validation: current 49 → union 88 (+39, 1.80x)

## Before/after (exact crops, OCR text boxes in crop pixels)

Green polys/boxes map 1:1 to the submitted crop bytes (worker predicts on the
decoded upload; scores are recognition scores).

- Rescued readable reject — 388:2:31 Saint Spring (full 12.37 → blurry before;
  norm@1000 34.88 → accepted after; OCR СВЯТОЙ/ИСТОЧНИК with polys):

![388:2:31 rescued Saint Spring with OCR boxes](/home/olesya/Projects/yolo-shop/reports/quality-gate/examples/388_2_31-ocr-boxes.jpg)

- Preserved small-crop success — 437:4:122 Aqua (full 16.19 accepted before and
  after; norm@1000 18.9 alone would have dropped this AKBA hit, which is why a
  pure-normalized gate was rejected in favor of the union):

![437:4:122 small Aqua success preserved](/home/olesya/Projects/yolo-shop/reports/quality-gate/examples/437_4_122-ocr-boxes.jpg)

- Accepted-empty detection failure — 571:3:311 Domik (full 31.60 accepted;
  zero lines/polys before and after; human-readable label, needs
  detection-side work, out of scope here):

![571:3:311 accepted Domik with zero OCR lines](/home/olesya/Projects/yolo-shop/reports/quality-gate/examples/571_3_311-ocr-boxes.jpg)

- Neighbor contamination — 463:7:160 Saint Spring crop containing the adjacent
  Domik label (OCR reads neighbor `2,50` only; geometry problem, not fixable
  by threshold):

![463:7:160 neighbor contamination](/home/olesya/Projects/yolo-shop/reports/quality-gate/examples/463_7_160-ocr-boxes.jpg)

- Neighbor coexistence — 580:3:329 Senezhskaya with P's 2,5% strip (target +
  neighbor both recovered; shared 2,5% alone cannot distinguish P/D):

![580:3:329 target plus neighbor text](/home/olesya/Projects/yolo-shop/reports/quality-gate/examples/580_3_329-ocr-boxes.jpg)

- Curved-brand recognition limit — 404:2:61 Prostokvashino (readable reject;
  re-run OCR gives МОЛОКО/2,5% only, brand missed):

![404:2:61 generic-only milk OCR](/home/olesya/Projects/yolo-shop/reports/quality-gate/examples/404_2_61-ocr-boxes.jpg)

## Implemented gate (narrowly scoped)

`quality_gate_accepts()` in `vision.py`: accept when
`full >= OCR_SHARPNESS_MIN (15.0)` **OR** `norm@1000 >= OCR_SHARPNESS_NORM_MIN (30.0)`
(`config.py`). Live code reproduces offline numbers exactly
(388:2:31 → 12.37/34.88 accept; 361:2:4 → 15.64/49.27 accept;
398:6:49 → 4.07/4.74 reject). Lifecycle, scheduling, duplicate suppression,
payload bounds and CPU config are untouched; diagnostics now also record both
sharpness values. Rationale: union preserves every current accept (no small-crop
regression, e.g. 437:4:122) while rescuing large readable labels the
full-resolution metric undervalues; held-out validation confirms +4 hits
including the headline case with no extra empties/neighbor-brand.

## CPU/workload tradeoff and remaining failures

- Cost: reps-level +14/+16 accepts (+44%/+53%), +18s/+21s OCR CPU on the eval
  pool; full-329 upper bound 2.23x/1.80x gate admits (duplicates partly
  suppressed downstream by dhash dedup + submit intervals + per-track slots).
  No Jev behavior change (Jev fires only on useful evidence).
- Still rejected but readable: 8 tuning + 10 validation brand hits below both
  thresholds (very blurry band) — rescuing them needs far lower thresholds at
  much higher empty-OCR cost; not pursued.
- Prostokvashino/Domik brands still unrecovered (curved/stylized
  recognition + accepted-empty detection failures). Needs (separately):
  label-region/perspective handling, neighbor exclusion, detector-side text
  detection work — explicitly out of scope here, no new OCR model added.
- No SKU-identification claim is made from OCR text alone.

## Changed files and tests

- `vision.py`: `crop_sharpness_at_height()`, `quality_gate_accepts()`, dual
  gate + both sharpness values in diagnostics.
- `config.py`: `OCR_SHARPNESS_NORM_HEIGHT=1000`, `OCR_SHARPNESS_NORM_MIN=30.0`.
- `tests/test_quality_gate.py`: 9 focused tests (config bounds, determinism,
  monotonic-with-blur, size variants, malformed/empty fail-closed with NaN/None,
  headline rescue, blurry reject, small-sharp preserve, no-regression subset).
- Results: new 9/9 pass; `test_crop_protocol` 28/28; `test_identification`
  25/25; `test_app` 14 (2 skipped, pre-existing).

## Replay verification (short)

1. From repo root with `.venv`: `python -m unittest discover -s tests -p test_quality_gate.py`
2. `python reports/quality-gate/evaluate_gates.py` — regenerates
   `eval-results.json` / `eval-table.md` from saved crops + saved CPU OCR
   (no camera, no Jev calls; re-running OCR optional via `ocr-targeted.py`
   pattern on CPU).
3. Spot-check gate on exact crops: compute `crop_sharpness` /
   `crop_sharpness_at_height(...,1000)` for
   `evidence/v1-ocr-cold/388_2_31-crop_jpeg.jpg` → expect ≈12.37/≈34.88 accept;
   `.../398_6_49-crop_jpeg.jpg` → reject; `.../361_2_4-crop_jpeg.jpg` → accept.
