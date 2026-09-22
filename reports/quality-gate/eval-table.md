# Quality-gate evaluation (tuning vs held-out validation)

Pool: 156 group-deduped reps + 2 headlines (404:2:61 P readable,
444:2:133 accepted-empty) = 158 OCR'd crops. +6 extra P blurry confirm
generic-only pattern (reported separately). OCR model/settings/
preprocessing identical for all gates (CPU PP-OCRv5, ocr_worker.py).
Brand-hit = strict target roots (AQUA/AKBA; СЕНЕЖ*; СВЯТ/CBЯT/ИСТОЧ*;
ПРОСТОКВАШИНО; ДОМИК/ДЕРЕВНЕ/DOMIK). Lone AKB and DOM fragments
do not count. Splits are group-aware (dhash hamming<=6).

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

Extra P confirmation: 6/6 additional
P blurry → МОЛОКО/2,5% only, 0 brand (curved-logo recognition limit).

Full-329 gate-level admits (sharpness only, incl. duplicates
that downstream dedup/throttle partly suppress):
- tuning: n=198 current=56 union=125 extra=69 (2.23x)
- validation: n=131 current=49 union=88 extra=39 (1.80x)
