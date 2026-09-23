# Detector-recall A/B (offline, CPU, no Jev/production change)

Correction: `dt_polys` is detector output with invalid/empty recognition
crops removed — not post-recognition survivors. Installed
`paddlex/inference/pipelines/ocr/pipeline.py`: detector output assigned at
`dt_polys_list = [item["dt_polys"] ...]` (L370) and `results[].dt_polys`
(L379); invalid/empty crops filtered at L415–426
(`sub_img.size>0 and shape>0`, `results[idx]["dt_polys"]=filtered_polys`);
recognition survivors appended only if
`rec_res["rec_score"] >= text_rec_score_thresh` (L480) with
`rec_polys.append(dt_polys[sno])` (L494). Effective `text_rec_score_thresh`
is 0.0 here, so all recognized lines survive; `dt_scores` key does not
exist (unavailable). Pre-invalid-removal raw counts were captured via a
runtime det-model wrapper: invalid removals were 0 in all 36 runs.

Conditions (all else identical): A `box_thresh=0.6` (default), B
`box_thresh=0.4` override via `predict(text_det_box_thresh=...)`.
Detector input = crop pixels (doc preprocessor off); `limit_type=min`,
`limit_side_len=64`, `thresh=0.3`, `unclip_ratio=1.5` identical every run
(recorded per-run in `results_compact.json`). Warmed A–B–B–A per crop over
identical bytes; A0==A3 and B1==B2 on all 9 crops (deterministic).

## Per-crop (visual verification required for milk recovery)

| Crop | A dt/rec | B dt/rec | Target localized / recognized | Wrong/neighbor/generic | Stage |
|---|---|---|---|---|---|
| 571:3:311 milk-D | 0/[] | 3/`2,OMUKBHE,Bgepe` | B localizes brand lines (polys on script, annot) but misreads; strict roots absent → **not correct recovery** | no neighbor present; `2` drops downstream |0618det-miss at 0.6; partial localization at 0.4, recognition miss |
| 404:2:61 milk-P | 3/generic only | 3/identical | brand never proposed either way | generic correct | Curved-brand detection miss |
| 437:3:121 milk-P | 3/generic+empty | 3/identical | brand absent | empty correctly dropped downstream | Same as 404 |
| 455:3:148 milk-D | 0/[] | 0/[] | absent both | empty | Rotated/small detection miss |
| 388:2:31 ctrl-W | 4/СВЯТОЙ,ИСТОЧНИК | 5/+НЕГАЗИРОВАННАЯ | retained both; extra is correct target-generic | no wrong/neighbor | Pass with reported addition |
| 437:4:122 ctrl-A | 3/AKBA | 3/identical | retained, no new outputs | `3` correctly dropped | Pass |
| 463:7:160 neg-W | 1/`2,50` | 4/+UCTA,`-`,KBHO | B adds target-region UCTA misread + neighbor KBHO frags | **additional neighbor outputs** | Reported regression |
| 580:3:329 neg-S | 2/both | 2/identical | unchanged coexist | unchanged | No change |
| 172:2:232 neg-W2 | 3/AKB,CKAA | 3/identical | no W either way | no new outputs | Historical ghost not reproduced (below) |

Milk full recovery: **0/4** both conditions. Mean polys A 2.11 → B 2.89
(1.37×, <2× pass); warmed mean ms A 794 → B 878 (1.11×, <1.5× pass).
Invalid-crop removals 0/36. Recognition inputs saved to
`/tmp/opencode/ab_raw/subs` (ignored); compact results + harness +
annot overlays live here.

## 172:2:232 provenance

Our input is `reports/quality-gate/replay-comparison/runs/v1-b2/crops/172_2_232.jpg`
(875×1716; sha in `results_compact.json`); rerun output `AKB/CKAA`, no W,
identical A/B. Same-run `browser-events.jsonl` diagnostics for that request
also show `AKB/CKAA`, no W — while the run summary attributes ghost W text
to it. Same PP-OCRv5 worker pipeline per reports. Cause of the
summary-level misattribution (track-level accumulation vs run variance)
**remains unexplained**; the historical ghost output is retained as a
regression example, not as single-crop ground truth.

## Verdict: NO-GO for broader evaluation

Fails the provisional gate on the primary prong (0/4 < 2/4 milk
recoveries) despite passing cost gates, with two reported regressions
(463 extra neighbor fragments; 388 extra generic line, benign). B
additionally converts 571 from empty to target-localized-but-misread —
localization signal only. Qualifying for broader held-out study is
**not recommended** on brand-recovery grounds.

Note: the 571 tight-label oracle (`/tmp/opencode/571_oracle_tight.jpg`,
manual box, **non-production**) found generic only (`2,4`,`OKO`), brand
still missed — consistent with recognition-hard script, not a downstream
drop. No quarantine/heuristics/perspective/model changes were made.
