# Direct-recognizer oracle (offline, CPU, no production change)

Question: can the installed Cyrillic recognizer read the missing milk
brands from correctly isolated lines?

Method: manual brand-line quads (pixel coords + GT text in
`results_compact.json`; uncertain portions labeled), extraction with the
installed `CropByPolys quad` crop op (min-area-rect perspective warp,
INTER_CUBIC, BORDER_REPLICATE), direct `text_rec_model` call bypassing
detection, unchanged settings. One predefined alternative per line:
axis-aligned bbox +4px. Scores are model confidences, not correctness.
Oracle isolation is manual and unavailable automatically in production.

## Per-line evidence (warmed rec-only ms ~32–62)

| Line | GT (uncertainty) | Primary → text / score | Alt → text / score | Verdict |
|---|---|---|---|---|
| 571-M1 upper script | ДОМИК (ligature?) | `H` / 0.29 | `""` / 0.0 | Fail; geometry-independent (B-boxes also fail below) |
| 571-M2 lower script | В ДЕРЕВНЕ (partial?) | `DegepeB` / 0.39 (Latin misread) | `Degerea` / 0.46 | Fail; script transliterated, strict roots absent |
| 571-B2 (exact B box) | ДОМИК (B geometry) | `OMUKBHE` / 0.33 | `ОЛaН` / 0.20 | Fail; same as manual → recognition, not geometry |
| 571-B3 (exact B box) | В ДЕРЕВНЕ (B geometry) | `Bgepe` / 0.82 | `Bgpe` / 0.40 | Fail despite 0.82 confidence — confidence ≠ correctness |
| 404-M1 curved | ПРОСТОКВАШИНО (effort; quad approx) | `ПОСТО` / 0.23 (5-char prefix frag) | `П` / 0.12 | Partial only; full word fails even isolated |
| 437-M1 small/partial | ПРОСТОКВАШИНО (uncertain) | `""` / 0.0 | `""` / 0.0 | Fail |
| 455-M1 rotated | ДОМИК В ДЕРЕВНЕ (highly uncertain) | `""` / 0.0 | `""` / 0.0 | Fail |
| 388-C1 ctrl | СВЯТОЙ | `СВЯТОЙ` / 0.70 | degraded | Pass (primary) |
| 388-C2 ctrl | ИСТОЧНИК | `ИСТОЧНИК` / 0.73 | degraded | Pass |
| 437-C ctrl | AKBA | `AKBA` / 0.98 | `AKBA` / 0.98 | Pass |

Strict brand recovery: controls 3/3 primary; milk 0/7 oracle inputs
(norm ratios: 404 prefix 0.556, rest ≤0.33 or 0). High-confidence failure
(571-B3 0.82) and empty failures coexist; alt extraction never rescues a
brand (controls degrade under alt, confirming primary warp is the right
comparison).

## Conclusion

Accurate manual isolation consistently fails on milk brands while the
same harness recovers flat control lines: the dominant limitation is the
**recognizer on stylized script / curved / small-rotated Cyrillic**, not
localization geometry (571 B-geometry == manual failure), not downstream
filtering (worker thresh 0.0; empties only), not extraction (controls
pass through identical code).

## Recommended next experiment (exactly one)

**Recognizer comparison** (offline): run the saved oracle subs
(`subs/*.jpg`, 10 primary + 10 alt) through one alternative Cyrillic
recognition configuration alongside the installed one, identical inputs.
Acceptance: strict recovery on ≥3/7 milk oracle lines with 3/3 controls
retained and per-line ms reported; else insufficient. No production,
threshold, perspective, or model swap until that gate passes. Localization
work is not recommended (isolation already correct and still fails);
further collection is unnecessary for this question.
