# EasyOCR cyrillic_g2 vs installed recognizer (offline head-to-head)

Scratch-only acquisition (`/tmp/easyocr-scratch`, never production envs):
easyocr 1.7.2 + torch 2.7.1+cpu / torchvision 0.22.1+cpu / numpy 2.4.6 /
opencv-headless 5.0.0.93 / pillow 12.3.0. Single asset downloaded —
`cyrillic_g2.pth` (15,256,523 B,
sha256 `48d0f3b5…07529f29`) from
`https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.1/cyrillic_g2.zip`
— via `Reader(['ru','en'], gpu=False, detector=False, recognizer=True)`
with a scratch model dir. Cyrillic capability confirmed
(`model_lang=cyrillic`, 207-char set incl. full Russian alphabet).
Measured evaluation ran with `download_enabled=False`.
No Jev calls; production code/deps/model selection untouched.

Identical inputs, direct recognition (detection bypassed), one fixed
config each with native preprocessing (Paddle: BGR sub as cropped;
EasyOCR: same pixels BGR→RGB per its contract). EasyOCR options:
`decoder=greedy`, `contrast_ths=0.1`, `adjust_contrast=0.5`,
`quantize=True`, `workers=0`, `allowlist=None`. OMP/torch threads 2.
Warmed repeats (dummy rec + 2 measured passes; second reported).
Confidence values are uncalibrated per-model scores, never compared.

## Grouped results (strict = correct-script brand roots)

| Group | Installed (Paddle) | EasyOCR (cyrillic_g2) |
|---|---|---|
| G571 Domik script (M1/M2/B2/B3) | partial Latin misreads (`DegepeB`, `OMUKBH`, `Bgepe`, `H`) — localized, not recovered | garbage (`'`, `28 гу`, `хомоген`, `8gce`) — not recovered |
| G404 curved ПРОСТОКВАШИНО | `ПОСТО` partial (norm ratio 0.556) — not recovered | `сюiвк уу` garbage — not recovered |
| 437-M1 / 455-M1 (exploratory) | `""` both | `(42.3` / `""` — not recovered |
| Controls 388-C1/C2, 437-C | **3/3 exact** (0.67/0.74/0.98) | **0/3**: `СВЯтOU`, `UСТОчHUK`, homoglyph `АквА` (visually near-Latin but wrong script → partial by frozen rule) |
| Negatives (7) | correct neighbor/generic or empty (`60`, `2,50`, `2,5%`, AKB/CKAA, empties) — no brand hallucination | garbage fragments only — no brand hallucination either |

Exact/normalized detail per line in `results_compare.json` (27 inputs).
Milk strict recovery: installed 0/2 groups, EasyOCR 0/2 groups.
Warmed latency: installed ~62 ms/line; EasyOCR ~6–26 ms/line (faster but
wrong). Process RSS peaks: Paddle proc 443 MB (base 59), EasyOCR proc
391 MB (base 340, torch preloaded) — separate processes, same order of
magnitude; practicality moot given accuracy.

## Verdict: NO-GO

Fails at every advance prong: neither milk brand recovered by the
alternative, 0/3 controls correct (vs 3/3 installed), so integration is
not authorized. No further models or preprocessing experiments added.
Lines still lacking automatic localization at production detector
settings: all milk brand lines (571 script partial boxes only at
box_thresh 0.4; 404 curved, 437 small curve, 455 rotated: none).
The 172 erratum and the 177 correct-target-text classification stand
unchanged.
