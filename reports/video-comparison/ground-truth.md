# Physical-bottle ground truth

Identity comes from `data/local/videos/video_descr.txt`, continuous physical motion, and readable labels; tracker IDs are not ground truth. Size/variant expectations below come from the project's current catalog and are not independently confirmed on every recorded label. No tape measure or camera calibration was supplied: camera height/angle are qualitative.

| Physical ID | Supplied identity | Expected catalog size/variant |
|---|---|---|
| A | Aqua Minerale | 0.5 L, still water |
| S | Сенежская (description spells “snezhskaya”) | 0.5 L, still water |
| W | Святой источник / Saint Spring | 0.75 L, still water |
| P | Простоквашино | 930 ml, pasteurized milk, 2.5% |
| D | Домик в деревне | 930 ml, pasteurized milk, 2.5% |

## V1: 20260921_205640.mp4 (48.09 s)

Appearance windows are intervals between inspected source frames, not sub-frame annotations.

- A enters by 1 s; front brand is clearly presented by 2 s. Held centrally through approximately 5 s, then placed on the right by 8.7 s.
- S enters between 8.7 and 10 s; front brand is readable at 10 s. Held left of A through approximately 14 s, then placed in front-left of A by 17.5 s.
- W enters between 17.5 and 19 s; front brand is readable at 19 s. Held left through 22 s, then placed back-left by 26.2 s.
- P enters between 26.2 and 28 s. Front label presented 28–32 s, then placed back-middle by 35 s. Large generic milk/fat text is visible; inspect native crop before declaring a readable brand opportunity.
- D enters between 35 and 37 s. Rear green label faces camera through the held interval ~37–44 s. By the end D sits front-left of P and partially occludes W. A rear slogan is not independent brand/size proof.
- Lighting changes blue/magenta/neutral during the entire recording. Camera moves, physical bottle scale and orientation change, and hands cover caps/necks. Background floor/table edges can contaminate enlarged crops.

## V2: 20260921_205735.mp4 (11.17 s)

All five are continuously present in the same arrangement: W back-left, D in front of W and left of P, P back-middle, S front-right, A behind/right of S. Camera rises over roughly the first 2–3 s, then remains in a higher oblique view with small hand motion. Bottles separate in the image but labels become smaller and more foreshortened. D's rear label remains visible, not its front brand. Colored illumination continues to change; angle is not the only changing variable.

## V3: 20260921_205754.mp4 (12.95 s)

- 0–1 s: same physical arrangement as V2.
- ~1.2–3.5 s: hand rotates D until its front brand faces camera. A new readable-brand opportunity appears by ~3.5 s; inspect native frames/crops for the earlier boundary.
- ~4.7–6 s: D moves left, partly beyond the left image edge; W moves right after contact. D/W overlap/hand occlusion makes momentary box ownership ambiguous.
- ~7–9.5 s: W is moved into the gap beside P; D returns to the left foreground. Keep the W/P physical identities through motion independently of detector assignments.
- ~10.5–12 s: S moves left toward the foreground. A remains on the right. By the final frame: D far-left/front, W behind it left of P, P center-back, S center-front, A right.

Contact sheets: `frames/*-contact.jpg`, `frames/placement-timeline.jpg`. These are viewing aids, not OCR input images.
