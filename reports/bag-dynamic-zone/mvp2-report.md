# MVP round 2 report: hidden items, unnamed counts, bag-self boxes, categories, 1080p OCR

Server: http://127.0.0.1:8001/ (`LIGHTSTORE_DEVICE=cpu`), page unchanged in
design. Full suite: **200 backend tests OK** (2 pre-existing skips) +
**34 frontend tests OK**. Raw video never committed, nothing uploaded.

## 0. Invisible products — root cause (fixed)

Same-frame comparison (raw worker rows → ByteTrack tracks → page payload):
the Paddle **GPU worker returns zero rows on every frame** (driver 580 /
CUDA 13 vs Paddle 3.3.1), while the same checkpoint on CPU returns
`Bottle 0.12–0.63`. Bag pause never suppressed boxes (display path is
independent). Fix: CPU detector (~145 ms/frame) pinned in local `.env`.
Live: bottles display while the zone is `stable`. One cold-start transient
noted: the first product inference of a fresh session intermittently shows
0 tracks, then 2/2 stably on identical repeats — single frame, under watch,
not the reported seconds-long blindness. Residual layer (unchanged):
`new_track_thresh: 0.50` still gates weak views; transparent top-down
bottles score ≤ 0.19.

## 1. Hand-hidden items — "Packing..." (implemented + tested)

A reliably-outside track vanishing where it is **not clearly outside**
(rim band or interior) enters a `Packing…` watch (`PENDING_PACK_FRAMES=4`,
≈2 s): reappear-outside cancels, reappear-inside or full-wait expiry counts
once, disappearance while clearly outside starts no watch. ID-change guard:
a new interior ID near a pending same-class box merges into one count.
Page shows one amber `Packing…` label at the last position (image-embedded;
no new panels). Unit tests: complete/cancel/expiry/elsewhere/merge, plus a
behind-upright-bag test that **documents the honest limitation**: moving a
bottle behind the bag vanishes at the rim exactly like a hidden insertion
and counts after the wait — the top-down camera cannot distinguish the two.
Mitigations: 2 s window, cancel-on-reappear-outside, no count for
elsewhere-disappearances.

## 2. Counts without a catalog match (implemented + tested)

Packing already never required identification (counts keyed by detector
class — verified in code, no tracker change needed). Display now matches:
packed rows read `Unidentified bottle / Unidentified can /
Unidentified box / Apple` (generic map + fallback), and a session-scoped
label cache upgrades a row to the recognized name when OCR completes —
even after the track is pruned — without adding a count. Sidebar and
`Last packed` use the display names (`packed_display_counts`,
`last_event.display_name`); raw `packed_counts` stay for compatibility.
Test: pack → generic → complete OCR → renamed → pruned → name sticks,
total stays 1.

## 3. Bag-self false detections (diagnosed + suppressed)

Saved-frame diagnosis (screenshot photo region, CPU worker): `Storage box`
0.33–0.34 on a bag-sized box [286,256,598,615] coinciding with the locked
footprint (live it exceeded 0.5 → track #13 + Recognition thumbnail of the
bag); `Canned` similar transient. Patterned plastic mimics packaging —
no detector/threshold change. Fix: geometric `is_bag_self` filter
(footprint overlap ≥ 0.55 AND box-inside ≥ 0.50; live false box scored
≈0.9/≈0.95) applied pre-tracking whenever the zone is stable/in grace, so
the bag gets no track, no OCR crop, no count. Genuine items beside/inside
score ≈0 on at least one ratio (milk carton, cola bottle, green bottle
checked). Integration test: bag-sized box suppressed (`suppressed_bag_self`
diagnostic in the response), small-inside and beside boxes kept. White-bag
video re-check: no genuine suppression. Live blue-bag confirmation still
wanted — the still-image seg run can't reproduce the overlay-polluted frame.

## 4. Four MVP categories (verified, misses separated from naming)

All four classes are enabled in the default profile and pack
class-agnostically (existing per-class transfer test covers Apple with no
identifier involved): Bottle ✓ (0.61–0.69, tracked live), Canned ✓ (Gorilla
can 0.31–0.35 at distance; ≥0.5 close-up per live track #13), Storage box ✓
(milk carton 0.10–0.16 at distance; tracked close-up per live entries),
Apple ✓ (class + `Apple` display name + transfer path; **no real apple
images exist in the repo — needs a live check**). Naming (OCR/Jev/catalog)
is untouched by all of the above by construction.

## 5. Phone 1920×1080 for OCR (implemented, browser check pending)

`getUserMedia` now tries 1920×1080 16:9 (+facingMode) → 1920×1080 →
facingMode → plain, aborting loudly only on permission/nodevice errors.
Resolution-agnostic pipeline confirmed in code: retained full-res grab +
OCR crops scale automatically; detection upload stays 640×480 q0.75;
crop JPEG stays q0.85 (no evidence to raise it). New `Camera W×H` readout
in Details shows what Chrome actually selected — please confirm it reads
1920×1080 on the Samsung, plus crop sizes/OCR hit rate/speed live; that
half cannot be verified from here (no camera device in this environment).
Frontend tests: first-attempt ideals + fallback chain + readout (34 pass).

## Live verification on 8001 (observed, CPU build)

Zone locks by frame 3 (conf ~0.47); bottles display with zone stable;
`pending: []`, `suppressed_bag_self` counter live in responses. Automated:
191→200 backend tests, 34 frontend tests, video harness (genuine ×1 /
stationary ×0 at both cadences). Still needs the live camera: blue-bag
suppression confirm, apple pack, 1080p readout, behind-bag behavior.

## Remaining limitations

1. Gathered-bag following (pauses instead). 2. Weak-view products under
0.50 never initiate tracks. 3. Behind-bag vs insertion indistinguishable.
4. OCR service reports unavailable (pre-existing, untouched). 5. First-frame
cold transient (single frame, under watch).
