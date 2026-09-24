# Bag footprint follow-up: recorded camera sequence

Source: user screen recording `Screencast From 2026-09-24 12-00-18.webm`
(kept out of git). It contains the real camera view, the drawn outline and
the on-screen packed counter. Frame numbers below are decoded video frames.

## Failure seen in the recording

| Frame | Visible result in the original run |
| --- | --- |
| 90–120 | Empty bag has a full, stable green outline; packed 0. |
| 125–145 | Cola and hand cross the rim. The outline becomes gray during reacquisition; packed 0. |
| 300–444 | Bag lost; packed 0. The cola remains in front of the bag. |
| 456 | Bag moving; packed 0. |
| 468–540 | Bag stable again, but its green outline has an inward notch around the cola. Packed stays 0. |
| 578 | Bag lost again; packed 0. |

The destructive transition was the old `moving`/`lost` reacquisition path:
`_try_lock()` could adopt a partial segmentation polygon. The current tracker
holds the acquired outline for ten seconds, then replaces it only after a
whole-shape check. Failed refreshes retry during a two-second grace period;
partial masks cannot relock the bag after loss.

## Checks from actual camera-view pixels

The `tests/fixtures/bag_live_20260924_masks.npz` fixture contains three bag
model masks extracted from the recording's camera region: frame 90 (full),
240 (small displacement) and 360 (partial view). It contains masks only, not
screen imagery. The full mask covers **1,886 grid cells**, while the partial
observation covers only **755**. The revised tracker rejects the partial mask
at refresh and keeps the prior outline through grace. Repeated partial
observations after loss leave the outline absent and status `lost`.

The screen recording's packed counter stays at **0**. It shows the cola
crossing the rim, then outside/in front of the bag, followed by bag movement;
it does not show a settled bottle inside the bag with stable outside evidence.
The recorded run therefore verifies **no count from bag movement**, but does
not establish the requested one-count insertion result. A post-fix live run
with the bottle held clearly outside and then placed inside is still needed.

Screen recording pixels include the old annotation overlay, which affects
model confidence when reprocessed. The mask fixture and outline checks above
use real recorded camera-view pixels, while the post-fix live run will verify
the complete camera-to-counter path without that replay artifact.
