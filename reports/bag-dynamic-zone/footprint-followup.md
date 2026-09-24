# Persistent bag footprint follow-up

The fixed-camera MVP now keeps its acquired working footprint independently
from each observed segmentation mask. Internal pixel changes can trigger a
model check but do not redraw the packing boundary. Missing rim pixels retain
the previous edge. Outward additions need three similar observations, at least
1% of the existing grid area, and no more than 20% in one observation. A
whole-bag shift needs three consistent observations before packing pauses and
the new location is acquired. The two-second loss grace remains in place.

Synthetic mask and tracker checks (`tests/test_bag_zone.py`):

| Case | Drawn outline and packing result |
| --- | --- |
| Bottle crossing and covering rim | Previous outline retained; one outside-to-inside insertion counted once. |
| Bottle remaining inside | Interior mask gap did not change outline or add a count. |
| Hand covering rim | Previous outline retained; no extra count. |
| Real outward bulge | Outline expanded only after three matching masks; shape change alone counted nothing. |
| One oversized mask | Ignored; outline unchanged. |
| Deliberate whole-bag move | Two observations kept old outline; third paused packing, then repeated observations relocked without counting a stationary product. |

These checks use deterministic masks, not new camera footage. With one fixed
camera, a persistent object or hand covering every edge used to infer a move
can be indistinguishable from a real bag move. A sustained false outward
segmentation bulge can also be indistinguishable from a physical bulge.
Occlusion that lasts beyond the two-second loss grace pauses packing if the
bag cannot be localized at all.
