# Ten-second bag outline refresh

The dynamic bag zone acquires three consistent masks, then freezes the drawn
and packing outline for ten seconds of wall time. Motion inside the bag region
sets `possible_bag_move_during_lock` in the bag diagnostics; it does not move
the outline. This warning can also be triggered by a product or hand.

At the deadline, segmentation runs again. A replacement must have comparable
area and perimeter, retain nearly all of the prior footprint after at most
32 px of translation, and avoid a large new fringe. This rejects partial
masks and a bottle-width inward notch. A larger jump is rejected as
implausible. Accepted geometry changes invalidate incomplete outside-to-inside
product evidence; already packed IDs and counts remain.

A failed refresh keeps the last outline for two seconds and retries on each
processed frame. If no credible whole mask appears, the zone becomes `lost`,
removes the outline, and pauses packing. A partial mask cannot relock it.

The fixed camera cannot know exactly where a bag moved during the ten-second
lock. The motion warning exposes that limitation; the next credible refresh
can relocate it by up to 32 px. Larger moves require a new session or a later
more capable location check.
