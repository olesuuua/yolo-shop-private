# Ten-second refresh verification, 2026-09-24

Backend: 208 tests passed, 2 skipped. Scripted mask tests cover the timed
freeze, one count across refresh, a bottle-width notch, a partial mask from
the recorded camera view, two-second retry/loss, and credible/implausible
moves. The paced real server replay (recorded bag frame plus bottle sprite)
passed: clear insertion 1 count, hand-hidden insertion 1, beside bag 0,
behind-bag occlusion 1 (known false positive). The clear insertion passed
through `grace` and `lost` after an occluded refresh but retained its count.
These are replay results, not a new live camera observation.

On port 8001 with the real CPU models and one recorded bag frame at 0.8 s
request spacing, the first three frames acquired the bag; frame 13 refreshed
it. Product detection after warmup was about 154–158 ms per frame. Bag work
was about 0.2 ms on locked frames versus 28.6–29.9 ms on inference frames.
Full request time was about 159–163 ms locked versus 190–192 ms at acquisition
or refresh, excluding the first model warmup frame (660 ms). Segmentation
savings are real but modest because product detection dominates. The server
and webpage are left on port 8001 for the requested live camera check.
