# Clean installation check — 2026-09-17

## Procedure

Source files were copied into a new isolated directory, excluding the developer's
virtual environments, PaddleDetection checkout, weights, datasets and training runs.
`python3.14 setup_lightstore.py --no-cache` created fresh Python environments and
downloaded packages, pinned upstream source and the released six-class weights.
No model training, private files or credentials were required.

Host: Apple Silicon Mac, macOS 15.5. Main Python: 3.14.2. Worker Python: 3.11.15,
Paddle 3.3.1, CPU, detector input 416×416.

## Issue found and fixed

The fresh worker environment initially had setuptools 82.0.1, which does not
provide `pkg_resources`. The pinned PaddleDetection model-zoo module still imports
it. The developer environment had setuptools 80.10.2, hiding this missing pin.
`requirements-paddle.txt` now explicitly requires the working 80.10.2 version.
Rerunning setup in the isolated copy installed that pin and passed real inference.
The main environment's previously unpinned transitive dependencies are pinned too.

## Results

- Both environments pass `pip check`; installed runtime dependencies match pins.
- Source commit: `b25522a0f4bde8c80603f3ba5e3472059972e3b5`, without local patches.
- Weights SHA-256: `a7d4f66ac449016bcc006d3ff46e64f46d6d05d3f8ad5364245e395eac666f83`.
- Real startup, home page and session API passed with the six expected labels.
- Three generated 640×480 black JPEG frames passed through WebSocket, real Paddle
  inference, NMS, ByteTrack and annotated JPEG output. Packing count remained zero.
- The same check passed from outside the project directory with networking
  restricted and HTTP/HTTPS proxies set to an unavailable localhost endpoint.
  Existing source and weights were reused; the old developer cache was unnecessary.
- Python tests in the isolated copy: 67 passed, 2 skipped for other model profiles.
  The six existing frontend tests also passed.

This verifies installation and execution on this Mac, not recognition accuracy or
identical floating-point predictions across computers. Linux x86_64/WSL2 has a
CPU-only installation path but was not tested on a Linux host. Native Windows,
Intel Macs and Raspberry Pi 3 are outside this pinned setup's supported targets.
