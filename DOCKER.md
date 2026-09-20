# LightStore in Docker

The container includes the released **six-class PP-YOLOE+ Small** model, both
Python environments and the pinned PaddleDetection source. It runs on CPU.
You do not need to install Python, Paddle, PyTorch or copy anyone's local cache.

## Start

Install and start [Docker Desktop](https://docs.docker.com/desktop/) on macOS or
Windows (use Linux containers), or Docker Engine with the Compose plugin on
Linux. Then clone this branch, or download its source ZIP and open a terminal
in the extracted folder:

```bash
git clone --branch codex/lightstore-packing-mvp https://github.com/RuslanGreenhead/yolo_shop.git
cd yolo_shop
docker compose up --build -d --wait
```

Open **http://localhost:8001/**, allow camera access and click **Start camera**.
The browser captures the camera and sends JPEGs to the container over WebSocket;
USB/device passthrough is not needed. Keep the URL on `localhost`/`127.0.0.1` for
browser camera permissions. The published port listens only on this computer.

The initial build downloads packages and model assets and can take several
minutes. Later starts reuse the image and work without internet access. Allow
at least 4 GB RAM for Docker (8 GB recommended), several CPU cores and 8 GB of
free disk for the image and build cache. No GPU is required.

```bash
docker compose logs -f          # Show startup/errors; Ctrl+C closes only the log view
docker compose down             # Stop the app
docker compose up -d --wait      # Start the existing image
```

Session counts live in RAM and reset when the container restarts. The service
supports one active camera and runs one web worker.

### If port 8001 is occupied

Stop the other local server, or choose another host port:

macOS/Linux:

```bash
LIGHTSTORE_PORT=8002 docker compose up -d --wait
```

Windows PowerShell:

```powershell
$env:LIGHTSTORE_PORT = "8002"
docker compose up -d --wait
```

Then open http://localhost:8002/. The container's internal port remains 8001.

## Architectures and limits

The Dockerfile targets Linux **amd64** (Intel/AMD) and **arm64** (Apple Silicon
and 64-bit ARM hosts). Docker builds the variant matching its engine, without
forcing x86 emulation on an ARM Mac. macOS/Windows use Docker's Linux VM.
Both architectures have native build and real-inference jobs in
[GitHub Actions](https://github.com/RuslanGreenhead/yolo_shop/actions).

The ARM Paddle wheel comes from the
[official Paddle CPU index](https://www.paddlepaddle.org.cn/packages/stable/cpu/paddlepaddle/),
because this version's Linux ARM wheel is absent from PyPI. Both Paddle wheel
URLs and their SHA-256 hashes are pinned. PyTorch uses official CPU-only wheels.
Both Python base images are pinned by digest and use the same Debian release.

Docker standardizes the software environment; it does not make every processor
equally fast or increase RAM. A Raspberry Pi 3 and 32-bit systems are not viable
targets for this runtime. The Windows/Linux/macOS Docker host must itself be
supported by Docker. Camera quality, lighting and hardware still affect results.

## Verify without a camera or internet

After the image has been built:

```bash
docker run --rm --network none --entrypoint /app/.venv/bin/python lightstore:grocery6 run_lightstore.py --check
```

This loads the real weights and sends three generated black frames through
HTTP/WebSocket, Paddle inference, tracking and JPEG encoding. It verifies model
identity and pipeline execution, not grocery recognition accuracy. The container
health check also verifies the expected six labels, CPU device and input size.

The image includes `apple`, `bottle`, `can`, `chocolate bar`, `pack of crisps`,
`pack of muffins`. Model checksum:
`a7d4f66ac449016bcc006d3ff46e64f46d6d05d3f8ad5364245e395eac666f83`.
See [the training report](reports/grocery6-overhead.md) for accuracy limitations.

The Docker context uses an allowlist: local environments, datasets, old weights,
training runs, personal files and the unrelated landing page are excluded.
The runtime runs as an unprivileged user and downloads nothing at startup.
