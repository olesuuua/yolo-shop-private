# Both official Python images use Debian bookworm and contain amd64/arm64 variants.
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3 AS paddle-python

FROM python:3.14.2-slim-bookworm@sha256:e87711ef5c86aaeaa7031718a69db79d334d94c545c709583f651b8185870941
ARG TARGETARCH
LABEL org.opencontainers.image.title="LightStore six-class grocery detector" \
      org.opencontainers.image.source="https://github.com/RuslanGreenhead/yolo_shop"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=paddle-python /usr/local/ /opt/paddle-python/
RUN printf '%s\n' /opt/paddle-python/lib > /etc/ld.so.conf.d/paddle-python.conf && ldconfig

WORKDIR /app
COPY requirements.txt requirements-paddle.txt ./
RUN python -m venv /app/.venv \
    && /opt/paddle-python/bin/python3.11 -m venv /app/.venv-ppyolo-export \
    && /app/.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        'torch==2.14.0+cpu' 'torchvision==0.29.0+cpu' \
    && /app/.venv/bin/python -m pip install --prefer-binary -r requirements.txt \
    && /app/.venv/bin/python -m pip check

# PyPI does not carry the Linux ARM wheel for this release; use Paddle's own CPU
# wheels on both architectures. Pip checks each archive against the pinned digest.
RUN case "$TARGETARCH" in \
      amd64) wheel='https://paddle-whl.cdn.bcebos.com/stable/cpu/paddlepaddle/paddlepaddle-3.3.1-cp311-cp311-linux_x86_64.whl#sha256=bb22396233d807f6957d9c4a6a1edc76693fb589c4ee3d17caf7e39113ca4047' ;; \
      arm64) wheel='https://paddle-whl.cdn.bcebos.com/stable/cpu/paddlepaddle/paddlepaddle-3.3.1-cp311-cp311-linux_aarch64.whl#sha256=66bf8aab382c84785988f46c6b1f653373684047d94bce09fc88f5b79350ba12' ;; \
      *) echo "Unsupported architecture: $TARGETARCH (use amd64 or arm64)" >&2; exit 1 ;; \
    esac \
    && /app/.venv-ppyolo-export/bin/python -m pip install --no-deps "$wheel" \
    && /app/.venv-ppyolo-export/bin/python -m pip install --prefer-binary -r requirements-paddle.txt \
    && /app/.venv-ppyolo-export/bin/python -m pip check

COPY *.py bytetrack-*.yaml *_classes.json ./
COPY static/ ./static/
COPY models/grocery6-overhead/best.json ./models/grocery6-overhead/best.json
COPY docker/ ./docker/
COPY tests/ ./tests/

ENV LIGHTSTORE_MODEL=ppyoloe_custom \
    LIGHTSTORE_PPYOLOE_MANIFEST=/app/models/grocery6-overhead/best.json \
    LIGHTSTORE_PADDLE_PYTHON=/app/.venv-ppyolo-export/bin/python \
    LIGHTSTORE_IMGSZ=416 \
    YOLO_CONFIG_DIR=/app/.cache/ultralytics \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    YOLO_AUTOINSTALL=false

RUN groupadd --gid 10001 lightstore \
    && useradd --uid 10001 --gid lightstore --create-home lightstore \
    && mkdir -p /app/.cache \
    && chown -R lightstore:lightstore /app/.cache /app/models
USER lightstore
# Bake source and verified model weights into the image, not a host-mounted cache.
RUN /app/.venv/bin/python prepare_grocery6.py

EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=4 \
    CMD ["python", "/app/docker/healthcheck.py"]
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
