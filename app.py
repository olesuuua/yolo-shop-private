import asyncio
import hashlib
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import (
    MODEL_HF_FILENAME, MODEL_HF_REPO, MODEL_HF_REVISION, MODEL_PATH,
    MODEL_PLATFORM_FILENAME, MODEL_PLATFORM_REF, MODEL_SHA256, MODEL_PROFILE,
)
from vision import FrameProcessor

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


def download_platform_checkpoint(model_root: Path) -> Path:
    """Fetch public weights; keep temporary signed URLs out of configuration."""
    import requests
    from ultralytics_platform import Platform

    cached_path = model_root / ".cache" / "platform-weights" / f"{MODEL_SHA256}.pt"
    if cached_path.is_file():
        return cached_path
    with Platform(api_key="", timeout=30, max_retries=1) as platform:
        files = platform.models.files(*MODEL_PLATFORM_REF)["files"]
    entry = next((file for file in files if file["name"] == MODEL_PLATFORM_FILENAME), None)
    if entry is None:
        raise FileNotFoundError(f"Platform model has no {MODEL_PLATFORM_FILENAME} checkpoint.")
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    partial = cached_path.with_suffix(".download")
    try:
        with requests.get(entry["downloadUrl"], stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        if partial.stat().st_size != entry["size"]:
            raise ValueError("Downloaded model size does not match the Platform metadata.")
        partial.replace(cached_path)
    finally:
        partial.unlink(missing_ok=True)
    return cached_path


def load_model():
    if MODEL_PROFILE in {"ppyoloe_objects365", "ppyoloe_custom"}:
        from ppyoloe_model import load_ppyoloe

        return load_ppyoloe(Path(__file__).resolve().parent)

    if MODEL_PROFILE in {"yoloe26n", "yoloe26x"}:
        from yoloe_model import load_yoloe

        return load_yoloe(Path(__file__).resolve().parent)

    from ultralytics import YOLO

    # Reuse project-local weights offline; download missing weights from their source.
    model_root = Path(__file__).resolve().parent
    model_path = model_root / MODEL_PATH
    checkpoint_path = model_path
    if not model_path.is_file():
        if Path(MODEL_PATH).is_absolute():
            raise FileNotFoundError(f"Local model checkpoint does not exist: {MODEL_PATH}")
        if MODEL_PLATFORM_REF:
            checkpoint_path = download_platform_checkpoint(model_root)
        else:
            from huggingface_hub import hf_hub_download

            checkpoint_path = Path(hf_hub_download(
                repo_id=MODEL_HF_REPO, filename=MODEL_HF_FILENAME,
                revision=MODEL_HF_REVISION, cache_dir=model_root / ".cache" / "huggingface", token=False,
            ))
    if MODEL_SHA256:
        with checkpoint_path.open("rb") as checkpoint:
            digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        if digest != MODEL_SHA256:
            raise ValueError("Model checkpoint checksum mismatch; restore the pinned weights.")
    if checkpoint_path != model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path = model_path.with_suffix(".pt.tmp")
        try:
            shutil.copyfile(checkpoint_path, staged_path)
            staged_path.replace(model_path)
        finally:
            staged_path.unlink(missing_ok=True)
    logger.info("Loading model %s", model_path.name)
    return YOLO(str(model_path))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading at startup keeps imports and API tests independent of model files.
    from identification import IdentificationService
    identifier = IdentificationService()
    identifier.start()
    app.state.identifier = identifier
    app.state.processor = await asyncio.to_thread(
        lambda: FrameProcessor(load_model(), identifier=identifier))
    app.state.active_camera = None
    try:
        yield
    finally:
        await asyncio.to_thread(identifier.close)
        close = getattr(app.state.processor.model, "close", None)
        if callable(close):
            await asyncio.to_thread(close)


app = FastAPI(title="LightStore", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/session")
def get_session():
    return app.state.processor.snapshot()


@app.get("/api/ident-readiness")
def get_ident_readiness():
    """Prerequisites and key availability; never exposes the key itself."""
    identifier = getattr(app.state, "identifier", None)
    if identifier is None:
        return {"catalog_ok": False, "catalog_error": "Identification service is not running.",
                "products": [], "ocr_available": False, "ocr_error": "",
                "ocr_device": "cpu", "jev_key_present": False,
                "jev_model": "", "jev_calls": 0}
    return identifier.readiness()


@app.get("/api/ident-debug")
def get_ident_debug():
    """Per-track OCR evidence and pipeline timings; text only, no key."""
    identifier = getattr(app.state, "identifier", None)
    if identifier is None:
        return {"capture": {}, "queue_depth": 0, "tracks": {}}
    return identifier.debug()


@app.get("/api/ident-crop/{track_id}")
def get_ident_crop(track_id: int):
    """Latest submitted OCR crop for a track (diagnostic preview)."""
    from fastapi.responses import Response
    identifier = getattr(app.state, "identifier", None)
    if identifier is None:
        return Response(status_code=503)
    with identifier.lock:
        evidence = identifier.tracks.get(track_id)
        jpeg = evidence.last_crop_jpeg if evidence is not None else b""
    if not jpeg:
        return Response(status_code=404)
    return Response(content=jpeg, media_type="image/jpeg")


@app.post("/api/reset")
def reset_session():
    return app.state.processor.reset()


@app.websocket("/ws/detect")
async def detect_websocket(websocket: WebSocket):
    await websocket.accept()
    # A persistent YOLO tracker cannot mix frames from different cameras.
    if app.state.active_camera is not None:
        await websocket.send_json({"error": "Another camera is already connected. Stop it first."})
        await websocket.close(code=1008)
        return
    app.state.active_camera = websocket
    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            try:
                response = await asyncio.to_thread(app.state.processor.process, image_bytes)
            except ValueError as error:
                await websocket.send_json({"error": str(error)})
                continue
            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Could not process camera stream")
        try:
            await websocket.send_json({"error": "Camera processing failed. See the server log."})
            await websocket.close(code=1011)
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        if app.state.active_camera is websocket:
            app.state.active_camera = None
