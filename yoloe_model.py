"""Pinned YOLOE weights and cached text prompts for offline packing inference."""

import hashlib
import json
import logging
from pathlib import Path

TEXT_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts"
TEXT_SHA256 = "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f"
logger = logging.getLogger(__name__)


def verify_file(path: Path, expected: str) -> None:
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Checksum mismatch for {path.name}; restore the pinned weights.")


def download_verified(url: str, path: Path, expected: str) -> Path:
    if path.is_file():
        verify_file(path, expected)
        return path
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".download")
    try:
        logger.info("Downloading %s", path.name)
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        verify_file(partial, expected)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def prepare_prompts(model, classes: tuple[str, ...], root: Path, checkpoint_sha: str) -> None:
    import ultralytics

    # Different labels, checkpoint or library version must not reuse stale embeddings.
    signature = json.dumps([checkpoint_sha, TEXT_SHA256, ultralytics.__version__, model.task, classes])
    key = hashlib.sha256(signature.encode()).hexdigest()
    cache = root / ".cache" / "yoloe-prompts" / f"{key}.npz"
    if cache.is_file():
        model.load_prompt_embeddings(cache)
    else:
        import torch
        from ultralytics.nn.text_model import MobileCLIPTS

        text_path = download_verified(TEXT_URL, root / "weights/yoloe-26x/mobileclip2_b.ts", TEXT_SHA256)
        # Set the encoder's absolute path without changing the server's working directory.
        model.model.clip_model = MobileCLIPTS(device=torch.device("cpu"), weight=str(text_path))
        try:
            embeddings = model.model.get_text_pe(list(classes), cache_clip_model=True)
            model.set_classes(list(classes), embeddings)
        finally:
            del model.model.clip_model
        cache.parent.mkdir(parents=True, exist_ok=True)
        staged = cache.with_suffix(".tmp.npz")
        try:
            model.save_prompt_embeddings(staged)
            staged.replace(cache)
        finally:
            staged.unlink(missing_ok=True)
    if list(model.names.values()) != list(classes):
        raise ValueError("YOLOE prompt labels do not match the configured food classes.")


def load_yoloe(root: Path):
    from ultralytics import YOLOE
    from config import MODEL_PATH, MODEL_PROFILE, MODEL_SHA256, YOLOE_CLASSES, YOLOE_MODEL_URL

    path = download_verified(YOLOE_MODEL_URL, root / MODEL_PATH, MODEL_SHA256)
    if MODEL_PROFILE == "yoloe26n":
        # Transfer the pretrained detection weights; packing never uses masks.
        model = YOLOE("yoloe-26n.yaml").load(str(path))
    else:
        model = YOLOE(str(path))
    prepare_prompts(model, YOLOE_CLASSES, root, MODEL_SHA256)
    logger.info("Loaded %s (%s) with classes: %s", MODEL_PROFILE, model.task, ", ".join(YOLOE_CLASSES))
    return model
