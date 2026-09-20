"""Fetch pinned official PaddleDetection source, Objects365 weights and labels."""

from pathlib import Path
import subprocess
import tempfile

from ppyoloe_assets import (CHECKPOINT_URL, CHECKPOINT_SHA256, LABELS_URL,
                           LABELS_SHA256, PADDLEDETECTION_COMMIT)
from yoloe_model import download_verified


def prepare_source(root):
    """Restore the exact upstream source; never accept local patches as a dependency."""
    root = Path(root)
    source = root / ".cache/PaddleDetection"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        # A failed download must not leave a broken checkout at the final path.
        with tempfile.TemporaryDirectory(prefix="paddledetection-", dir=source.parent) as temporary:
            staged = Path(temporary) / "source"
            subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                            "https://github.com/PaddlePaddle/PaddleDetection.git", str(staged)], check=True)
            subprocess.run(["git", "-C", str(staged), "checkout", "--detach", PADDLEDETECTION_COMMIT], check=True)
            staged.rename(source)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != PADDLEDETECTION_COMMIT:
        raise ValueError(f"Existing PaddleDetection checkout must be at {PADDLEDETECTION_COMMIT}; it was not changed.")
    changes = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=normal"], text=True)
    if changes.strip():
        raise ValueError("PaddleDetection has local modifications. Move that checkout aside and rerun setup; it was not changed.")
    return source


def main():
    root = Path(__file__).resolve().parent
    prepare_source(root)
    assets = root / "weights/ppyoloe-objects365"
    download_verified(CHECKPOINT_URL, assets / "ppyoloe_crn_s_obj365_pretrained.pdparams", CHECKPOINT_SHA256)
    download_verified(LABELS_URL, assets / "objects365_labels.txt", LABELS_SHA256)
    print("PP-YOLOE+ Small source, weights and labels verified. No ONNX export is needed.")


if __name__ == "__main__":
    main()
