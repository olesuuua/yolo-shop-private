"""Download and verify the published six-class overhead checkpoint."""

from pathlib import Path

from ppyoloe_checkpoint import read_manifest
from prepare_ppyoloe import prepare_source
from yoloe_model import download_verified

RELEASE_URL = (
    "https://github.com/RuslanGreenhead/yolo_shop/releases/download/"
    "grocery6-overhead-2026-09-17"
)


def main():
    root = Path(__file__).resolve().parent
    prepare_source(root)
    manifest_path = root / "models/grocery6-overhead/best.json"
    manifest, checkpoint = read_manifest(manifest_path, verify_weights=False)
    download_verified(f"{RELEASE_URL}/{checkpoint.name}", checkpoint,
                      manifest["checkpoint_sha256"])
    read_manifest(manifest_path)
    print(f"Six-class weights verified: {checkpoint}")
    print(f"LIGHTSTORE_PPYOLOE_MANIFEST={manifest_path}")


if __name__ == "__main__":
    main()
