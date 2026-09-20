"""Fail health checks if the app did not load the expected released model."""

import json
from pathlib import Path
import urllib.request


def main():
    manifest = json.loads((Path(__file__).resolve().parents[1] / "models/grocery6-overhead/best.json").read_text())
    with urllib.request.urlopen("http://127.0.0.1:8001/api/session", timeout=3) as response:
        state = json.load(response)
    if (state.get("model_profile") != "ppyoloe_custom"
            or state.get("enabled_classes") != manifest["labels"]
            or state.get("inference_device") != "cpu"
            or state.get("inference_size") != manifest["input_size"]):
        raise SystemExit("Unexpected model configuration")


if __name__ == "__main__":
    main()
