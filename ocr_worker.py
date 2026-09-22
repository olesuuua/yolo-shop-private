"""Persistent CPU OCR worker: loads PP-OCRv5 once, reads JPEG crops, returns lines.

Runs under .venv-ocr (never in the web process). One private socket carries
length-prefixed messages (see ppyoloe_protocol): the parent sends raw JPEG
bytes per crop and receives a JSON reply. EOF on the socket stops the worker.
No image or text leaves this machine; no external API is used here.
"""

import argparse
import json
import os
import socket
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_ocr(device):
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / ".cache/paddlex"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if device == "cpu":
        # PP-OCRv5 detection uses a oneDNN op missing from this Paddle build;
        # PaddleX documents this flag to select plain CPU inference instead.
        os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    from paddleocr import PaddleOCR
    return PaddleOCR(
        device=device,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _line_entries(result):
    """One entry per recognized line with crop-pixel geometry.

    Coordinates are pixels of the exact submitted crop: the worker decodes
    the sent JPEG and predicts on it, so rec_polys/rec_boxes map 1:1 to the
    bytes the browser uploaded (origin top-left). Only the poly (4-point
    textline polygon) and the axis-aligned box are exposed; det-time
    polygons and recognition scores are unchanged, and nothing is
    re-recognized or re-cropped.
    """
    import numpy as np

    lines = []
    for text, score, poly, box in zip(
        result["rec_texts"], result["rec_scores"], result["rec_polys"],
        result["rec_boxes"],
    ):
        entry = {"text": str(text), "score": float(score)}
        try:
            points = [[int(x), int(y)] for x, y in np.asarray(poly)][:4]
            if len(points) == 4:
                entry["poly"] = points
        except (TypeError, ValueError):
            pass
        try:
            box_values = [int(v) for v in np.asarray(box)][:4]
            if len(box_values) == 4:
                entry["box"] = box_values
        except (TypeError, ValueError):
            pass
        lines.append(entry)
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu:0"], default="cpu")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from ppyoloe_protocol import receive, send
    with socket.socket(fileno=args.fd) as connection:
        try:
            import cv2
            import numpy as np
            ocr = load_ocr(args.device)
            import paddle
            send(connection, json.dumps({
                "type": "ready", "device": args.device,
                "paddle_version": paddle.__version__,
                "languages": ["ru", "en"],
            }, ensure_ascii=False).encode())
            while True:
                try:
                    payload = receive(connection)
                except EOFError:
                    break
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    send(connection, json.dumps({"type": "error", "message": "Could not decode crop."}).encode())
                    continue
                lines = []
                for result in ocr.predict(frame):
                    lines.extend(_line_entries(result))
                send(connection, json.dumps({"type": "lines", "lines": lines}, ensure_ascii=False).encode())
        except Exception as error:
            traceback.print_exc()
            try:
                send(connection, json.dumps({"type": "error", "message": str(error)}).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
            raise


if __name__ == "__main__":
    main()
