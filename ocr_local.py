"""Read Russian/English packaging text locally on CPU or GPU; save reviewable JSON."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

CPU_NO_MKLDNN = "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"

def main():
    python = ROOT / ".venv-ocr/bin/python"
    if Path(sys.prefix) != ROOT / ".venv-ocr":
        os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["gpu:0", "cpu"], default="gpu:0")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("Input image does not exist.")
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve reviewed text.")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(ROOT / ".cache/paddlex"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if args.device == "cpu":
        # PP-OCRv5 detection uses a oneDNN op missing from this Paddle build;
        # PaddleX documents this flag to select plain CPU inference instead.
        os.environ.setdefault(CPU_NO_MKLDNN, "False")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        device=args.device,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    lines = []
    for result in ocr.predict(str(args.image)):
        for text, score in zip(result["rec_texts"], result["rec_scores"]):
            lines.append({"text": text, "score": float(score)})
    report = {"image": str(args.image.resolve()), "device": args.device,
              "languages": ["ru", "en"], "reviewed": False, "lines": lines}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
