"""Clear-view auto OCR. Installed pipeline only, default settings. CPU."""
import cv2, hashlib, json, sys, time
from pathlib import Path
from PIL import Image, ImageOps
ROOT = Path("/home/olesya/Projects/yolo-shop")
sys.path.insert(0, str(ROOT))
from ocr_worker import load_ocr, _line_entries
import numpy as np

FILES = ["P-front-081946.jpg", "P-side-082007.jpg", "D-front-2288.jpg", "D-side-2289.jpg"]
t0=time.monotonic(); ocr = load_ocr("cpu"); print("load_s", round(time.monotonic()-t0,2), flush=True)
# warmup on small crop
w = cv2.imdecode(np.frombuffer((ROOT/"reports/video-comparison/evidence/v2-warm-a/437_4_122-crop_jpeg.jpg").read_bytes(),dtype=np.uint8), cv2.IMREAD_COLOR)
list(ocr.predict(w))
out = {}
for f in FILES:
    raw = (ROOT/"reports/quality-gate/clearview-baseline/inputs"/f).read_bytes()
    pil = ImageOps.exif_transpose(Image.open(ROOT/"reports/quality-gate/clearview-baseline/inputs"/f))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    t1=time.monotonic(); res = list(ocr.predict(img))[0]; ms=(time.monotonic()-t1)*1000
    lines = _line_entries(res)
    e = {"file": f"inputs/{f}", "sha256": hashlib.sha256(raw).hexdigest(), "raw_dims_wh": [w, h],
         "exif_transposed": True, "predict_ms": round(ms,1),
         "dt_n": len(res.get("dt_polys",[])), "rec_n": len(res.get("rec_texts",[])),
         "det_params": {k: res.get("text_det_params",{}).get(k) for k in ("limit_type","limit_side_len","thresh","box_thresh","unclip_ratio")},
         "rec_score_thresh": res.get("text_rec_score_thresh"),
         "lines": [{"text": l["text"], "score": l["score"], "poly": l.get("poly"), "box": l.get("box")} for l in lines]}
    out[f] = e
    print(f, w, "x", h, "dt:", e["dt_n"], "ms:", round(ms), flush=True)
    for l in lines: print("   ", repr(l["text"])[:50], round(l["score"],3), l.get("box"))
Path("/tmp/opencode/clearview_auto.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("wrote /tmp/opencode/clearview_auto.json")
