import cv2, json, sys, time
from pathlib import Path
from PIL import Image, ImageOps
ROOT = Path("/home/olesya/Projects/yolo-shop")
sys.path.insert(0, str(ROOT))
from ocr_worker import load_ocr
import numpy as np
MANUAL = {
 "P-front-081946.jpg": [("P-arc", [[850,2250],[2150,2250],[2150,2790],[850,2790]], "ПРОСТОКВАШИНО")],
 "D-front-2288.jpg": [("D-up", [[1080,1920],[1760,1920],[1760,2240],[1080,2240]], "ДОМИК"),
                      ("D-lo", [[1140,2020],[1920,2020],[1920,2380],[1140,2380]], "В ДЕРЕВНЕ")],
}
ocr = load_ocr("cpu")
crop_op = ocr.paddlex_pipeline._pipeline._crop_by_polys
rec = ocr.paddlex_pipeline._pipeline.text_rec_model
# warmup
w = np.zeros((32,128,3), np.uint8)+200
list(rec([w]))
out = {}
sdir = Path("reports/quality-gate/clearview-baseline/subs"); sdir.mkdir(exist_ok=True)
for f, lines in MANUAL.items():
    pil = ImageOps.exif_transpose(Image.open(ROOT/"reports/quality-gate/clearview-baseline/inputs"/f))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    for lid, quad, gt in lines:
        sub = list(crop_op(img, [quad]))[0]
        cv2.imwrite(str(sdir/(lid+".jpg")), sub)
        t=time.monotonic(); r=list(rec([sub]))[0]; ms=(time.monotonic()-t)*1000
        e={"file":f,"quad":quad,"gt":gt,"sub_shape":list(sub.shape),
           "text":str(r.get("rec_text","")),"score":float(r.get("rec_score",-1)),"ms":round(ms,1)}
        out[lid]=e
        print(lid, repr(e["text"])[:40], e["score"], e["sub_shape"], e["ms"], flush=True)
Path("/tmp/opencode/clearview_manual.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("wrote /tmp/opencode/clearview_manual.json")
