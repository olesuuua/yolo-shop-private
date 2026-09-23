"""Direct-recognizer oracle. CPU only. Wrappers only; no package/production edits."""
import cv2, difflib, hashlib, json, sys, time
from pathlib import Path
ROOT = Path("/home/olesya/Projects/yolo-shop")
sys.path.insert(0, str(ROOT))
from ocr_worker import load_ocr
import numpy as np

ocr = load_ocr("cpu")
inner = ocr.paddlex_pipeline._pipeline
crop_op = inner._crop_by_polys  # normal crop operation (quad)
rec_model = inner.text_rec_model
print("crop_op:", type(crop_op).__name__, "det_box_type=", getattr(crop_op, "det_box_type", "?"))

LINES = [
    ("571-M1", "reports/video-comparison/evidence/v3-warm-b/571_3_311-crop_jpeg.jpg", [[170,640],[480,610],[490,710],[180,740]], "ДОМИК", "script; ligature uncertainty"),
    ("571-M2", "reports/video-comparison/evidence/v3-warm-b/571_3_311-crop_jpeg.jpg", [[190,720],[470,700],[480,800],[200,820]], "В ДЕРЕВНЕ", "script; partial uncertainty"),
    ("571-B2", "reports/video-comparison/evidence/v3-warm-b/571_3_311-crop_jpeg.jpg", [[239,743],[469,663],[486,712],[255,792]], "ДОМИК (B geometry)", "exact B-run brand box 1"),
    ("571-B3", "reports/video-comparison/evidence/v3-warm-b/571_3_311-crop_jpeg.jpg", [[215,768],[409,720],[423,777],[229,825]], "В ДЕРЕВНЕ (B geometry)", "exact B-run brand box 2"),
    ("404-M1", "reports/video-comparison/evidence/v1-ocr-cold/404_2_61-crop_jpeg.jpg", [[360,1380],[820,1330],[850,1560],[380,1600]], "ПРОСТОКВАШИНО", "curved band; quad approximate"),
    ("437-M1", "reports/video-comparison/evidence/v2-warm-a/437_3_121-crop_jpeg.jpg", [[120,950],[380,960],[370,1080],[120,1070]], "ПРОСТОКВАШИНО", "small+partial; uncertain"),
    ("455-M1", "reports/video-comparison/evidence/v3-warm-a/455_3_148-crop_jpeg.jpg", [[380,450],[498,430],[498,750],[380,770]], "ДОМИК В ДЕРЕВНЕ", "rotated/small; highly uncertain"),
    ("388-C1", "reports/video-comparison/evidence/v1-ocr-cold/388_2_31-crop_jpeg.jpg", [[232,860],[632,749],[662,858],[262,970]], "СВЯТОЙ", "positive control"),
    ("388-C2", "reports/video-comparison/evidence/v1-ocr-cold/388_2_31-crop_jpeg.jpg", [[176,935],[672,815],[704,948],[207,1068]], "ИСТОЧНИК", "positive control"),
    ("437-C", "reports/video-comparison/evidence/v2-warm-a/437_4_122-crop_jpeg.jpg", [[102,661],[357,698],[342,803],[86,766]], "AKBA", "positive control"),
]
from identification import normalize
def norm(s): return normalize(s)

# warmup rec
_w = np.zeros((32, 128, 3), np.uint8) + 200
list(rec_model([_w]))
RAW = Path("/tmp/opencode/oracle_raw"); (RAW/"subs").mkdir(parents=True, exist_ok=True)
out = {"provenance": {"models": ["PP-OCRv5_mobile_det(not used here)", "cyrillic_PP-OCRv5_mobile_rec"], "crop_op": "CropByPolys quad get_minarea_rect_crop (perspective warp, INTER_CUBIC, BORDER_REPLICATE)", "rec_preprocessing": "model-internal resize/normalize; UNAVAILABLE for inspection beyond input sub shapes", "alt_extraction": "axis-aligned bbox +4px pad (boundary check only)"}, "lines": {}}
for lid, rel, quad, gt, note in LINES:
    raw = (ROOT/rel).read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    e = {"source": rel, "sha256": hashlib.sha256(raw).hexdigest(), "img_wh": [img.shape[1], img.shape[0]], "quad": quad, "gt": gt, "note": note}
    for variant, poly in (("primary", quad),):
        subs = list(crop_op(img, [poly]))
        assert len(subs) == 1
        sub = subs[0]
        e["sub_shape"] = list(sub.shape)
        cv2.imwrite(str(RAW/"subs"/f"{lid}.jpg"), sub)
        t1=time.monotonic(); res=list(rec_model([sub])); ms=(time.monotonic()-t1)*1000
        r=res[0]
        e["raw_text"]=str(r.get("rec_text","")); e["score"]=float(r.get("rec_score",-1)); e["ms"]=round(ms,1)
    # one predefined alternative: axis bbox +4px
    xs=[p[0] for p in quad]; ys=[p[1] for p in quad]
    x0,y0,x1,y1=max(0,min(xs)-4),max(0,min(ys)-4),min(img.shape[1],max(xs)+4),min(img.shape[0],max(ys)+4)
    alt=img[y0:y1,x0:x1]
    e["alt_shape"]=list(alt.shape)
    cv2.imwrite(str(RAW/"subs"/f"{lid}_alt.jpg"), alt)
    t1=time.monotonic(); res=list(rec_model([alt])); ms=(time.monotonic()-t1)*1000
    r=res[0]
    e["alt_text"]=str(r.get("rec_text","")); e["alt_score"]=float(r.get("rec_score",-1)); e["alt_ms"]=round(ms,1)
    # accuracy
    for key,txt in (("raw_text","\0"),):
        pass
    def acc(pred, gt):
        pn, gn = norm(pred), norm(gt)
        return {"norm_pred": pn, "ratio": round(difflib.SequenceMatcher(None, pn, gn).ratio(), 3), "exact": pn == gn}
    e["acc"]=acc(e["raw_text"], gt); e["alt_acc"]=acc(e["alt_text"], gt)
    out["lines"][lid]=e
    print(lid, repr(e["raw_text"]), e["score"], '| alt:', repr(e["alt_text"]), e["alt_score"], e["sub_shape"], round(e["ms"]), flush=True)
Path("/tmp/opencode/oracle_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("wrote /tmp/opencode/oracle_result.json")
