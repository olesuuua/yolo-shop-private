"""Controlled detector-recall A/B. CPU only. Wrappers only; no package/production edits."""
import cv2, hashlib, json, sys, time
from pathlib import Path
ROOT = Path("/home/olesya/Projects/yolo-shop")
sys.path.insert(0, str(ROOT))
from ocr_worker import load_ocr, _line_entries
import numpy as np

CROPS = [
    ("571:3:311", "reports/video-comparison/evidence/v3-warm-b/571_3_311-crop_jpeg.jpg", "milk-D"),
    ("404:2:61", "reports/video-comparison/evidence/v1-ocr-cold/404_2_61-crop_jpeg.jpg", "milk-P"),
    ("437:3:121", "reports/video-comparison/evidence/v2-warm-a/437_3_121-crop_jpeg.jpg", "milk-P"),
    ("455:3:148", "reports/video-comparison/evidence/v3-warm-a/455_3_148-crop_jpeg.jpg", "milk-D"),
    ("388:2:31", "reports/video-comparison/evidence/v1-ocr-cold/388_2_31-crop_jpeg.jpg", "ctrl-W"),
    ("437:4:122", "reports/video-comparison/evidence/v2-warm-a/437_4_122-crop_jpeg.jpg", "ctrl-A"),
    ("463:7:160", "reports/video-comparison/evidence/v3-warm-a/463_7_160-crop_jpeg.jpg", "neg-W"),
    ("580:3:329", "reports/video-comparison/evidence/v3-warm-b/580_3_329-crop_jpeg.jpg", "neg-S"),
    ("172:2:232", "reports/quality-gate/replay-comparison/runs/v1-b2/crops/172_2_232.jpg", "neg-W2"),
]
# Manual regions (approx, human-estimated from crop viewing; ±50px).
MANUAL = {
    "571:3:311": {"target": [150, 620, 500, 760], "neighbor": None, "generic": [180, 760, 430, 840]},
    "404:2:61": {"target": [330, 1350, 850, 1560], "neighbor": [0, 538, 87, 684], "generic": [429, 1548, 711, 1731]},
    "437:3:121": {"target": [120, 950, 380, 1080], "neighbor": None, "generic": [167, 1088, 329, 1175]},
    "455:3:148": {"target": [380, 450, 498, 750], "neighbor": None, "generic": None},
    "388:2:31": {"target": [176, 749, 704, 1068], "neighbor": [200, 579, 523, 797], "generic": None},
    "437:4:122": {"target": [86, 661, 357, 803], "neighbor": None, "generic": None},
    "463:7:160": {"target": [280, 380, 470, 520], "neighbor": [0, 663, 136, 796], "generic": None},
    "580:3:329": {"target": [107, 503, 479, 613], "neighbor": [428, 0, 510, 31], "generic": [428, 0, 510, 31]},
    "172:2:232": {"target": None, "neighbor": [0, 1267, 226, 1424], "generic": None},
}
RAWDIR = Path("/tmp/opencode/ab_raw"); RAWDIR.mkdir(parents=True, exist_ok=True)

ocr = load_ocr("cpu")
inner = ocr.paddlex_pipeline._pipeline
orig_det, orig_rec = inner.text_det_model, inner.text_rec_model
LOG = {"det_raw": [], "rec": []}
def det_wrapper(images, **kw):
    res = list(orig_det(images, **kw))
    for r in res:
        LOG["det_raw"].append({"dt_raw": len(r.get("dt_polys", [])), "kw_box_thresh": kw.get("box_thresh", "<default>")})
    return iter(res)
def rec_wrapper(subs, **kw):
    subs = list(subs)
    LOG["rec"].append({"n_subs": len(subs), "shapes": [list(s.shape) for s in subs]})
    # save subs (bounded: 9 crops * ~6 boxes * 4 runs = ~200 tiny images)
    base = LOG.get("save_base", "x")
    for i, s in enumerate(subs):
        try:
            cv2.imwrite(str(RAWDIR / "subs" / f"{base}_{i}.jpg"), s)
        except Exception:
            pass
    res = list(orig_rec(subs, **kw))
    LOG["rec"][-1]["rec_texts"] = [str(x.get("rec_text", ""))[:30] for x in res]
    LOG["rec"][-1]["rec_scores"] = [float(x.get("rec_score", -1)) for x in res]
    return iter(res)
inner.text_det_model, inner.text_rec_model = det_wrapper, rec_wrapper
(RAWDIR / "subs").mkdir(exist_ok=True)

# warmup (discarded)
_w = cv2.imdecode(np.frombuffer((ROOT / CROPS[5][1]).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
list(ocr.predict(_w))
LOG.clear(); LOG.update({"det_raw": [], "rec": []})

out = {"conditions": {"A": {"box_thresh": 0.6, "note": "config default"}, "B": {"box_thresh": 0.4, "note": "override only"}}, "sequence": "A-B-B-A per crop (warmed)", "crops": {}}
for rid, rel, kind in CROPS:
    raw = (ROOT / rel).read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    entry = {"path": rel, "sha256": hashlib.sha256(raw).hexdigest(), "dims_wh": [w, h], "bytes": len(raw), "kind": kind, "manual": MANUAL[rid], "runs": []}
    for seq, (cond, kw) in enumerate([("A", {}), ("B", {"text_det_box_thresh": 0.4}), ("B", {"text_det_box_thresh": 0.4}), ("A", {})]):
        LOG["det_raw"].clear(); LOG["rec"].clear()
        LOG["save_base"] = f"{rid.replace(':','_')}_{cond}_{seq}"
        t1 = time.monotonic()
        res = list(ocr.predict(img, **kw))[0]
        ms = (time.monotonic() - t1) * 1000
        det_raw_n = LOG["det_raw"][0]["dt_raw"] if LOG["det_raw"] else "<UNAVAILABLE>"
        dt_final = len(res.get("dt_polys", []))
        invalid_removed = (det_raw_n - dt_final) if isinstance(det_raw_n, int) else "<UNAVAILABLE>"
        lines = _line_entries(res)
        tdp = dict(res.get("text_det_params", {}))
        run = {"seq": seq, "cond": cond, "ms": round(ms, 1), "det_input_wh": [w, h],
               "limit_type": tdp.get("limit_type", "<UNAVAILABLE>"), "limit_side_len": tdp.get("limit_side_len", "<UNAVAILABLE>"),
               "thresh": tdp.get("thresh", "<UNAVAILABLE>"), "box_thresh_eff": tdp.get("box_thresh", "<UNAVAILABLE>"),
               "unclip": tdp.get("unclip_ratio", "<UNAVAILABLE>"), "rec_score_thresh": res.get("text_rec_score_thresh", "<UNAVAILABLE>"),
               "dt_raw": det_raw_n, "dt_final": dt_final, "invalid_removed": invalid_removed,
               "rec_n_subs": LOG["rec"][0]["n_subs"] if LOG["rec"] else "<UNAVAILABLE>",
               "rec_texts": [r.get("text") for r in lines], "rec_scores": [r.get("score") for r in lines],
               "rec_polys": [r.get("poly") for r in lines], "lines": lines}
        entry["runs"].append(run)
        print(rid, cond, seq, "dt_raw", det_raw_n, "dt", dt_final, "rec", [t for t in run["rec_texts"]], round(ms), flush=True)
    out["crops"][rid] = entry
inner.text_det_model, inner.text_rec_model = orig_det, orig_rec
Path("/tmp/opencode/ab_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("wrote /tmp/opencode/ab_result.json")
