"""Offline gate comparison (CPU OCR held constant).

OCR outputs are reused from ocr-representatives.json (156 group-deduped reps)
plus ocr-targeted.json headlines (404:2:61 readable P, 444:2:133
accepted-empty). Recognition model/settings/preprocessing are identical for
every gate variant: gates only decide accept/reject, never re-run OCR.
Tuning/validation splits are group-aware (near-duplicates never split).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QD = Path(__file__).resolve().parent


def norm(text):
    return re.sub(r"\s+", "", str(text).upper())


def target_hit(pid, lines):
    txt = " ".join(norm(l.get("text", "")) for l in lines)
    if pid == "A":
        return "AQUA" in txt or "AKBA" in txt
    if pid == "S":
        return ("СЕНЕЖ" in txt or "СEНЕЖ" in txt or "CEНЕЖ" in txt
                or "SENEZH" in txt)
    if pid == "W":
        return ("СВЯТ" in txt or "CBЯT" in txt or "ИСТОЧ" in txt
                or "UСТОЧ" in txt)
    if pid == "P":
        return "ПРОСТОКВАШИНО" in txt or "PROSTOKVASHINO" in txt
    if pid == "D":
        return "ДОМИК" in txt or "ДЕРЕВНЕ" in txt or "DOMIK" in txt
    return False


def neighbor_hit(pid, lines):
    txt = " ".join(norm(l.get("text", "")) for l in lines)
    brands = set()
    if "AQUA" in txt or "AKBA" in txt:
        brands.add("A")
    if "СЕНЕЖ" in txt or "СEНЕЖ" in txt or "CEНЕЖ" in txt:
        brands.add("S")
    if "СВЯТ" in txt or "CBЯT" in txt or "ИСТОЧ" in txt or "UСТОЧ" in txt:
        brands.add("W")
    if "ПРОСТОКВАШИНО" in txt:
        brands.add("P")
    if "ДОМИК" in txt or "ДЕРЕВНЕ" in txt or "DOMIK" in txt:
        brands.add("D")
    return bool(brands - {pid})


def empty_ocr(lines):
    return not any(str(l.get("text", "")).strip() for l in lines)


def main():
    manifest = json.loads((QD / "crops-manifest.json").read_text())
    reps = json.loads((QD / "ocr-representatives.json").read_text())["crops"]
    targeted = {}
    p = QD / "ocr-targeted.json"
    if p.exists():
        targeted = json.loads(p.read_text())
    rows = {r["request_id"]: r for r in manifest["rows"]}
    sharp_all = json.loads((QD / "sharpness-all.json").read_text())

    # Eval OCR pool: 156 reps + 2 headlines (404:2:61, 444:2:133).
    pool = dict(reps)
    for rid in ("404:2:61", "444:2:133"):
        if rid in targeted:
            pool[rid] = targeted[rid]
    # Extra P confirmation set (reported separately, not in gate rates).
    extra_p = [r for r in ("523:2:240", "410:2:77", "409:2:74", "405:2:63",
                           "519:2:230", "520:2:232") if r in targeted]

    gates = {
        "current_full15": lambda e: (e["full"] or -1) >= 15.0,
        "bypass": lambda e: True,
        "norm1000_gte30": lambda e: (e["normalized"]["1000"] or -1) >= 30.0,
        "union_full15_or_n1000g30": lambda e: ((e["full"] or -1) >= 15.0
                    or (e["normalized"]["1000"] or -1) >= 30.0),
    }
    out = {"schema": "quality-gate-eval-2", "pool_n": len(pool),
           "extra_p_generic_only": sum(
               1 for r in extra_p
               if not target_hit("P", targeted[r]["ocr"]["lines"])
               and any("МОЛОКО" in str(l["text"]).upper().replace("О", "О")
                       or "MOLOKO" in norm(l["text"])
                       for l in targeted[r]["ocr"]["lines"])),
           "splits": {}}
    for split in ("tuning", "validation"):
        ids = [k for k in pool if rows[k]["split"] == split
               and rows[k]["physical_id"] in "ASWPD"]
        res = {"n": len(ids), "gates": {}}
        for name, fn in gates.items():
            sel = [k for k in ids if fn(pool[k])]
            hits = sum(1 for k in sel
                       if target_hit(rows[k]["physical_id"],
                                     pool[k]["ocr"]["lines"]))
            empt = sum(1 for k in sel if empty_ocr(pool[k]["ocr"]["lines"]))
            nb = sum(1 for k in sel
                     if neighbor_hit(rows[k]["physical_id"],
                                     pool[k]["ocr"]["lines"]))
            ms = sum(pool[k]["ocr"]["ocr_ms"] for k in sel)
            by_pid = {}
            for pid in "ASWPD":
                s = [k for k in sel if rows[k]["physical_id"] == pid]
                by_pid[pid] = {"acc": len(s), "hit": sum(
                    1 for k in s if target_hit(pid, pool[k]["ocr"]["lines"]))}
            res["gates"][name] = {
                "acc": len(sel), "hit": hits, "empty": empt,
                "neighbor": nb, "ocr_ms": round(ms, 1),
                "ms_mean": round(ms / max(1, len(sel)), 1), "by_pid": by_pid,
                "missed_hits": sum(1 for k in ids
                    if k not in sel and target_hit(
                        rows[k]["physical_id"], pool[k]["ocr"]["lines"])),
            }
        out["splits"][split] = res

    # Full-329 gate-level workload (all crops, sharpness only).
    wl = {}
    for split in ("tuning", "validation"):
        ids = [r["request_id"] for r in manifest["rows"]
               if r["split"] == split]
        wl[split] = {
            "n": len(ids),
            "cur": sum(1 for k in ids if sharp_all[k]["full"] >= 15.0),
            "union": sum(1 for k in ids if sharp_all[k]["full"] >= 15.0
                         or sharp_all[k]["normalized"]["1000"] >= 30.0),
        }
    out["full329_workload"] = wl
    (QD / "eval-results.json").write_text(
        json.dumps(out, indent=1) + "\n")

    lines = ["# Quality-gate evaluation (tuning vs held-out validation)",
             "",
             "Pool: 156 group-deduped reps + 2 headlines (404:2:61 P readable,",
             "444:2:133 accepted-empty) = %d OCR'd crops. +6 extra P blurry confirm" % len(pool),
             "generic-only pattern (reported separately). OCR model/settings/",
             "preprocessing identical for all gates (CPU PP-OCRv5, ocr_worker.py).",
             "Brand-hit = strict target roots (AQUA/AKBA; СЕНЕЖ*; СВЯТ/CBЯT/ИСТОЧ*;",
             "ПРОСТОКВАШИНО; ДОМИК/ДЕРЕВНЕ/DOMIK). Lone AKB and DOM fragments",
             "do not count. Splits are group-aware (dhash hamming<=6).",
             "",
             "| split | gate | acc | target-hit | empty | neighbor-brand | ocr_ms | missed |",
             "|---|---|---|---|---|---|---|---|"]
    for split in ("tuning", "validation"):
        for name in ("current_full15", "bypass", "norm1000_gte30",
                     "union_full15_or_n1000g30"):
            g = out["splits"][split]["gates"][name]
            lines.append(
                f"| {split} | {name} | {g['acc']} | {g['hit']} | {g['empty']} "
                f"| {g['neighbor']} | {g['ocr_ms']} | {g['missed_hits']} |")
    lines += ["",
              f"Extra P confirmation: {out['extra_p_generic_only']}/6 additional",
              "P blurry → МОЛОКО/2,5% only, 0 brand (curved-logo recognition limit).",
              "",
              "Full-329 gate-level admits (sharpness only, incl. duplicates",
              "that downstream dedup/throttle partly suppress):"]
    for split in ("tuning", "validation"):
        w = wl[split]
        lines.append(
            f"- {split}: n={w['n']} current={w['cur']} union={w['union']} "
            f"extra={w['union']-w['cur']} ({w['union']/max(1,w['cur']):.2f}x)")
    (QD / "eval-table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
