"""Build the quality-gate evaluation set from saved video-comparison crops.

Groups near-duplicate crops (perceptual dhash, union-find on hamming distance)
and assigns each group to the tuning or held-out split so tuning and
validation never use nearly identical images. Reads per-request.json for
sharpness/status/physical identity recorded by the original comparison runs.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPARISON = ROOT / "reports/video-comparison"
sys.path.insert(0, str(ROOT))
from identification import dhash, hamming  # noqa: E402

GROUP_HAMMING_MAX = 6
# Blur-rejection band and accepted-empty cases are always included in the
# labeled review; other crops are sampled per group.
INTERESTING_BAND = (7.0, 18.0)


def load_rows():
    rows = json.loads((COMPARISON / "per-request.json").read_text())
    for row in rows:
        row["abs_crop"] = COMPARISON / row["crop_path"]
    return rows


def group_rows(rows):
    """Union-find groups over dhash similarity (scale-independent)."""
    hashes = {}
    dims = {}
    for row in rows:
        image = cv2.imdecode(np.fromfile(row["abs_crop"], dtype=np.uint8), cv2.IMREAD_COLOR)
        row["crop_wh"] = [int(image.shape[1]), int(image.shape[0])]
        hashes[row["request_id"]] = dhash(
            row["abs_crop"].read_bytes())
    parent = {request_id: request_id for request_id in hashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ids = list(hashes)
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            distance = hamming(hashes[left], hashes[right])
            if distance <= GROUP_HAMMING_MAX:
                a, b = find(left), find(right)
                if a != b:
                    parent[b] = a
    groups = {}
    for request_id in ids:
        groups.setdefault(find(request_id), []).append(request_id)
    for row in rows:
        row["group"] = find(row["request_id"])
    return groups


def main():
    rows = load_rows()
    groups = group_rows(rows)
    # Stable split: sort groups by representative request id, alternate.
    ordered = sorted(groups, key=lambda group: min(groups[group]))
    split = {}
    for index, group in enumerate(ordered):
        split[group] = "tuning" if index % 2 == 0 else "validation"
    for row in rows:
        row["split"] = split[row["group"]]
        row["interesting"] = (
            (row["reason"] in {"blurry", "accepted"} and row.get("sharpness") is not None
             and INTERESTING_BAND[0] <= row["sharpness"] <= INTERESTING_BAND[1])
            or row["ocr_status"] == "done" and not (row.get("ocr_text") or "").strip()
        )
    out = {
        "schema": "quality-gate-eval-1",
        "rows": [{k: (str(v) if isinstance(v, Path) else v)
                  for k, v in row.items() if k != "abs_crop"} for row in rows],
        "groups": {group: members for group, members in groups.items()},
        "group_hamming_max": GROUP_HAMMING_MAX,
    }
    destination = Path(__file__).resolve().parent / "crops-manifest.json"
    destination.write_text(json.dumps(out, indent=1) + "\n")
    sizes = [len(members) for members in groups.values()]
    print("crops:", len(rows), "groups:", len(groups),
          "group sizes: min/median/max:", min(sizes), sorted(sizes)[len(sizes)//2], max(sizes))
    print("interesting crops:", sum(1 for r in rows if r["interesting"]))
    print("split balance:", {s: len({r["group"] for r in rows if r["split"] == s})
                             for s in ("tuning", "validation")})


if __name__ == "__main__":
    main()
