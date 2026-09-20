"""Validate a YOLO detection ZIP and convert it to COCO without changing its split."""

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

from PIL import Image
import yaml

from ppyoloe_checkpoint import SOURCE_CLASSES, valid_labels


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def yolo_box(line, width, height, class_count):
    values = line.split()
    if len(values) != 5:
        raise ValueError("Expected: class_id x_center y_center width height (detection boxes).")
    cls, x, y, w, h = map(float, values)
    if (not all(math.isfinite(v) for v in (cls, x, y, w, h))
            or cls != int(cls) or not 0 <= cls < class_count
            or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)):
        raise ValueError("Invalid class ID or normalized box coordinates.")
    x1, y1, x2, y2 = x-w/2, y-h/2, x+w/2, y+h/2
    # Accommodate decimal rounding at the image boundary, not malformed boxes.
    if min(x1, y1) < -1e-5 or max(x2, y2) > 1+1e-5:
        raise ValueError("Box lies outside the image.")
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(1, x2), min(1, y2)
    return int(cls), [x1*width, y1*height, (x2-x1)*width, (y2-y1)*height]


def video_group(filename):
    stem = Path(filename).stem.split(".rf.")[0]
    match = re.match(r"^(.*(?:_mp4|_mov|_avi))-\d+(?:_jpg|_png)?$", stem, re.I)
    return match.group(1) if match else None


def prepare_dataset(archive, destination):
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    digest = sha256_file(archive)
    if destination.exists():
        manifest = destination / "dataset.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text())
            if (data.get("archive_sha256") == digest and data.get("schema_version") == 1
                    and all((destination/p).is_file() and sha256_file(destination/p) == h
                            for p, h in data.get("files", {}).items()) and data.get("files")):
                return data
        raise FileExistsError(f"Dataset directory already exists or was modified: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".grocery-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as z:
            entries = {}
            if sum(i.file_size for i in z.infolist()) > 2*1024**3:
                raise ValueError("Dataset ZIP exceeds the 2 GiB expanded-size limit.")
            for info in z.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                    raise ValueError("Unsafe ZIP path.")
                if info.is_dir():
                    continue
                if info.filename in entries:
                    if z.read(info) != z.read(entries[info.filename]):
                        raise ValueError(f"Conflicting duplicate ZIP entry: {info.filename}")
                entries[info.filename] = info
            configs = [n for n in entries if PurePosixPath(n).name == "data.yaml"]
            if len(configs) != 1:
                raise ValueError("ZIP must contain one data.yaml (identical duplicate entries are accepted).")
            config_name = configs[0]
            prefix = config_name[:-len("data.yaml")]
            config = yaml.safe_load(z.read(entries[config_name]))
            labels = config.get("names")
            if isinstance(labels, dict):
                labels = [labels[i] for i in range(len(labels))]
            if not valid_labels(labels) or config.get("nc", len(labels)) != len(labels):
                raise ValueError(f"Expected unique class names and matching nc; got {labels}")
            categories = [{"id": i+1, "name": name} for i, name in enumerate(labels)]
            files, stats = {}, {}
            groups, image_hashes = defaultdict(set), defaultdict(set)
            for split in ("train", "valid", "test"):
                image_prefix = prefix + split + "/images/"
                images = sorted(n for n in entries if n.startswith(image_prefix)
                                and Path(n).suffix.lower() in {".jpg", ".jpeg", ".png"})
                if not images:
                    raise ValueError(f"Missing images in {split}/images.")
                coco = {"info": {}, "images": [], "annotations": [], "categories": categories}
                counts = Counter({name: 0 for name in labels})
                source_groups = Counter()
                negatives = 0
                expected_labels = set()
                for image_id, name in enumerate(images, 1):
                    filename = name[len(image_prefix):]
                    if "/" in filename:
                        raise ValueError("Expected flat images/ and labels/ directories in each split.")
                    raw = z.read(entries[name])
                    image = Image.open(BytesIO(raw)); image.load()
                    width, height = image.size
                    image_hashes[hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()].add(split)
                    group = video_group(filename)
                    if group:
                        groups[group].add(split); source_groups[group] += 1
                    label_name = prefix + split + "/labels/" + Path(filename).stem + ".txt"
                    expected_labels.add(label_name)
                    if label_name not in entries:
                        raise ValueError(f"Missing label file: {label_name}; use empty files for negatives.")
                    lines = [line for line in z.read(entries[label_name]).decode().splitlines() if line.strip()]
                    negatives += not lines
                    for lineno, line in enumerate(lines, 1):
                        try:
                            cls, box = yolo_box(line, width, height, len(labels))
                        except ValueError as error:
                            raise ValueError(f"{label_name}:{lineno}: {error}") from error
                        counts[labels[cls]] += 1
                        coco["annotations"].append({"id": len(coco["annotations"])+1,
                            "image_id": image_id, "category_id": cls+1, "bbox": box,
                            "area": box[2]*box[3], "iscrowd": 0})
                    relative = f"{split}/images/{filename}"
                    output = stage / relative; output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(raw)
                    files[relative] = hashlib.sha256(raw).hexdigest()
                    coco["images"].append({"id": image_id, "file_name": filename,
                                           "width": width, "height": height})
                actual_labels = {n for n in entries if n.startswith(prefix+split+"/labels/") and n.endswith(".txt")}
                if actual_labels != expected_labels:
                    raise ValueError(f"Orphan labels in {split}.")
                if split == "train" and any(v == 0 for v in counts.values()):
                    raise ValueError("Every class must have training examples.")
                annotation = f"annotations/{split}.json"
                (stage / annotation).parent.mkdir(exist_ok=True)
                (stage / annotation).write_text(json.dumps(coco, indent=2)+"\n")
                files[annotation] = sha256_file(stage / annotation)
                stats[split] = {"images": len(images), "objects": dict(counts),
                                "background_images": negatives, "video_groups": dict(source_groups)}
            shared = {g: sorted(s) for g, s in groups.items() if len(s) > 1}
            duplicate_images = sum(len(s) > 1 for s in image_hashes.values())
            warnings = []
            if shared:
                warnings.append("Frames from the same source videos occur in multiple splits; validation/test are not independent camera sessions.")
            if duplicate_images:
                warnings.append(f"{duplicate_images} identical decoded images occur across splits.")
            warnings.append("Video groups are inferred from filenames; verify split independence manually for other naming conventions.")
            result = {"schema_version": 1, "archive_sha256": digest, "labels": labels,
                      "source_class_mapping": {label: SOURCE_CLASSES.get(label) for label in labels},
                      "splits": stats, "shared_video_groups": shared,
                      "cross_split_duplicate_images": duplicate_images,
                      "warnings": warnings, "files": files}
            (stage / "dataset.json").write_text(json.dumps(result, indent=2)+"\n")
        stage.rename(destination)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_dataset(args.archive, args.output)
    print(json.dumps({k:v for k,v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
