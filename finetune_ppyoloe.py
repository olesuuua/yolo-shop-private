"""Prepare the annotated ZIP and fine-tune the current PP-YOLOE weights on CPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from prepare_grocery_dataset import prepare_dataset, sha256_file
from prepare_ppyoloe import main as prepare_weights
from ppyoloe_checkpoint import read_manifest

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="YOLO detection ZIP with named grocery classes")
    parser.add_argument("--init-manifest", type=Path, help="Continue from a previous run's best.json, matching classes by name")
    parser.add_argument("--eval-prefix", help="Also report before/after metrics for this filename prefix")
    parser.add_argument("--output", type=Path, required=True, help="New run directory; existing runs are never overwritten")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--freeze", choices=("head", "backbone", "none"), default="head",
                        help="head trains only the detection head; backbone trains neck+head; none trains all layers")
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Three optimizer steps plus validation/save/reload, not a full training run")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.imgsz < 128 or args.imgsz % 32 or not 0 < args.lr < 1 or args.patience < 1:
        parser.error("Use positive epochs/batch/patience/lr, and imgsz >=128 divisible by 32.")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Choose a new run directory: {output} already exists.")
    python = Path(os.environ.get("LIGHTSTORE_PADDLE_PYTHON", ROOT / ".venv-ppyolo-export/bin/python"))
    if not python.is_file():
        parser.error("Missing Paddle environment; follow README setup first.")
    if args.init_manifest:
        read_manifest(args.init_manifest)
    dataset = ROOT / ".cache/datasets" / sha256_file(args.dataset)[:16]
    metadata = prepare_dataset(args.dataset, dataset)
    print(json.dumps({"dataset": str(dataset), "splits": metadata["splits"],
                      "warnings": metadata["warnings"]}, indent=2), flush=True)
    prepare_weights()  # Reuse verified local assets; fetch official pinned assets if missing.
    command = [str(python), str(ROOT / "train_ppyoloe_worker.py"), "--data", str(dataset),
               "--output", str(output), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--imgsz", str(args.imgsz), "--lr", str(args.lr), "--freeze", args.freeze,
               "--seed", str(args.seed), "--patience", str(args.patience)]
    if args.smoke:
        command.append("--smoke")
    if args.init_manifest:
        command.extend(["--init-manifest", str(args.init_manifest.resolve())])
    if args.eval_prefix:
        command.extend(["--eval-prefix", args.eval_prefix])
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "VECLIB_MAXIMUM_THREADS": "4", "MPLCONFIGDIR": str(ROOT / ".cache/matplotlib"),
                "GLOG_minloglevel": "2"})
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
