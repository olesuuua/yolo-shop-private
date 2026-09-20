# Fine-tuning PP-YOLOE+ Small on grocery classes

This pipeline starts from **the same official Objects365 checkpoint used by the
LightStore app**, or continues from a previous fine-tuned checkpoint with
`--init-manifest`. Its output classes follow the dataset's names and order.
Classes absent from the new dataset are not retained. Previous weights remain intact.

## 1. Prepare the environments

Follow [README setup](README.md#start-from-a-clean-computer) once. Training uses the existing
Python 3.11 / Paddle 3.3.1 CPU environment. No new framework or GPU is required.
Run the following commands from the repository root with the main `.venv` active.

## 2. Check the complete pipeline

```bash
python finetune_ppyoloe.py \
  --dataset "/path/to/4classes_132frames.zip" \
  --output runs/grocery4-smoke \
  --smoke
```

This validates and converts the archive, initializes the detector, performs three
real optimizer steps, evaluates it, saves weights and checks their reload. It
also verifies that trainable tensors changed and frozen tensors did not. A smoke
checkpoint is for pipeline verification, not a quality claim.

## 3. Fine-tune locally

```bash
python finetune_ppyoloe.py \
  --dataset "/path/to/4classes_132frames.zip" \
  --output runs/grocery4-finetune \
  --epochs 30 --batch-size 2 --imgsz 416 --freeze head
```

Use a new output directory for each run. Existing runs are never overwritten.
The dataset is cached under `.cache/datasets/<archive-hash>/` and revalidated
before reuse. Missing official source/weights are downloaded by
`prepare_ppyoloe.py`; valid local assets are reused offline. Downloaded weights
and training outputs are not committed to Git.

Defaults are a conservative first experiment, not optimized hyperparameters:

- CPU only, input 416×416, batch size 2, AdamW at `1e-4`, gradient clipping.
- One epoch of learning-rate warmup, followed by cosine decay.
- `--freeze head`: freeze backbone and neck; train the detection head.
- `--freeze backbone`: train neck and head; `--freeze none`: train all layers.
- Pretrained BatchNorm running statistics remain fixed for these small batches.
- Horizontal flips and mild brightness changes; box coordinates change with flips.
- Validation every epoch, early stopping after 8 epochs without improvement
  (`--patience`), selection by validation COCO mAP50–95. The initial transferred
  model is also a candidate, so `best_epoch: 0` means training did not improve it.
- `--init-manifest` continues model weights with a fresh optimizer and schedule;
  it does not resume optimizer state. Without it, initialization uses the verified
  original Objects365 weights. Keep each run directory to compare experiments.

The upstream PP-YOLOE loss and TaskAlignedAssigner are used directly. The first
four-class run transfers all 423 state tensors. Its six classification
weight/bias tensors are reduced from 365 outputs by copying these pretrained rows:

| Dataset label | Objects365 label | Original ID |
| --- | --- | ---: |
| apple | Apple | 82 |
| bottle | Bottle | 8 |
| can | Canned | 64 |
| pack of crisps | Chips | 303 |

Thus both the shared visual features and the relevant classifiers start from
trained weights. The new examples adapt broad categories to the actual products.
Every other tensor must match exactly; unexpected shapes or hashes fail loudly.

### Continue on the overhead footage and six classes

```bash
python finetune_ppyoloe.py \
  --dataset "/path/to/6classes_286frames.zip" \
  --init-manifest runs/grocery4-finetune/best.json \
  --output runs/grocery6-overhead \
  --eval-prefix more_cls_horiz \
  --epochs 40 --patience 10 --batch-size 2 --imgsz 416 --freeze head
```

The six-class dataset orders labels as `apple`, `bottle`, `can`, `chocolate bar`,
`pack of crisps`, `pack of muffins`. Classifier rows are transferred **by name**:
the already-trained crisps row moves from index 3 to index 4. New chocolate/muffin
rows retain PP-YOLOE's initial classifier values (zero convolution weights and a
1% prior-probability bias), then learn from the new annotations. Shared features,
localization and existing class rows come from the previous four-class weights.
New names are not silently equated to a different broad Objects365 category.

`run.json` and each checkpoint manifest record the parent checkpoint SHA-256 and
the initialization source for each class. Matching keys and tensor dimensions
are required. Both four-class and six-class checkpoints remain loadable.

`--eval-prefix more_cls_horiz` reports before/after validation metrics and final
test metrics separately for the two overhead clips whose filenames begin with
that prefix. COCO evaluation is restricted to those image IDs. Checkpoint
selection still uses the full validation set, preserving attention to old views.
This option is an additional diagnostic, not a new independent test set.

## Data contract and validation limits

The ZIP should contain `data.yaml` plus `train`, `valid`, and `test` directories,
each containing `images/` and `labels/`. Each text label row is YOLO detection
format: `class_id x_center y_center width height`, normalized to 0–1. Empty label
files are valid background images. Segmentation polygons are rejected.
Every visible target item should be labeled, including multiple items in a frame.

Conversion creates COCO JSON, preserving supplied splits and class order. It
checks missing/orphan labels, invalid coordinates, image decoding, duplicate ZIP
entries and exact image duplicates across splits. Source videos are inferred from
frame filenames and shared videos are reported; no images are silently reassigned.

The supplied 132-frame archive has 92 training, 27 validation and 13 test images.
All three source videos occur in all splits. These validation/test scores describe
held-out frames of familiar videos, **not generalization to a new recording**.
Bottle/can have only 31/29 training boxes; apple/chips have 87/82. Add independent
recordings of the real packing station, especially bottles/cans and confusing
background objects, before making accuracy claims. Test is evaluated only after
checkpoint selection and is never used to choose weights or stop training.

The later 286-frame archive has 200 training, 58 validation and 28 test images.
It includes the old footage and two overhead clips. All five source clips appear
in all three splits. Only one chocolate-bar instance is in the test split; its
per-class test score is especially unstable. An independent camera recording is
still needed for a reliable operational check.

Validation uses the same RGB normalization and square resize as the live app,
class-agnostic NMS at IoU 0.45, and confidence 0.001 for a COCO precision/recall
sweep. Camera tracking uses its separate stricter new-track threshold. Detection
mAP does not measure missed transfers, duplicate counting or error-free orders.

## Outputs

- `run.json`: source checksum, labels, class mapping, settings and dataset summary.
- `history.json`: baseline and each epoch's loss, timing and validation metrics.
- `best.pdparams` + `best.json`: validation-selected weights and checked manifest.
- `last.pdparams` + `last.json`: last completed epoch, even if it was worse.
- `summary.json`: final validation/test results, elapsed time, process peak RSS,
  changed/frozen tensor checks and checkpoint-reload verification.

An interrupted run may have weights from the last completed epoch, but has no
successful `summary.json`. The manifest records `purpose: smoke` or `finetune`.
The manifest and matching `.pdparams` must stay together when copied.

## 4. Run the trained detector in LightStore

To run the published six-class model without training or downloading the dataset,
follow [the release download and launch commands](README.md#published-six-class-model).
Its manifest is committed at `models/grocery6-overhead/best.json`; the weights
are downloaded separately from GitHub Releases and verified by SHA-256. This
manifest can also be passed to `--init-manifest` when continuing on a new dataset.

Use a separate port while the original camera server is running:

```bash
LIGHTSTORE_MODEL=ppyoloe_custom \
LIGHTSTORE_PPYOLOE_MANIFEST="$PWD/runs/grocery6-overhead/best.json" \
python -m uvicorn app:app --host 127.0.0.1 --port 8002
```

The app verifies the checkpoint checksum, model structure and exact label order.
The manifest's input size is the default; `LIGHTSTORE_IMGSZ` can override it for
an explicit speed/accuracy comparison. The same ByteTrack and transfer-counting
pipeline is used with the checkpoint's labels. Opening the training pipeline does
not restart or switch the existing camera app. To return to the original model,
launch with `LIGHTSTORE_MODEL=ppyoloe_objects365`.

Model and fine-tuning background:
[official PaddleDetection customization guide](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/docs/advanced_tutorials/customization/detection_en.md).
