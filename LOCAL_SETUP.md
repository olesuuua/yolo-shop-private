# Local GPU experiment

Source: `RuslanGreenhead/yolo_shop`, branch `codex/lightstore-packing-mvp`,
commit `a3fa921f19a939d24e86d8f220088f3bce680325`.
This folder is an independent working copy. Nothing is pushed to GitHub.

## Run

```bash
python3 run_local.py --check
python3 run_local.py
```

Open http://127.0.0.1:8001. The local launcher selects the released six-class
model and GPU 0. The web app and ByteTrack use the main `.venv`; actual detection
runs in `.venv-ppyolo-export` with CUDA-enabled Paddle 3.3.1. CPU PyTorch in the
web environment is intentional. Python 3.11 and uv are inside `.runtime`.
Set `LIGHTSTORE_DEVICE=cpu` to run the same weights on CPU instead.

Installation uses the original `requirements.txt` with the PyTorch CPU wheel
index and `requirements-paddle-gpu.txt` with Paddle's official CUDA 12.9 index:
`https://www.paddlepaddle.org.cn/packages/stable/cu129/`. The latter replaces
`paddlepaddle==3.3.1` with `paddlepaddle-gpu==3.3.1`. Model source and weights keep
their original pinned revision and checksum. Installed package snapshots are
saved under `work/` after verification.

The original camera app still uses a fixed bag rectangle and six broad classes.
OCR and Jev recognition have not yet been connected to the camera pipeline.

Verified on this machine: the six-class detector processed three generated
frames through HTTP/WebSocket on `gpu:0`; 69 regression tests ran successfully
with two optional-profile skips. The separate OCR utility read Russian and
English generated label text on `gpu:0`. These are runtime checks, not packaging
accuracy measurements. Evidence is in `.cache/runtime-check.json`,
`work/tests.log`, and `work/ocr-smoke.json`.

## API key

Create a dedicated key in https://console.typesafe.ai and paste it into `.env`:

```dotenv
TYPESAFE_API_KEY=your_key_here
LIGHTSTORE_DEVICE=gpu:0
```

The file is ignored by Git and readable only by its owner. Do not paste the key
into chat or browser JavaScript. The backend launcher loads it without printing
it. No API request is made by the current camera app.

## Product reference data

`data/local/catalog.json` contains only identification data. For each actual
product, record `sku`, `name`, `object_classes`, `brand`, `category`, `variant`,
`size`, `aliases`, and `verified_label_text`. Use an empty string or list when
text is unknown; do not add visual packaging descriptions or store copy.
Store reference photos in `data/local/products/<sku>/`.

Use store descriptions for structured identity and packaging photos for exact
visible text. Manually correct OCR before using it as reference data. Record
distinctive flavor/size/sugar words; a complete ingredient list is optional.
Use separate footage for evaluating recognition, including unknown items and
cases where the distinguishing label is hidden.

Russian is the primary language; English brand names must be retained. OCR uses
the PP-OCRv5 Cyrillic recognition model, which also supports English. Other
scripts are outside the initial prototype's tested scope. Keep Russian product
names, units and packaging text verbatim rather than replacing them with a
translation. Jev's Russian-language performance needs its own evaluation.

To extract text from a reference photo for manual review:

```bash
python3 ocr_local.py data/local/products/photo.jpg --output data/local/products/photo-ocr.json
```

This utility uses `.venv-ocr`, separately from the detector, and sends no image
or text to an external API. Model assets download on the first run and are
cached under `.cache/paddlex`. The OCR scores are not SKU probabilities.

Jev runs through an external API; the planned integration sends OCR text and
candidate product descriptions. Detection and OCR can run on this computer.
