# LightStore

**Docker launch (Windows, macOS, Linux):** follow [DOCKER.md](DOCKER.md), then run
`docker compose up --build -d --wait` and open <http://localhost:8001/>. The image
includes the six-class trained weights and their runtime; no local Python setup
is required. Native amd64 and arm64 variants are built from the same Dockerfile.

An overhead-camera packing MVP built on the original FastAPI + browser WebSocket
prototype. PP-YOLOE+ Small Objects365 detects 78 selected grocery classes on CPU,
ByteTrack gives each visible product a persistent ID, and a
small state machine counts transfers into a fixed bag zone. Annotated
JPEG frames and counters are returned over the existing `/ws/detect` connection.

To fine-tune these weights on annotated grocery footage, see
[the grocery training pipeline](TRAINING.md). It accepts a YOLO detection ZIP,
converts the labels, transfers the current Objects365 weights, trains on CPU and
exports a checkpoint that can be selected with `LIGHTSTORE_MODEL=ppyoloe_custom`.
Use `--init-manifest` to continue from an existing fine-tuned model when adding
new views or classes; classifier rows are matched by name.

## Published six-class model

The latest fine-tuned checkpoint is available in the
[six-class overhead release](https://github.com/RuslanGreenhead/yolo_shop/releases/tag/grocery6-overhead-2026-09-17).
It detects `apple`, `bottle`, `can`, `chocolate bar`, `pack of crisps` and
`pack of muffins`, using CPU inference at 416×416. The release includes the actual
trained weights (~31 MB) and their manifest; the dataset is not required to run it.

### Start from a clean computer

Install **Git, Python 3.14 and Python 3.11** first. The setup script targets
Apple Silicon macOS 14+ and Linux x86_64 (Windows users can use x86_64 WSL2).
The clean-install procedure is verified on Apple Silicon; Linux/WSL still needs
verification on that hardware. Native Windows, Intel Macs and Raspberry Pi 3
are not supported by this pinned runtime. On Debian/Ubuntu, OpenCV may also need
the system packages `libgl1` and `libglib2.0-0`.

```bash
git clone --branch codex/lightstore-packing-mvp https://github.com/RuslanGreenhead/yolo_shop.git
cd yolo_shop
python3.14 setup_lightstore.py
python3 run_lightstore.py
```

Open <http://127.0.0.1:8001/>. No environment activation, personal paths, training
ZIPs or copied cache folders are needed. The setup creates `.venv` (Python 3.14)
and `.venv-ppyolo-export` (Python 3.11), installs pinned dependencies, restores
the exact PaddleDetection revision, downloads the released weights, verifies
SHA-256 and runs three frames through the real CPU model and HTTP/WebSocket app.
It does not download the original 365-class weights for this six-class launch.

Use `--paddle-python /path/to/python3.11` if that interpreter is not on PATH.
Add `--no-cache` to setup to bypass pip's download cache. Linux setup requests
CPU-only PyTorch wheels. Allow roughly 3 GB of disk for the two environments,
source and weights, plus installation temporary files.

Subsequent launches use `python3 run_lightstore.py`; add `--port 8002` if 8001 is
already occupied. Stop the server with Ctrl+C. The launcher selects the published
six-class checkpoint, CPU and 416×416 input even if the shell previously selected
another model. To check without starting a listening server, use
`python3 run_lightstore.py --check`. This is an installation/inference smoke test,
not a measurement of detection quality.

### What is cached

- `models/grocery6-overhead/best.pdparams`: published weights, checked by SHA-256.
- `.cache/PaddleDetection`: upstream source at the pinned commit. Local source
  changes are rejected; setup never silently depends on patches from this Mac.
- `.cache/matplotlib` and `.cache/ultralytics`: disposable framework settings.
- `.cache/datasets`: converted training data, used only during fine-tuning.

Runtime assets are restored on first setup; later launches work offline when
source and weights are present. Detection results are **not cached**. ByteTrack
and packing counts keep session state in RAM; Reset or restart clears packing
state. Keep your original training archives and `runs/` checkpoints if you need
to repeat/continue training; they are separate from disposable download caches.

See [the clean-install check](reports/clean-install.md) for the tested environment
and the dependency issue found and fixed during that check.

The fixed model, classes and thresholds reproduce the configuration. Frame rate
and detections can still differ with hardware, camera, lighting and floating-point
rounding. Training results and limitations are recorded in
[the overhead training report](reports/grocery6-overhead.md). The validation/test
frames share source videos with training, so these scores do not establish
accuracy on a new camera recording.

## Manual setup and earlier models

The pinned web environment uses Python 3.14 on macOS.
The default detector runs in a separate Python 3.11 environment with Paddle 3.3.1.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Prepare the separate CPU detector environment once.
python3.11 -m venv .venv-ppyolo-export
.venv-ppyolo-export/bin/python -m pip install -r requirements-paddle.txt
python prepare_ppyoloe.py

uvicorn app:app --host 127.0.0.1 --port 8001
```

On Windows, use WSL and the same commands above: the original frozen requirements
include `uvloop`, which is Unix-only.
The default `ppyoloe_objects365` profile loads Baidu's official Objects365 weights
from `weights/ppyoloe-objects365/ppyoloe_crn_s_obj365_pretrained.pdparams` (~39 MB).
`prepare_ppyoloe.py` downloads the pinned PaddleDetection revision, weights and label
file. Startup verifies their hashes, class order and all 423 checkpoint parameters.
The source revision and checksums are recorded in `ppyoloe_assets.py`.

Paddle 3.3.1 processes frames directly on CPU. **No ONNX export is used.**
The `.venv-ppyolo-export` directory keeps its name from the earlier export attempt;
it now runs the detector. Set `LIGHTSTORE_PADDLE_PYTHON` to use another prepared
Python executable. The worker communicates through a private local socket pair;
it does not open a network port. The main app handles NMS, ByteTrack and packing
counts. Closing the server also closes its worker. A failed worker produces an
error, with details in `.cache/ppyoloe-worker.log`, rather than silently switching
models. Weights, downloaded source and environments are ignored by Git. Normal
startup and inference reuse local assets without network access.

Open <http://127.0.0.1:8001>, allow camera access, and click **Start camera**.
Browsers require localhost or HTTPS for camera capture. Point the camera down at
the packing table and align the physical bag opening with the yellow rectangle.
Start with products **outside** the rectangle, move one inside, and wait for the
green `PACKED` label before hiding it inside an opaque bag.

- **Stop** releases the camera and preserves session counts and track state.
- **Reset session** clears counts, events, packed-ID history and per-track packing
  state. ByteTrack IDs remain continuous. Products currently inside are not
  counted again until they have been observed outside and then enter the zone.
- The MVP supports **one active camera connection and one shared packing session
  per server process**. A second camera is rejected with a clear message. Use one
  Uvicorn worker. Reset affects this shared session, including from another tab.
- A server restart or development reload clears the in-memory session. Use Reset
  before changing the camera/table or starting a new packing demonstration.

The current model is shown under the page heading. The default profile retains
78 grocery labels from the checkpoint's 365-class vocabulary. These
are category labels, not store SKU identities: `Bottle` does not identify water,
and `Egg` does not guarantee recognition of a closed egg carton. Inference
explicitly uses `device="cpu"`, even when a GPU is available. Validate recognition
on the actual camera and products before relying on packing counts.

The previous models remain available. Stop the server, then choose a profile:

```bash
LIGHTSTORE_MODEL=grocery_checkout uvicorn app:app --host 127.0.0.1 --port 8001
# Or:
LIGHTSTORE_MODEL=sku110k uvicorn app:app --host 127.0.0.1 --port 8001
# Or:
LIGHTSTORE_MODEL=openimages uvicorn app:app --host 127.0.0.1 --port 8001
```

The `rpc_yolo26s` profile uses `weights/rpc-yolo26s/exp-2.pt` (~20.6 MB), downloaded
from [Xiang Zhang's RPC model](https://platform.ultralytics.com/xiang-zhang-2/test/exp-2)
through the public Ultralytics Platform API and verified with a pinned SHA-256.
Its 200 classes are specific RPC product identities; ten food SKUs are enabled.

The `grocery_checkout` profile uses `weights/grocery-checkout/best.pt` (~5.5 MB)
from [cvtechniques/GroceryCheckoutDetection](https://huggingface.co/cvtechniques/GroceryCheckoutDetection),
pinned to `420cfb36aba255780835be225e920c8491c319a1` with a SHA-256 check. It is a
YOLO11n trained on RPC-derived data with labels collapsed into 17 categories.
Hugging Face profiles use the project-local `.cache/huggingface` download cache.

The `sku110k` profile reuses `weights/sku110k-yolo11-s640.pt` (~57 MB), or downloads
it from [chistopat/sku110k-yolo11-object-detector](https://huggingface.co/chistopat/sku110k-yolo11-object-detector),
pinned to `ee1b8ac34eb3b68969ffa8165e50c43457fe4e35` with a SHA-256 check. It detects
one generic `object` class and does not identify product categories or SKUs.

The `openimages` profile reuses `yolov8n-oiv7.pt` (~7.2 MB), or downloads it from
[Blue2020Panda/YOLOv8nOIV7](https://huggingface.co/Blue2020Panda/YOLOv8nOIV7),
pinned to revision `6085b73632b3e7e56ab64d60b1623d38679bf120`.
Use `LIGHTSTORE_MODEL=rpc_yolo26s` for the previous RPC model.
Use `LIGHTSTORE_MODEL=yoloe26x` for the previous large YOLOE model (~172 MB).
Use `LIGHTSTORE_MODEL=yoloe26n` for the previous eight-prompt nano model.
Use `LIGHTSTORE_MODEL=openimages` for the previous 65-category Open Images profile.
Use `LIGHTSTORE_MODEL=ppyoloe_objects365`, or omit the variable, for the current default.
Switching profiles requires a restart and starts a new packing session.

## CPU and Raspberry Pi 3

The detector input defaults to 640 pixels; display and bag-zone coordinates stay
at 640 × 480. `LIGHTSTORE_IMGSZ=320` tries a smaller detector input, which may miss
small or partially hidden products. It is not enabled by default.

On this M3 Pro Mac, PP-YOLOE measured 115.7 ms median per processed frame
(about 8.6 processing FPS), using three warmup frames and 12 timed repetitions of
a 640 × 480 JPEG of the packing illustration. This includes decoding, worker
communication, CPU inference, NMS, tracking, drawing and JPEG encoding; it excludes
browser capture and transport. The initial benchmark used the old 0.60 new-track
threshold. The PP-YOLOE profile now uses 0.50. Through the full WebSocket pipeline,
a real photo produced `Canned` and `Bottle` tracks, while the JPEG illustration
still produced no confirmed tracks. These are
smoke checks, not a measured camera accuracy score or proof of reliable SKU matching.

The current detector uses Paddle on the Mac, with PyTorch used by the tracker
adapter. Raspberry Pi 3 performance and memory usage have not been tested; this
Mac environment is not a verified Pi deployment recipe. Low frame rates can miss
the outside/inside observations required by the counter.

The later 30-second WebSocket benchmark measured about 8.0 FPS and 1.45 GiB of
combined process RSS on the M3 Pro. See the [resource report](reports/ppyoloe-resources.md)
for the measurement conditions and limitations. To measure your own local images
after setup, run the following on macOS (port 8002 must be free):

```bash
python benchmarks/measure_ppyoloe_resources.py --image /path/to/groceries.jpg --seconds 30 --output /tmp/lightstore-resources.json
```

Repeat `--image` to alternate between several images. This starts and stops a
separate server, leaving the camera session on port 8001 running. Test images are
not included in the repository; the benchmark measures resource use, not accuracy.

## Previous YOLOE profiles

The first launch downloads the official [YOLOE-26n weights](https://docs.ultralytics.com/models/yoloe/)
(`yoloe-26n-seg.pt`, ~11.7 MB) and MobileCLIP2 text encoder (~254 MB) from the
Ultralytics assets v8.4.0 release. Both are checked against the SHA-256 digests
published in the release metadata before loading. Files are saved under
`weights/yoloe-26n/`; the shared text encoder reuses `weights/yoloe-26x/mobileclip2_b.ts`.
The nano profile loads the pretrained weights into the matching detection-only
architecture, removing the unused segmentation mask branch. The CLIP tokenizer
dependency is pinned in `requirements.txt`.
The eight class prompts are encoded once, then cached under `.cache/yoloe-prompts/`.
The cache is bound to the exact weights, encoder, task, Ultralytics version and ordered
class list. Later starts reuse it without loading the text encoder or accessing
the network. Changing the classes rebuilds the prompt cache.
Internet access is needed for installation and the initial download; no account
token or dataset download is required. Weights and caches are ignored by Git. ByteTrack's
`lap` dependency is installed explicitly with the other requirements.

Select `LIGHTSTORE_MODEL=yoloe26n` or `LIGHTSTORE_MODEL=yoloe26x` explicitly to use
these profiles. Their text prompts are not used by the current Objects365 model.

## Configuration

All tuning lives in `config.py`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | `640`, `480` | Size after every incoming frame is resized |
| `INFERENCE_DEVICE` | `cpu` | Explicit inference device; no automatic GPU selection |
| `INFERENCE_SIZE` | `640` | Detector input size from `LIGHTSTORE_IMGSZ`; multiple of 32, at least 128; does not change ROI coordinates |
| `BAG_ROI` | `(220, 140, 420, 380)` | Rectangle `(x1, y1, x2, y2)` in processed-image pixels |
| `BAG_OVERLAP_THRESHOLD` | `0.30` | Minimum fraction of the **object's** box inside the ROI |
| `MIN_OUTSIDE_FRAMES` | `3` | Consecutive observed outside frames needed to arm a transfer |
| `MIN_INSIDE_FRAMES` | `3` | Consecutive observed inside frames needed to confirm packing |
| `TRACK_TTL_FRAMES` | `60` | Remove geometry after more than this many unseen processed frames |
| `CONF_THRESHOLD` | `0.10` | Lowest detection confidence passed to ByteTrack; new tracks require `0.50` for PP-YOLOE, `0.60` for other profiles |
| `AGNOSTIC_NMS` | `True` | Suppress overlapping detections across different classes |
| `NMS_IOU_THRESHOLD` | `0.45` | Box IoU above which the lower-confidence detection is suppressed, across labels |
| `JEV_WORKERS` | `4` | Concurrent Jev requests for different product tracks, selected by `LIGHTSTORE_JEV_WORKERS` (1–8) |
| `PACKED_BANNER_FRAMES` | `30` | How long the latest event stays on the video |
| `MAX_PACKED_OVERLAY_ROWS` | `8` | Maximum class rows on the video; the sidebar keeps all counts |
| `MODEL_PROFILE` | `ppyoloe_objects365` | Selected by `LIGHTSTORE_MODEL`; also accepts `ppyoloe_custom`, `yoloe26n`, `yoloe26x`, `rpc_yolo26s`, `grocery_checkout`, `sku110k` and `openimages` |
| `PPYOLOE_MANIFEST` | Unset | Required `LIGHTSTORE_PPYOLOE_MANIFEST` for `ppyoloe_custom`; fixes weights, ordered classes and default input size |
| `MODEL_PATH` | `weights/ppyoloe-objects365/ppyoloe_crn_s_obj365_pretrained.pdparams` | Local checkpoint, relative to `app.py` or absolute |
| `MODEL_PLATFORM_REF`, `MODEL_PLATFORM_FILENAME` | See `config.py` | Public Platform model and named checkpoint to download |
| `MODEL_HF_FILENAME`, `MODEL_HF_REPO`, `MODEL_HF_REVISION` | See `config.py` | Source for Hugging Face profiles when their local checkpoint is absent |
| `MODEL_SHA256` | See `config.py` | Required checksum for YOLOE, RPC, GroceryCheckoutDetection and SKU-110K weights |
| `TRACKER_CONFIG` | `bytetrack-ppyoloe.yaml` | Local ByteTrack configuration, resolved relative to `config.py` |

Coordinates start at the top-left corner: x increases to the right, y downward.
For the default ROI, the bag zone is 200 pixels wide and 240 pixels tall. Keep
`0 <= x1 < x2 <= FRAME_WIDTH` and `0 <= y1 < y2 <= FRAME_HEIGHT`. These coordinates
apply after resizing, independently of camera capture resolution. Adjust the
rectangle to match the real bag opening. Changing settings requires a reload.

`FOOD_CLASSES` is built from the selected profile's `FOOD_CLASS_GROUPS`. Names are
case-sensitive and class IDs are resolved from the checkpoint. Startup rejects
weights missing any enabled class.

The default PP-YOLOE profile enables **78 categories**: 76 food categories plus
`Bottle` and `Storage box`. The exact groups and full ordered 365-class vocabulary
are in [objects365_classes.json](objects365_classes.json). Examples include
`Apple`, `Banana`, `Canned`, `Chips`, `Cookies`, `Cheese`, `Sausage` and `Egg`.
The detector keeps its original 365-class head. For each candidate box we first
select the best class across all 365, then apply the grocery allowlist, so excluded
objects are not forced into a food category. Changing the allowlist does not retrain
the model or guarantee a speedup. A class such as `Chips` does not guarantee that a
closed packet or a specific brand will be recognized.

The previous `yoloe26n` and `yoloe26x` profiles use these **eight classes**, in class-ID order:

`food can`, `apple`, `water bottle`, `banana`, `bag of potato chips`,
`packaged cheese`, `packaged sausage`, `egg carton`.

Edit `YOLOE_CLASSES` in `config.py` to change the prompts, then restart. YOLOE
sets its classification head to this vocabulary before tracking starts; the same
allowlist also filters detections, annotations and packing counts. The published
segmentation checkpoint supplies pretrained detection weights. The nano profile
removes the mask branch; the large profile retains it, but packing does not use masks.

The `rpc_yolo26s` profile enables these **10 food SKU labels**:

| Group | Enabled RPC labels |
| --- | --- |
| Snacks and sweets | `1_puffed_food`, `13_dried_fruit`, `122_chocolate`, `142_candy` |
| Packaged food | `26_dried_food`, `64_dessert`, `109_canned_food` |
| Drinks and milk | `32_instant_drink`, `71_drink`, `97_milk` |

Other SKUs, including tissue, stationery and personal hygiene products, are
excluded from detections and packing counts. Edit the RPC `FOOD_CLASS_GROUPS` in
`config.py` and restart to change this selection. The complete 200-label catalog
remains in [rpc_classes.json](rpc_classes.json), checked against both the
checkpoint and the public model metadata. The weights still have 200 classes;
the inference allowlist filters outputs without retraining or guaranteeing a
significant speedup. Enabling `97_milk` does not enable the other milk SKUs.

The `grocery_checkout` profile enables all 17 checkpoint labels:
`alcohol`, `candy`, `canned_food`, `chocolate`, `dessert`, `dried_food`, `dried_fruit`,
`drink`, `gum`, `instant_drink`, `instant_noodles`, `milk`, `personal_hygiene`,
`puffed_food`, `seasoner`, `stationery`, `tissue`.
The `sku110k` profile enables only `object`.

The `openimages` profile enables these **65 categories**, including packaging:

| Group | Enabled Open Images V7 labels |
| --- | --- |
| Fruit and berries (16) | Apple, Banana, Orange, Lemon, Pear, Peach, Grape, Grapefruit, Mango, Pineapple, Pomegranate, Strawberry, Watermelon, Cantaloupe, Coconut, Common fig |
| Vegetables and mushrooms (13) | Tomato, Cucumber, Potato, Carrot, Bell pepper, Broccoli, Cabbage, Pumpkin, Radish, Zucchini, Garden Asparagus, Artichoke, Mushroom |
| Bread and bakery (10) | Bread, Croissant, Bagel, Muffin, Cookie, Pretzel, Waffle, Pancake, Cake, Tart |
| Prepared food (10) | Pizza, Sandwich, Hamburger, Hot dog, Sushi, Pasta, Salad, Burrito, Taco, French fries |
| Dairy and eggs (5) | Milk, Cheese, Cream, Egg (Food), Dairy Product |
| Sweets and drinks (6) | Candy, Ice cream, Popcorn, Juice, Tea, Coffee |
| Packaging (5) | Bottle, Box, Tin can, Plastic bag, Container |

Edit `OPEN_IMAGES_CLASS_GROUPS` to change that profile's allowed classes. See the
[full upstream class list](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/open-images-v7.yaml).

Filtering happens both at inference and before tracking-state updates/drawing;
unselected class labels are ignored. In the Open Images profile this excludes
people, furniture and phones. The SKU-110K model has no such labels and can still
produce false product detections on non-product objects. Detection
boxes without a track ID are withheld until ByteTrack confirms them.
Inference explicitly sets `nms=True`. On YOLO26 this selects the one-to-many head
so the configured class-agnostic IoU suppression runs before ByteTrack, as it does
with YOLO11. It is not the default NMS-free inference mode; platform-reported
metrics should not be treated as measured accuracy for this application mode.
The ByteTrack profiles separate finding a new product from keeping an existing
one: a new track requires confidence `0.50` in `bytetrack-ppyoloe.yaml` and `0.60`
in `bytetrack-lightstore.yaml`; existing tracks can match boxes from
`0.35` and use a second pass on scores between `0.10` and `0.35`. Lost tracks are
retained for 60 processed frames to recover brief occlusions. Association uses
geometry without score fusion, so confidence changes do not directly inflate the
matching cost. Raising `CONF_THRESHOLD` would remove recovery candidates before
ByteTrack sees them. The stricter new-track threshold reduces weak false positives
but may miss unfamiliar products. Stronger NMS may suppress heavily overlapping
real products, so keep items separated where possible.
`visible_counts` counts displayed tracked objects, including packaging. A track
keeps its first observed enabled class to reduce label flicker. Labels retain
their checkpoint spelling in the video and JSON responses.

Open Images includes overlapping meanings such as `Milk`, `Dairy Product` and
`Bottle`. Class-agnostic NMS suppresses highly overlapping boxes before ByteTrack,
keeping the highest-confidence label to reduce duplicate tracks for one object.
This is a heuristic: nearby overlapping products may be suppressed, and a larger
package enclosing a smaller visible product can still receive a separate ID.
Packaging follows exactly the same outside → inside packing rule as food.

For a custom checkpoint, place it locally and update `MODEL_PATH` and the class
groups together; update the checksum and HF source to match the new weights.

## Packing state machine

A valid box is **inside** if its center is inside/on the bag rectangle **OR** its
intersection area with the ROI is at least 30% of its own area. This is not IoU.
Zero-area, inverted or non-finite boxes are ignored.

1. **Unarmed:** a new track seen inside has no evidence of transfer. It is not
   counted, however long it stays inside.
2. **Outside / armed:** after `MIN_OUTSIDE_FRAMES` consecutive outside observations,
   the track can start a transfer. A gap or an inside observation resets an
   unfinished outside streak; a one-frame boundary jump cannot arm a new ID.
3. **Entering:** each consecutive observed inside frame advances confirmation.
   Returning outside or missing a frame clears this streak. A missing frame does
   not erase the previously observed outside position unless the track expires.
4. **Packed:** after `MIN_INSIDE_FRAMES`, emit one timestamped event for this ID,
   increment its class count and mark it packed. Staying inside, disappearing or
   exiting/re-entering with the same ID does not emit another event.

Stale track geometry is pruned. Already packed IDs are retained separately until
Reset, so pruning cannot accidentally allow a duplicate for the same ID. An
expired **unpacked** ID must be observed outside again. All counters, model
inference and reset operations are serialized to prevent concurrent updates.
Frame counts refer to processed frames, not elapsed seconds.

## API

- `GET /` — existing browser camera frontend.
- `GET /api/session` — packed counts, total, event count, latest event, session version, model profile/label and enabled classes.
- `POST /api/reset` — reset and return the cleared session.
- `WS /ws/detect` — send binary camera JPEGs, receive JSON with annotated JPEGs.

Example WebSocket response (image omitted):

```json
{
  "image": "<base64 JPEG>",
  "visible_counts": {"apple": 1},
  "counts": {"apple": 1},
  "packed_counts": {"apple": 2, "water bottle": 1},
  "packed_total": 3,
  "event_count": 3,
  "session_version": 0,
  "model_profile": "yoloe26x",
  "model_label": "YOLOE-26x · 8 food and packaging classes",
  "enabled_classes": ["food can", "apple", "water bottle", "banana", "bag of potato chips", "packaged cheese", "packaged sausage", "egg carton"],
  "last_event": {
    "track_id": 7,
    "class_name": "apple",
    "event": "packed",
    "timestamp": "2026-09-14T19:00:00+00:00",
    "frame_number": 42
  },
  "events": []
}
```

`counts` is a compatibility alias for `visible_counts`. `events` contains only
new events from this frame, including simultaneous transfers; `last_event` is
the latest session event or `null`. The full chronological log is retained as
`PackingTracker.packed_events` in memory. `session_version` increments on Reset
so the browser can ignore late responses from before that reset. The client
keeps one frame in flight to avoid building up a video-processing queue.

## Tests

The geometry/event layer uses only Python's standard library and never imports
YOLO or OpenCV:

```bash
python -m unittest discover -s tests -p 'test_tracking.py' -v
```

After installing requirements, run all Python tests. The API tests use a fake
model and real JPEG encoding/decoding; they do not download or load weights.

```bash
python -m unittest discover -s tests -v
LIGHTSTORE_MODEL=grocery_checkout python -m unittest discover -s tests -v
LIGHTSTORE_MODEL=sku110k python -m unittest discover -s tests -v
LIGHTSTORE_MODEL=openimages python -m unittest discover -s tests -v
python -m compileall -q app.py config.py tracking.py vision.py tests
```

The optional browser-controller tests use Node's built-in runner, with no npm
packages or physical camera required:

```bash
node --test tests/frontend.test.cjs
node --check static/app.js
```

For a real-camera acceptance check, move one product outside → inside, hold it for
several frames, and verify one event. Move it out and back with the same displayed
ID and verify no duplicate. Repeat with two products, a bottle or box, non-product
objects to check false detections, and Reset. Automated geometry tests cannot establish real-world detection
or tracking accuracy.

## Files

- `app.py`: Platform/HF checkpoint loading, FastAPI lifecycle, WebSocket and session/reset endpoints.
- `rpc_classes.json`: 200 SKU labels from the pinned RPC checkpoint.
- `config.py`: model profiles, grouped labels, checkpoint source/checksum, ROI and thresholds.
- `tracking.py`: dependency-free geometry, per-ID state, counts and event log.
- `vision.py`: YOLO tracking adapter, class validation/filtering, NMS, annotation and session lock.
- `static/index.html`, `static/app.js`: original camera UI, capture loop, packed
  counters and reset controls.
- `bytetrack-lightstore.yaml`: confidence thresholds, lost-track lifetime and association settings.
- `tests/`: geometry, event, API and browser-controller regression tests.

## MVP limitations

- An ROI crossing is a proxy for packing; this does not verify that an item is
  actually inside a physical bag. Brief passes through the ROI can count once
  they meet the configured confirmation length.
- The default RPC model recognizes the 200 training SKU labels. Similar-looking
  unfamiliar packaging can receive a wrong SKU label. There is no open-set
  recognition or order integration in this MVP.
- The optional GroceryCheckoutDetection model predicts broad categories from predominantly Chinese
  packaged groceries. Unseen packaging, other camera angles and fresh produce
  can cause missed or incorrect detections. The app does not match item identities
  against an order, and these categories do not identify individual SKUs.
- The optional SKU-110K model provides generic boxes with a single `object` label; it does
  not recognize individual SKUs or check a bag against an order's item identities.
  Its shelf training data differs from overhead packing footage.
- The optional 65 Open Images classes are generic categories, not store SKUs.
  Brand, size and flavour are not distinguished. A food label does not guarantee
  recognition through a closed package; packaging labels describe its shape,
  not its contents. Lighting, occlusion,
  fast motion and overlapping products can cause missed detections or ID swaps.
  A physical product that acquires a new ID can be counted again after a new
  outside → inside transition; exact-once applies to a track ID within a session.
- If a product disappears before enough inside frames are observed, it is not
  counted. No event is inferred from disappearance alone.
- This is a local, in-memory demonstration, with no authentication, persistence,
  multiple workers, order database, segmentation, hand/pose detection, scales,
  barcodes or custom training. It does not subtract items when they leave the bag.
- Follow the existing Ultralytics/PyTorch platform requirements. GPU acceleration
  is optional; achievable frame rate depends on the computer.
