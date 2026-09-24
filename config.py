"""MVP settings. ROI coordinates refer to the resized, processed image."""

import os
import json
from pathlib import Path

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Keep CPU as the upstream default; run_local.py opts into the Paddle GPU worker.
INFERENCE_DEVICE = os.environ.get("LIGHTSTORE_DEVICE", "cpu")
if INFERENCE_DEVICE not in {"cpu", "gpu:0"}:
    raise ValueError("LIGHTSTORE_DEVICE must be cpu or gpu:0.")
TRACKER_CONFIG = str(Path(__file__).resolve().with_name("bytetrack-lightstore.yaml"))
# Low-confidence boxes can maintain an existing ID. New tracks require 0.50
# in the PP-YOLOE profile and 0.60 in the other profiles.
CONF_THRESHOLD = 0.10
# Suppress overlapping boxes before assigning persistent track IDs.
AGNOSTIC_NMS = True
NMS_IOU_THRESHOLD = 0.45

OPEN_IMAGES_CLASS_GROUPS = {
    "Fruit and berries": {
        "Apple", "Banana", "Orange", "Lemon", "Pear", "Peach", "Grape",
        "Grapefruit", "Mango", "Pineapple", "Pomegranate", "Strawberry",
        "Watermelon", "Cantaloupe", "Coconut", "Common fig",
    },
    "Vegetables and mushrooms": {
        "Tomato", "Cucumber", "Potato", "Carrot", "Bell pepper", "Broccoli",
        "Cabbage", "Pumpkin", "Radish", "Zucchini", "Garden Asparagus",
        "Artichoke", "Mushroom",
    },
    "Bread and bakery": {
        "Bread", "Croissant", "Bagel", "Muffin", "Cookie", "Pretzel",
        "Waffle", "Pancake", "Cake", "Tart",
    },
    "Prepared food": {
        "Pizza", "Sandwich", "Hamburger", "Hot dog", "Sushi", "Pasta",
        "Salad", "Burrito", "Taco", "French fries",
    },
    "Dairy and eggs": {"Milk", "Cheese", "Cream", "Egg (Food)", "Dairy Product"},
    "Sweets and drinks": {"Candy", "Ice cream", "Popcorn", "Juice", "Tea", "Coffee"},
    "Packaging": {"Bottle", "Box", "Tin can", "Plastic bag", "Container"},
}

# Restart the server after changing the model; track IDs belong to one model/session.
MODEL_PROFILE = os.environ.get("LIGHTSTORE_MODEL", "ppyoloe_objects365")
MODEL_PLATFORM_REF = None
MODEL_PLATFORM_FILENAME = None
MODEL_HF_REPO = MODEL_HF_FILENAME = MODEL_HF_REVISION = None
PPYOLOE_MANIFEST = None
DEFAULT_INFERENCE_SIZE = 640
if MODEL_PROFILE == "ppyoloe_objects365":
    TRACKER_CONFIG = str(Path(__file__).resolve().with_name("bytetrack-ppyoloe.yaml"))
    MODEL_PATH = "weights/ppyoloe-objects365/ppyoloe_crn_s_obj365_pretrained.pdparams"
    MODEL_SHA256 = None  # The Paddle worker checks the pinned checkpoint digest.
    MODEL_LABEL = f"PP-YOLOE+ Small · Objects365 · {INFERENCE_DEVICE.upper()} · 78 grocery classes"
    catalog = json.loads(Path(__file__).with_name("objects365_classes.json").read_text())
    FOOD_CLASS_GROUPS = {name: set(labels) for name, labels in catalog["grocery_groups"].items()}
elif MODEL_PROFILE == "ppyoloe_custom":
    from ppyoloe_checkpoint import read_manifest

    if not os.environ.get("LIGHTSTORE_PPYOLOE_MANIFEST"):
        raise ValueError("Set LIGHTSTORE_PPYOLOE_MANIFEST to the fine-tuning run's best.json.")
    PPYOLOE_MANIFEST = str(Path(os.environ["LIGHTSTORE_PPYOLOE_MANIFEST"]).resolve())
    custom, checkpoint = read_manifest(PPYOLOE_MANIFEST, verify_weights=False)
    MODEL_PATH = str(checkpoint)
    MODEL_SHA256 = custom["checkpoint_sha256"]
    DEFAULT_INFERENCE_SIZE = custom["input_size"]
    TRACKER_CONFIG = str(Path(__file__).resolve().with_name("bytetrack-ppyoloe.yaml"))
    MODEL_LABEL = f"PP-YOLOE+ Small · Fine-tuned · {INFERENCE_DEVICE.upper()} · {len(custom['labels'])} grocery classes"
    FOOD_CLASS_GROUPS = {"Custom grocery": set(custom["labels"])}
elif MODEL_PROFILE in {"yoloe26n", "yoloe26x"}:
    variant = "n" if MODEL_PROFILE == "yoloe26n" else "x"
    MODEL_PATH = f"weights/yoloe-26{variant}/yoloe-26{variant}-seg.pt"
    MODEL_SHA256 = {
        "n": "1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b",
        "x": "d08d390a08f98195f7c87807839fe4ff93a5491645fef1bc3bf0700efafdd639",
    }[variant]
    YOLOE_MODEL_URL = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26{variant}-seg.pt"
    MODEL_LABEL = f"YOLOE-26{variant} · CPU · 8 food and packaging classes"
    # Stable order defines class IDs 0–7 in the prompted detector.
    YOLOE_CLASSES = (
        "food can", "apple", "water bottle", "banana", "bag of potato chips",
        "packaged cheese", "packaged sausage", "egg carton",
    )
    FOOD_CLASS_GROUPS = {"Food and packaging": set(YOLOE_CLASSES)}
elif MODEL_PROFILE == "rpc_yolo26s":
    MODEL_PATH = "weights/rpc-yolo26s/exp-2.pt"
    MODEL_PLATFORM_REF = ("xiang-zhang-2", "test", "exp-2")
    MODEL_PLATFORM_FILENAME = "exp-2.pt"
    MODEL_SHA256 = "a96961e8c216bea390d1acd03f71ecd39f7a6af0275ba9501be48dbc7080dd2c"
    MODEL_LABEL = "YOLO26s · RPC exp-2 · 10 selected food SKU classes"
    # Exact SKU labels, not broad categories: other variants stay excluded.
    # The full checkpoint catalog remains in rpc_classes.json for reference.
    FOOD_CLASS_GROUPS = {
        "Snacks and sweets": {"1_puffed_food", "13_dried_fruit", "122_chocolate", "142_candy"},
        "Packaged food": {"26_dried_food", "64_dessert", "109_canned_food"},
        "Drinks and milk": {"32_instant_drink", "71_drink", "97_milk"},
    }
elif MODEL_PROFILE == "grocery_checkout":
    MODEL_PATH = "weights/grocery-checkout/best.pt"
    MODEL_HF_REPO = "cvtechniques/GroceryCheckoutDetection"
    MODEL_HF_FILENAME = "best.pt"
    MODEL_HF_REVISION = "420cfb36aba255780835be225e920c8491c319a1"
    MODEL_SHA256 = "0fdc32318fbf2ff51d7a277f6a4be08c2672a1169ed73262edac32fb105831dd"
    MODEL_LABEL = "YOLO11n · GroceryCheckoutDetection · 17 checkout categories"
    FOOD_CLASS_GROUPS = {
        "Checkout products": {
            "alcohol", "candy", "canned_food", "chocolate", "dessert",
            "dried_food", "dried_fruit", "drink", "gum", "instant_drink",
            "instant_noodles", "milk", "personal_hygiene", "puffed_food",
            "seasoner", "stationery", "tissue",
        },
    }
elif MODEL_PROFILE == "sku110k":
    MODEL_PATH = "weights/sku110k-yolo11-s640.pt"
    MODEL_HF_REPO = "chistopat/sku110k-yolo11-object-detector"
    MODEL_HF_FILENAME = MODEL_PATH
    MODEL_HF_REVISION = "ee1b8ac34eb3b68969ffa8165e50c43457fe4e35"
    MODEL_SHA256 = "b7e86554a746c2f02ee0626bcde5b77038b84692c225a642a5a5ebc150f6718a"
    MODEL_LABEL = "YOLO11s · SKU-110K · generic product detection"
    FOOD_CLASS_GROUPS = {"Retail products": {"object"}}
elif MODEL_PROFILE == "openimages":
    MODEL_PATH = "yolov8n-oiv7.pt"
    MODEL_HF_REPO = "Blue2020Panda/YOLOv8nOIV7"
    MODEL_HF_FILENAME = MODEL_PATH
    MODEL_HF_REVISION = "6085b73632b3e7e56ab64d60b1623d38679bf120"
    MODEL_SHA256 = None
    MODEL_LABEL = "YOLOv8n · Open Images V7 · CPU · 65 food and packaging classes"
    FOOD_CLASS_GROUPS = OPEN_IMAGES_CLASS_GROUPS
else:
    raise ValueError("LIGHTSTORE_MODEL must be 'ppyoloe_objects365', 'ppyoloe_custom', 'yoloe26n', 'yoloe26x', 'rpc_yolo26s', 'grocery_checkout', 'sku110k' or 'openimages'.")

# Detector resolution is independent of display/ROI coordinates (640 x 480).
INFERENCE_SIZE = int(os.environ.get("LIGHTSTORE_IMGSZ", str(DEFAULT_INFERENCE_SIZE)))
if INFERENCE_SIZE < 128 or INFERENCE_SIZE % 32:
    raise ValueError("LIGHTSTORE_IMGSZ must be a multiple of 32, at least 128.")

# Keep the existing setting name for the detector and packing-state filter.
FOOD_CLASSES = frozenset(name for group in FOOD_CLASS_GROUPS.values() for name in group)

# (left, top, right, bottom), in pixels, with origin at the top left.
BAG_ROI = (220, 140, 420, 380)
BAG_OVERLAP_THRESHOLD = 0.30
# Dynamic whole-bag zone ("dynamic") or the legacy fixed rectangle ("fixed").
# The dynamic zone never falls back silently: without a locked bag, packing
# confirmations pause and the page says so explicitly.
BAG_ZONE_MODE = os.environ.get("LIGHTSTORE_BAG_ZONE", "dynamic")
if BAG_ZONE_MODE not in {"dynamic", "fixed"}:
    raise ValueError("LIGHTSTORE_BAG_ZONE must be 'dynamic' or 'fixed'.")
# Separate YOLOE-26n segmentation checkpoint for bag localization. The
# product profile (yoloe26n) transfers detection-only weights and has no
# mask head, so the full seg checkpoint is loaded independently here.
BAG_MODEL_PATH = "weights/yoloe-26n/yoloe-26n-seg.pt"
BAG_MODEL_SHA256 = "1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b"
BAG_PROMPTS = ("plastic bag",)
BAG_CONF_THRESHOLD = 0.10
# Minimum mask area (fraction of the 640x480 frame) to accept: kills small
# false fragments (measured 0.01-0.08) while real spread bags cover 0.19+.
BAG_MIN_FRAC = 0.08
# Startup/acquisition runs every processed frame until this many consecutive
# mutually consistent masks (pairwise IoU >= BAG_RELOCK_IOU) lock the zone.
BAG_ACQUIRE_STABLE = 3
# While stable: heartbeat re-inference cadence in processed frames
# (~6 s at the measured ~1.75 processed fps) plus a cheap motion trigger.
BAG_HEARTBEAT_FRAMES = 10
# Heartbeat mask overlapping the locked zone by >= this is ordinary jitter
# (adopt directly); below it the bag relocated (freeze + pause + relock).
BAG_ADOPT_IOU = 0.50
BAG_RELOCK_IOU = 0.70
# Mean abs grayscale diff (0-255) inside the expanded zone bbox that forces
# an immediate re-check. Calibrated on the bag video: stable pairs 3.5-5.2,
# bag-moving pairs 20-41.
BAG_MOTION_THRESHOLD = 12.0
# Consecutive heartbeat/motion misses before a locked zone stops trusting
# its outline. The zone first enters a wall-clock grace period (keeps the
# ghost outline and keeps counting); only after BAG_GRACE_PERIOD_S without
# reacquisition is it declared lost (outline removed, packing paused).
BAG_MISSES_TO_LOSE = 2
BAG_GRACE_PERIOD_S = 2.0
# Rim hysteresis for outside evidence, in 640x480 pixels: a product must
# clear the footprint by this margin before an outside observation counts.
# Calibrated on the bag video: mask-rim flicker from a reaching hand carved
# ~20 px, while genuine removals travel much farther.
BAG_RIM_MARGIN_PX = 16.0
# Implausible-mask rejection: a footprint covering more than this fraction
# of the frame can never be the tabletop bag (live failure: a 0.16-conf
# mask flooded the whole screen). Low-confidence large masks are rejected
# at a lower size bar; white-bag stable masks cover ~0.3-0.45 at 0.14+.
BAG_MAX_FOOTPRINT_FRAC = 0.70
BAG_IMPLAUSIBLE_CONF = 0.20
BAG_IMPLAUSIBLE_FRAC = 0.45
# Bag-self suppression: a product box covering this fraction of the locked
# footprint while itself lying this fraction inside is the bag, not a
# product (live blue-bag scene: false Storage box scored ~0.9/~0.95).
BAG_SELF_FOOT_FRAC = 0.55
BAG_SELF_BOX_FRAC = 0.50
# "Packing..." pending state for items hidden mid-insertion (e.g. by a
# hand): a reliably-outside track that vanishes at the bag boundary waits
# this many seconds of real time. Reappearance outside cancels; expiry (or
# reappearance inside) counts the item once. The watch resolves on product
# evidence alone, so it also completes while the bag zone itself is
# moving/lost — that is what fixes insertions that shift the bag.
PENDING_PACK_SECONDS = 2.0
MIN_OUTSIDE_FRAMES = 3
MIN_INSIDE_FRAMES = 3
TRACK_TTL_FRAMES = 60
PACKED_BANNER_FRAMES = 30
MAX_PACKED_OVERLAY_ROWS = 8
MAX_FRAME_BYTES = 2_000_000
# Identification stage: detection still runs on FRAME_WIDTH x FRAME_HEIGHT
# while OCR crops come from the higher-resolution uploaded frame.
OCR_MIN_CROP_WIDTH = 60
OCR_MIN_CROP_HEIGHT = 80
OCR_CROP_MARGIN = 0.08
# Calibrated on CPU OCR (Sep 2026): brand text still reads at sharpness ~15
# with minor noise and survives down to ~7; 60 rejected perfectly good crops.
OCR_SHARPNESS_MIN = 15.0
# Normalized-scale rescue (video-comparison eval, reports/quality-gate/):
# full-resolution variance undervalues large close labels (e.g. 758x1650
# Saint Spring 388:2:31 scores 12.37 full but 34.9 at canonical height 1000
# and OCR-recovers СВЯТОЙ ИСТОЧНИК). Tuning + held-out validation support
# the union gate full>=15 OR norm@1000>=30: +7/+4 target-brand hits, zero
# extra empty OCRs and zero extra neighbor-brand cases on both splits.
# P/D brands still never recover (curved-logo recognition limit) and very
# blurry crops stay rejected. Lifecycle/scheduling/dedup/payload/CPU untouched.
OCR_SHARPNESS_NORM_HEIGHT = 1000
OCR_SHARPNESS_NORM_MIN = 30.0
OCR_CROP_JPEG_QUALITY = 85
# A track absent this many consecutive processed frames loses its OCR
# evidence and identity: at ~1 fps on CPU this clears a removed bottle
# while tolerating brief detection flicker. ByteTrack may reuse an ID for
# a newly placed bottle, so evidence must not outlive a visible gap.
IDENT_ABSENT_FRAMES = 3
# Stage 1: browser-supplied OCR crops (detection frames stay raw JPEG).
# The backend assigns each detection frame a frame_id, returns crop
# requests in the detection response, and accepts crop JPEGs as enveloped
# binary messages (see docs/detect-crop-protocol.md). Detection upload
# resolution and JPEG quality are unchanged in this stage.
# Pending crop requests per connection/session. A track keeps at most one
# outstanding request: a newer detection must not invalidate a crop that is
# still travelling from the browser, so a valid request is left in flight
# and no duplicate is issued until it is answered or expires. Accumulated
# OCR evidence is never cleared by request lifecycle changes.
MAX_PENDING_CROP_REQUESTS = 16
MAX_OUTSTANDING_CROP_REQUESTS_PER_TRACK = 1
# Upper bound on crop requests attached to a single detection response so
# one crowded frame cannot flood the browser/uplink.
MAX_CROP_REQUESTS_PER_FRAME = 4
# Unanswered crop requests expire; late browser responses are rejected so
# old evidence cannot attach to a reused track ID. Expiry uses server
# monotonic time and frees the track's slot so OCR requests are never
# starved by an unanswered request.
CROP_REQUEST_TTL_S = 8.0
# Answered/expired/invalidated request ids are remembered (bounded) so a
# late duplicate delivery reports why it was rejected instead of a generic
# unknown_request.
MAX_CROP_TOMBSTONES = 128
# Crop JPEG responses share the detection frame size cap; larger payloads
# are rejected instead of entering the OCR queue.
MAX_CROP_RESPONSE_BYTES = 2_000_000
