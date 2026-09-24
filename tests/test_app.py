"""Exercise HTTP, WebSocket, JPEG and filtering without loading YOLO weights."""

import base64
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import cv2
from fastapi.testclient import TestClient
import numpy as np

import app as application
from config import (
    CONF_THRESHOLD, FOOD_CLASSES, MODEL_HF_FILENAME, MODEL_HF_REPO,
    MODEL_HF_REVISION, MODEL_PATH, MODEL_PROFILE, NMS_IOU_THRESHOLD,
    MIN_INSIDE_FRAMES, MIN_OUTSIDE_FRAMES, MODEL_PLATFORM_REF, TRACKER_CONFIG,
    INFERENCE_SIZE,
)
import vision
from vision import FrameProcessor


class Array:
    """Minimal tensor interface used by the YOLO result adapter."""
    def __init__(self, values):
        self.values = values

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    # Class zero is a valid product ID; resolve labels from model.names.
    names = {index: name for index, name in enumerate(sorted(FOOD_CLASSES))}
    names.update({1000: "Person", 1063: "Laptop"})

    def __init__(self):
        self.frames = []
        self.calls = []

    def track(self, frame, **kwargs):
        self.calls.append(kwargs)
        rows = self.frames.pop(0) if self.frames else []
        boxes = SimpleNamespace(
            xyxy=Array([r[0] for r in rows]),
            cls=Array([r[1] for r in rows]),
            id=Array([r[2] for r in rows]) if rows else None,
        )
        return [SimpleNamespace(boxes=boxes, names=self.names)]


def jpeg():
    return cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()


OUTSIDE = (80, 180, 120, 220)
INSIDE = (260, 180, 300, 220)
CLASS_IDS = {name: class_id for class_id, name in FakeModel.names.items()}
PRIMARY_CLASS = sorted(FOOD_CLASSES)[0]
PACKING_FRAMES = [OUTSIDE] * MIN_OUTSIDE_FRAMES + [INSIDE] * (MIN_INSIDE_FRAMES + 1)
PACKING_TOTALS = [0] * (MIN_OUTSIDE_FRAMES + MIN_INSIDE_FRAMES - 1) + [1, 1]


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        # These cases cover the conventional YOLO loader; YOLOE has its own tests.
        profile = patch.object(application, "MODEL_PROFILE", "rpc_yolo26s")
        profile.start()
        self.addCleanup(profile.stop)

    @unittest.skipUnless(MODEL_HF_REPO, "The selected model is not downloaded from Hugging Face.")
    def test_downloads_pinned_hf_checkpoint_when_local_file_is_missing(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory).resolve() / MODEL_PATH
            cached_path = Path(directory).resolve() / ".cache" / "huggingface" / MODEL_HF_FILENAME
            data = b"test checkpoint"
            def downloaded(**kwargs):
                cached_path.parent.mkdir(parents=True, exist_ok=True)
                cached_path.write_bytes(data)
                return str(cached_path)
            download = Mock(side_effect=downloaded)
            model_factory = Mock()
            with patch.object(application, "__file__", str(Path(directory) / "app.py")), patch.object(
                application, "MODEL_SHA256", hashlib.sha256(data).hexdigest(),
            ), patch.dict(sys.modules, {
                "huggingface_hub": SimpleNamespace(hf_hub_download=download),
                "ultralytics": SimpleNamespace(YOLO=model_factory),
            }):
                application.load_model()
            download.assert_called_once_with(
                repo_id=MODEL_HF_REPO, filename=MODEL_HF_FILENAME,
                revision=MODEL_HF_REVISION,
                cache_dir=Path(directory).resolve() / ".cache" / "huggingface", token=False,
            )
            self.assertEqual(model_path.read_bytes(), data)
            self.assertFalse(model_path.with_suffix(".pt.tmp").exists())
            model_factory.assert_called_once_with(str(model_path))

    def test_public_platform_download_validates_weights_before_installing(self):
        for scenario in ("valid", "bad_hash", "short_download"):
            with self.subTest(scenario=scenario), TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                destination = root / "weights/platform/exp-2.pt"
                data = b"checkpoint bytes"
                digest = hashlib.sha256(b"different" if scenario == "bad_hash" else data).hexdigest()
                platform_factory = MagicMock()
                platform = platform_factory.return_value.__enter__.return_value
                platform.models.files.return_value = {"files": [
                    {"name": "other.pt", "size": 999, "downloadUrl": "https://example.com/other"},
                    {"name": "exp-2.pt", "size": len(data) + (scenario == "short_download"),
                     "downloadUrl": "https://example.com/weights"},
                ]}
                response = MagicMock()
                response.__enter__.return_value = response
                response.iter_content.return_value = [data]
                model_factory = Mock()
                with patch.object(application, "__file__", str(root / "app.py")), patch.object(
                    application, "MODEL_PATH", "weights/platform/exp-2.pt",
                ), patch.object(application, "MODEL_SHA256", digest), patch.object(
                    application, "MODEL_PLATFORM_REF", ("owner", "project", "exp-2"),
                ), patch.object(application, "MODEL_PLATFORM_FILENAME", "exp-2.pt"), patch.dict(sys.modules, {
                    "ultralytics_platform": SimpleNamespace(Platform=platform_factory),
                    "ultralytics": SimpleNamespace(YOLO=model_factory),
                }), patch("requests.get", return_value=response) as download:
                    if scenario == "valid":
                        application.load_model()
                        self.assertEqual(destination.read_bytes(), data)
                        model_factory.assert_called_once_with(str(destination))
                    else:
                        with self.assertRaises(ValueError):
                            application.load_model()
                        self.assertFalse(destination.exists())
                        model_factory.assert_not_called()
                platform_factory.assert_called_once_with(api_key="", timeout=30, max_retries=1)
                platform.models.files.assert_called_once_with("owner", "project", "exp-2")
                download.assert_called_once_with("https://example.com/weights", stream=True, timeout=(15, 60))
                self.assertFalse(list(root.rglob("*.download")))

    def test_existing_weights_load_without_network(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory).resolve() / MODEL_PATH
            model_path.parent.mkdir(parents=True, exist_ok=True)
            data = b"test checkpoint"
            model_path.write_bytes(data)
            download = Mock(side_effect=AssertionError("Unexpected network access"))
            model_factory = Mock()
            with patch.object(application, "__file__", str(Path(directory) / "app.py")), patch.object(
                application, "MODEL_SHA256", hashlib.sha256(data).hexdigest(),
            ), patch.dict(sys.modules, {
                "huggingface_hub": SimpleNamespace(hf_hub_download=download),
                "ultralytics": SimpleNamespace(YOLO=model_factory),
            }):
                application.load_model()
            download.assert_not_called()
            model_factory.assert_called_once_with(str(model_path))

    def test_corrupt_checkpoint_is_rejected_before_loading(self):
        with TemporaryDirectory() as directory:
            model_path = Path(directory).resolve() / MODEL_PATH
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(b"corrupted")
            model_factory = Mock()
            with patch.object(application, "__file__", str(Path(directory) / "app.py")), patch.object(
                application, "MODEL_SHA256", hashlib.sha256(b"expected").hexdigest(),
            ), patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=model_factory)}):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    application.load_model()
            model_factory.assert_not_called()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.model_patch = patch.object(application, "load_model", return_value=self.model)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        # These tests pin the legacy fixed-ROI product path: no bag localizer,
        # so packing never pauses for a missing bag. The dynamic zone path is
        # covered by tests/test_bag_zone.py (unit + full FrameProcessor run).
        self.zone_patch = patch.object(
            vision.FrameProcessor, "_build_bag_zone", return_value=None)
        self.zone_patch.start()
        self.addCleanup(self.zone_patch.stop)
        self.client = TestClient(application.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_home_and_script(self):
        self.assertIn("LightStore", self.client.get("/").text)
        session = self.client.get("/api/session").json()
        self.assertEqual(session["model_profile"], MODEL_PROFILE)
        self.assertEqual(session["inference_device"], "cpu")
        self.assertEqual(session["inference_size"], INFERENCE_SIZE)
        self.assertEqual(set(session["enabled_classes"]), FOOD_CLASSES)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)

    def test_ident_diagnostics_endpoints(self):
        readiness = self.client.get("/api/ident-readiness").json()
        self.assertTrue(readiness["catalog_ok"])
        self.assertIn("jev_key_present", readiness)
        self.assertNotIn("TYPESAFE_API_KEY", str(readiness))
        debug = self.client.get("/api/ident-debug").json()
        self.assertIn("capture", debug)
        self.assertIn("queue_depth", debug)
        self.assertIn("tracks", debug)
        self.assertEqual(self.client.get("/api/ident-crop/999").status_code, 404)

    def test_websocket_filtering_tracking_packing_and_reset(self):
        self.model.frames = [
            [(bbox, CLASS_IDS[PRIMARY_CLASS], 7), (bbox, CLASS_IDS["Person"], 1), (bbox, CLASS_IDS["Laptop"], 2)]
            for bbox in PACKING_FRAMES
        ]
        with self.client.websocket_connect("/ws/detect") as ws:
            for expected_total in PACKING_TOTALS:
                ws.send_bytes(jpeg())
                result = ws.receive_json()
                self.assertEqual(result["visible_counts"], {PRIMARY_CLASS: 1})
                self.assertEqual(result["packed_total"], expected_total)
                self.assertEqual(result["session_version"], 0)
            self.assertEqual(result["events"], [])
            self.assertEqual(result["last_event"]["track_id"], 7)
            image = cv2.imdecode(np.frombuffer(base64.b64decode(result["image"]), np.uint8), 1)
            self.assertEqual(image.shape, (480, 640, 3))
            # The fixed packing rectangle is present: off-white halo
            # beside its charcoal core on the left edge.
            self.assertGreater(int(image[250, 218, 1]), 150)
            self.assertLess(int(image[250, 220, 1]), 80)
            self.assertEqual(self.client.get("/api/session").json()["packed_counts"], {PRIMARY_CLASS: 1})
            reset = self.client.post("/api/reset").json()
            self.assertEqual(reset["packed_counts"], {})
            self.assertEqual(reset["event_count"], 0)
            self.assertIsNone(reset["last_event"])
            self.assertEqual(reset["session_version"], 1)
            ws.send_bytes(jpeg())
            self.assertEqual(ws.receive_json()["session_version"], 1)
        for call in self.model.calls:
            self.assertEqual(
                {self.model.names[class_id] for class_id in call["classes"]}, FOOD_CLASSES,
            )
            self.assertTrue(call["persist"])
            self.assertEqual(call["conf"], CONF_THRESHOLD)
            self.assertEqual(call["tracker"], TRACKER_CONFIG)
            self.assertEqual(call["device"], "cpu")
            self.assertEqual(call["imgsz"], INFERENCE_SIZE)
            self.assertTrue(call["agnostic_nms"])
            self.assertTrue(call["nms"])
            self.assertEqual(call["iou"], NMS_IOU_THRESHOLD)

    def test_all_enabled_product_and_packaging_classes_reach_websocket_counts(self):
        rows = [(name, CLASS_IDS[name]) for name in sorted(FOOD_CLASSES)]
        self.model.frames = [
            [(bbox, class_id, class_id) for _, class_id in rows]
            for bbox in PACKING_FRAMES
        ]
        with self.client.websocket_connect("/ws/detect") as ws:
            for total in [count * len(rows) for count in PACKING_TOTALS]:
                ws.send_bytes(jpeg())
                result = ws.receive_json()
                self.assertEqual(result["visible_counts"], {name: 1 for name, _ in rows})
                self.assertEqual(result["packed_total"], total)
            self.assertEqual(result["packed_counts"], {name: 1 for name, _ in rows})
            self.assertEqual(result["events"], [])

    def test_partial_class_match_fails_instead_of_silently_dropping_products(self):
        names = {key: value for key, value in FakeModel.names.items() if value != PRIMARY_CLASS}
        with self.assertRaisesRegex(ValueError, f"missing configured FOOD_CLASSES: {PRIMARY_CLASS}"):
            FrameProcessor(SimpleNamespace(names=names))

    @unittest.skipUnless(MODEL_PROFILE == "rpc_yolo26s", "RPC-specific SKU allowlist")
    def test_rpc_excluded_skus_never_reach_display_or_packing_counts(self):
        # Use the real, ordered class catalog, including unselected food variants.
        catalog = json.loads(Path(__file__).resolve().parents[1].joinpath("rpc_classes.json").read_text())
        self.model.names = dict(enumerate(catalog))
        processor = FrameProcessor(self.model)
        self.assertEqual(len(processor.food_class_ids), 10)
        excluded = {"2_puffed_food", "98_milk", "174_tissue", "164_personal_hygiene", "196_stationery"}
        class_ids = {name: index for index, name in self.model.names.items()}
        enabled = "97_milk"
        rows = [(name, class_ids[name]) for name in excluded | {enabled}]
        self.model.frames = [
            [(bbox, class_id, class_id + 1) for _, class_id in rows]
            for bbox in PACKING_FRAMES
        ]
        for total in PACKING_TOTALS:
            result = processor.process(jpeg())
            self.assertEqual(result["visible_counts"], {enabled: 1})
            self.assertEqual(result["packed_total"], total)
        self.assertEqual(result["packed_counts"], {enabled: 1})
        self.assertEqual({state.class_name for state in processor.tracker.tracks.values()}, {enabled})
        for call in self.model.calls:
            selected = {self.model.names[index] for index in call["classes"]}
            self.assertEqual(selected, FOOD_CLASSES)
            self.assertTrue(selected.isdisjoint(excluded))

    def test_ordered_socket_reset_and_idle_status(self):
        with self.client.websocket_connect("/ws/detect") as ws:
            ws.send_bytes(jpeg())
            ws.send_json({"type": "reset"})
            before = ws.receive_json()
            reset = ws.receive_json()
            self.assertEqual(reset["type"], "reset_ack")
            self.assertGreater(reset["session_version"], before["session_version"])
            ws.send_json({"type": "status", "diagnostics": True})
            status = ws.receive_json()
            self.assertEqual(status["type"], "status")
            self.assertEqual(status["session_version"], reset["session_version"])
            self.assertEqual(status["packed_counts"], {})
            ws.send_bytes(jpeg())
            self.assertEqual(ws.receive_json()["session_version"], reset["session_version"])

    def test_malformed_frame_does_not_kill_stream(self):
        with self.client.websocket_connect("/ws/detect") as ws:
            ws.send_bytes(b"not a jpeg")
            self.assertIn("error", ws.receive_json())
            ws.send_bytes(jpeg())
            self.assertEqual(ws.receive_json()["visible_counts"], {})

    def test_concurrent_camera_rejected_and_slot_released(self):
        with self.client.websocket_connect("/ws/detect") as first:
            # Receive a response before connecting the second camera.
            first.send_bytes(jpeg())
            first.receive_json()
            with self.client.websocket_connect("/ws/detect") as second:
                self.assertIn("Another camera", second.receive_json()["error"])
            first.send_bytes(jpeg())
            self.assertIn("image", first.receive_json())
        with self.client.websocket_connect("/ws/detect") as third:
            third.send_bytes(jpeg())
            self.assertIn("image", third.receive_json())

    def test_no_matching_model_classes_fails_instead_of_detecting_everything(self):
        with self.assertRaises(ValueError):
            FrameProcessor(SimpleNamespace(names={0: "Person"}))


if __name__ == "__main__":
    unittest.main()
