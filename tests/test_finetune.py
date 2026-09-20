"""Dataset integrity, transfer learning and custom checkpoint/runtime contracts."""

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings
import zipfile

import numpy as np
from PIL import Image
import yaml

import app
from ppyoloe_assets import CHECKPOINT_SHA256, PADDLEDETECTION_COMMIT
from ppyoloe_checkpoint import file_digest, read_manifest, transfer_state, continue_state
from ppyoloe_model import load_ppyoloe
from ppyoloe_postprocess import postprocess, select_candidates
from prepare_grocery_dataset import prepare_dataset
from train_ppyoloe_worker import CocoImages, numpy_nms, training_example

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["apple", "bottle", "can", "pack of crisps"]


def archive_at(path, *, bad_label=None, missing_label=False, unsafe=False, labels=LABELS):
    with zipfile.ZipFile(path, "w") as archive:
        config = yaml.safe_dump({"names": labels, "nc": len(labels)})
        archive.writestr("data.yaml", config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive.writestr("data.yaml", config)
        if unsafe:
            archive.writestr("../outside.txt", "unsafe")
        for idx, split in enumerate(("train", "valid", "test")):
            image = BytesIO(); Image.new("RGB", (100, 50), (idx*50, 20, 30)).save(image, format="PNG")
            name = f"scene_mp4-{idx:04d}_png.rf.example"
            archive.writestr(f"{split}/images/{name}.png", image.getvalue())
            if missing_label and split == "train":
                continue
            label = "\n".join(f"{i} 0.5 0.5 0.4 0.4" for i in range(len(labels)))
            archive.writestr(f"{split}/labels/{name}.txt", bad_label if bad_label is not None else label)


def manifest_at(root):
    weights = root / "best.pdparams"; weights.write_bytes(b"test-checkpoint")
    info = {"schema_version": 1, "architecture": "ppyoloe_plus_crn_s", "labels": LABELS,
            "source_checkpoint_sha256": CHECKPOINT_SHA256, "paddledetection_commit": PADDLEDETECTION_COMMIT,
            "weights": weights.name, "checkpoint_sha256": file_digest(weights), "input_size": 416}
    manifest = root / "best.json"; manifest.write_text(json.dumps(info))
    return manifest, weights, info


class DatasetTests(unittest.TestCase):
    def test_six_class_conversion_keeps_reordered_ids_and_new_names(self):
        labels = ['apple', 'bottle', 'can', 'chocolate bar', 'pack of crisps', 'pack of muffins']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive_at(root/"data.zip", labels=labels)
            result = prepare_dataset(root/"data.zip", root/"data")
            self.assertEqual(result['labels'], labels)
            coco = json.loads((root/"data/annotations/train.json").read_text())
            self.assertEqual(coco['categories'][4], {'id':5, 'name':'pack of crisps'})
            self.assertEqual(len(coco['annotations']),6)
            self.assertEqual(len(CocoImages(root/'data','valid',prefix='scene_mp4').images),1)
            with self.assertRaises(ValueError):
                CocoImages(root/'data','valid',prefix='missing_view')

    def test_conversion_preserves_split_class_order_and_pixel_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive_at(root/"data.zip")
            result = prepare_dataset(root/"data.zip", root/"data")
            coco = json.loads((root/"data/annotations/train.json").read_text())
            self.assertEqual(result["labels"], LABELS)
            self.assertEqual([c["id"] for c in coco["categories"]], [1, 2, 3, 4])
            np.testing.assert_allclose(coco["annotations"][0]["bbox"], [30, 15, 40, 20])
            self.assertEqual(result["shared_video_groups"], {"scene_mp4": ["test", "train", "valid"]})
            self.assertEqual(prepare_dataset(root/"data.zip", root/"data"), result)
            (root/"data/annotations/train.json").write_text("changed")
            with self.assertRaises(FileExistsError):
                prepare_dataset(root/"data.zip", root/"data")

    def test_bad_labels_missing_partners_and_unsafe_archives_are_rejected(self):
        cases = [{"bad_label": "0 nan .5 .2 .2"}, {"bad_label": "4 .5 .5 .2 .2"},
                 {"bad_label": "0 .9 .5 .5 .5"}, {"bad_label": "0 .1 .2 .3 .4 .5 .6"},
                 {"missing_label": True}, {"unsafe": True}]
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); archive_at(root/"data.zip", **case)
                with self.assertRaises(ValueError):
                    prepare_dataset(root/"data.zip", root/"out")
                self.assertFalse((root/"out").exists())

    def test_training_coordinates_flip_and_empty_background(self):
        image = np.zeros((50, 100, 3), np.uint8)
        annotations = [{"bbox": [10, 5, 20, 10], "category_id": 4}]
        rng = Mock(); rng.random.return_value = 0; rng.uniform.return_value = 1
        tensor, boxes, labels = training_example(image, annotations, 320, rng)
        self.assertEqual(tensor.shape, (3, 320, 320))
        np.testing.assert_allclose(boxes, [[224, 32, 288, 96]])
        np.testing.assert_array_equal(labels, [3])
        self.assertEqual(training_example(image, [], 320)[1].shape, (0, 4))


class CheckpointTests(unittest.TestCase):
    def test_continuation_uses_names_and_leaves_new_rows_initialized(self):
        labels = ['apple', 'bottle', 'can', 'chocolate bar', 'pack of crisps', 'pack of muffins']
        source = {'backbone.weight':np.full((2,2), 7)}
        target = {'backbone.weight':np.zeros((2,2))}
        for scale in range(3):
            for suffix, tail in [('weight',(2,3,3)), ('bias',())]:
                key=f'yolo_head.pred_cls.{scale}.{suffix}'
                source[key]=np.arange(np.prod((4,*tail))).reshape((4,*tail))
                target[key]=np.full((6,*tail),-99)
        result, remapped, indices=continue_state(source,target,LABELS,labels)
        self.assertEqual(indices,[0,1,2,None,3,None])
        for key in remapped:
            np.testing.assert_array_equal(result[key][4],source[key][3])
            np.testing.assert_array_equal(result[key][3],target[key][3])
            np.testing.assert_array_equal(result[key][5],target[key][5])
        np.testing.assert_array_equal(result['backbone.weight'],source['backbone.weight'])
        with self.assertRaises(ValueError):
            continue_state(source,target,LABELS+['unexpected'],labels)
        with self.assertRaises(ValueError):
            continue_state(source,target,LABELS,labels[:-1]+['apple'])

    def test_all_classifier_rows_are_transferred_in_dataset_order(self):
        source = {"backbone.weight": np.ones((2, 2))}
        target = {"backbone.weight": np.zeros((2, 2))}
        ids = [82, 8, 64, 303]
        for scale in range(3):
            for suffix, shape in (("weight", (365, 2, 3, 3)), ("bias", (365,))):
                name = f"yolo_head.pred_cls.{scale}.{suffix}"
                source[name] = np.arange(np.prod(shape)).reshape(shape)
                target[name] = np.zeros((4, *shape[1:]))
        result, remapped = transfer_state(source, target, ids)
        self.assertEqual(len(remapped), 6)
        for name in remapped:
            np.testing.assert_array_equal(result[name], source[name][ids])
        np.testing.assert_array_equal(result["backbone.weight"], source["backbone.weight"])
        with self.assertRaises(ValueError):
            transfer_state({**source, "unexpected": np.zeros(1)}, target, ids)
        source["backbone.weight"] = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            transfer_state(source, target, ids)

    def test_manifest_rejects_corruption_and_wrong_class_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest, weights, info = manifest_at(root)
            self.assertEqual(read_manifest(manifest)[1], weights.resolve())
            weights.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_manifest(manifest)
            for change in ({"labels": LABELS[:-1]+["apple"]}, {"weights": "../else.pdparams"},
                           {"input_size": 415}, {"source_checkpoint_sha256": "bad"}):
                manifest.write_text(json.dumps({**info, **change}))
                with self.assertRaises(ValueError):
                    read_manifest(manifest, verify_weights=False)

    def test_custom_profile_routes_and_checks_worker_identity(self):
        with patch.object(app, "MODEL_PROFILE", "ppyoloe_custom"), patch("ppyoloe_model.load_ppyoloe") as load:
            self.assertIs(app.load_model(), load.return_value)
        with tempfile.TemporaryDirectory() as directory:
            manifest, _, info = manifest_at(Path(directory))
            worker = Mock(metadata={"labels": LABELS, "size": 416, "device": "cpu", "paddle_version": "3.3.1",
                                    "checkpoint_sha256": info["checkpoint_sha256"]})
            with patch("config.PPYOLOE_MANIFEST", str(manifest)), patch("config.FOOD_CLASSES", set(LABELS)), \
                    patch("config.INFERENCE_SIZE", 416), patch("ppyoloe_model.PaddleWorker", return_value=worker) as spawn:
                model = load_ppyoloe(ROOT)
                self.assertEqual(list(model.names.values()), LABELS)
                self.assertEqual(spawn.call_args.kwargs["manifest"], str(manifest))
                worker.metadata["checkpoint_sha256"] = "wrong"
                with self.assertRaises(ValueError):
                    load_ppyoloe(ROOT)
                worker.close.assert_called_once()

    def test_custom_score_shapes_and_validation_nms_match_live_pipeline(self):
        boxes = np.array([[[10,10,80,80], [12,12,82,82], [90,10,150,80]]], np.float32)
        scores = np.array([[[.9,.01,.01], [.01,.8,.01], [.01,.01,.7], [.01,.01,.01]]], np.float32)
        rows = select_candidates(boxes, scores, .1, num_classes=4)
        expected = postprocess(rows, (120,240,3), 160, list(range(4)), .1, .45, True)
        np.testing.assert_allclose(numpy_nms(rows, (120,240,3), 160), expected)
        with self.assertRaises(ValueError):
            select_candidates(boxes, scores, .1)  # default still enforces the original 365-class model


if __name__ == "__main__":
    unittest.main()
