"""Check weight integrity and prompt reuse without downloading large models."""

import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import app
from yoloe_model import download_verified, prepare_prompts


class YoloeLoadingTests(unittest.TestCase):
    def test_yoloe_profile_uses_prompted_loader(self):
        for profile in ("yoloe26n", "yoloe26x"):
            with self.subTest(profile=profile), patch.object(app, "MODEL_PROFILE", profile), patch("yoloe_model.load_yoloe") as loader:
                self.assertIs(app.load_model(), loader.return_value)
                loader.assert_called_once_with(Path(app.__file__).resolve().parent)

    def test_download_checks_hash_and_reuses_local_file_offline(self):
        data = b"checkpoint"
        digest = hashlib.sha256(data).hexdigest()
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [data]
        with TemporaryDirectory() as directory, patch("requests.get", return_value=response) as get:
            path = Path(directory) / "model.pt"
            download_verified("https://example.com/weights", path, digest)
            self.assertEqual(path.read_bytes(), data)
            download_verified("https://example.com/weights", path, digest)
            get.assert_called_once()
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                download_verified("https://example.com/weights", path, digest)
            self.assertEqual(get.call_count, 1)

    def test_failed_download_never_installs_partial_weights(self):
        for error in (None, OSError("interrupted download")):
            with self.subTest(error=error), TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                response = MagicMock()
                response.__enter__.return_value = response
                response.iter_content.return_value = [b"wrong bytes"]
                if error:
                    response.iter_content.side_effect = error
                with patch("requests.get", return_value=response), self.assertRaises((ValueError, OSError)):
                    download_verified("https://example.com/weights", path, "incorrect digest")
                self.assertFalse(path.exists())
                self.assertFalse(path.with_suffix(".pt.download").exists())

    def test_prompt_cache_is_bound_to_classes_order_and_checkpoint(self):
        classes = ("food can", "apple", "water bottle", "banana", "bag of potato chips",
                   "packaged cheese", "packaged sausage", "egg carton")
        model = Mock()
        model.task = "detect"
        def set_classes(names, embeddings):
            model.names = dict(enumerate(names))
        model.set_classes.side_effect = set_classes
        model.save_prompt_embeddings.side_effect = lambda path: path.write_bytes(b"cached embeddings")
        encoder = Mock()
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {
            "ultralytics.nn.text_model": SimpleNamespace(MobileCLIPTS=encoder),
        }), patch("yoloe_model.download_verified", return_value=Path(directory) / "text.ts") as download:
            root = Path(directory)
            prepare_prompts(model, classes, root, "checkpoint-a")
            model.model.get_text_pe.assert_called_once_with(list(classes), cache_clip_model=True)
            self.assertEqual(list(model.names.values()), list(classes))
            self.assertNotIn("clip_model", vars(model.model))
            prepare_prompts(model, classes, root, "checkpoint-a")
            model.load_prompt_embeddings.assert_called_once()
            self.assertEqual(download.call_count, 1)
            # Reordering the same labels changes class IDs and must rebuild prompts.
            prepare_prompts(model, tuple(reversed(classes)), root, "checkpoint-a")
            prepare_prompts(model, classes, root, "checkpoint-b")
            self.assertEqual(download.call_count, 3)
            self.assertEqual(len(list(root.rglob("*.npz"))), 3)
            model.names = {0: "unexpected cached class"}
            with self.assertRaisesRegex(ValueError, "prompt labels"):
                prepare_prompts(model, classes, root, "checkpoint-a")


if __name__ == "__main__":
    unittest.main()
