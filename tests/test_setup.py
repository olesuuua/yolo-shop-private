"""Reproducible setup must reject hidden source changes and inherited profiles."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import prepare_ppyoloe
import run_lightstore


class SetupTests(unittest.TestCase):
    def test_pinned_source_reuses_offline_and_rejects_local_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / ".cache/PaddleDetection"
            source.mkdir(parents=True)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
            git("init", "--quiet")
            (source / "model.py").write_text("original\n")
            git("add", "model.py")
            git("-c", "user.name=Setup Test", "-c", "user.email=test@example.invalid",
                "commit", "--quiet", "-m", "fixture")
            with patch.object(prepare_ppyoloe, "PADDLEDETECTION_COMMIT", git("rev-parse", "HEAD")):
                self.assertEqual(prepare_ppyoloe.prepare_source(root), source)
                (source / "model.py").write_text("local change\n")
                with self.assertRaisesRegex(ValueError, "local modifications"):
                    prepare_ppyoloe.prepare_source(root)
                self.assertEqual((source / "model.py").read_text(), "local change\n")

    def test_launcher_pins_profile_paths_and_disables_implicit_install(self):
        with patch.dict(os.environ, {"LIGHTSTORE_MODEL": "openimages", "LIGHTSTORE_IMGSZ": "640",
                                     "PYTHONPATH": "/different/project", "PYTHONHOME": "/different/python"}):
            environment = run_lightstore.runtime_environment()
        self.assertEqual(environment["LIGHTSTORE_MODEL"], "ppyoloe_custom")
        self.assertEqual(environment["LIGHTSTORE_IMGSZ"], "416")
        self.assertEqual(Path(environment["LIGHTSTORE_PPYOLOE_MANIFEST"]), run_lightstore.MANIFEST)
        self.assertEqual(environment["YOLO_AUTOINSTALL"], "false")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)


if __name__ == "__main__":
    unittest.main()
