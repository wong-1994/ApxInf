import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MinimalVlaExportTest(unittest.TestCase):
    def test_export_is_deterministic_and_complete(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for output in (first, second):
                subprocess.run([sys.executable, str(ROOT / "scripts/export_minimal_vla.py"), output], check=True)
            one = Path(first)
            two = Path(second)
            self.assertEqual((one / "model.safetensors").read_bytes(), (two / "model.safetensors").read_bytes())
            manifest = json.loads((one / "export-manifest.json").read_text())
            self.assertEqual(manifest["requested_tuples"], [{"target": "nvidia-thor", "precision": "bf16"}])
            self.assertEqual(manifest["parameter_count"], 3)
            self.assertEqual(len(manifest["parameters"]), 3)
            self.assertTrue(all(set(item) == {"parameter", "dtype", "shape", "transformation", "sha256"} for item in manifest["parameters"]))
            self.assertEqual(manifest["weights_sha256"], hashlib.sha256((one / "model.safetensors").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
