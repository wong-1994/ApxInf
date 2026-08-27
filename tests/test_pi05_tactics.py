import pathlib
import tempfile
import unittest
from unittest import mock

from scripts import _pi05_tactics


class Pi05TacticSelectionTest(unittest.TestCase):
    def test_selects_validated_tactics(self):
        cases = [
            (87, "bf16", "orin_sm87_bf16_v2_v3_h10_tactics.json"),
            (89, "bf16", "rtx4090_sm89_bf16_v2_v3_h10_tactics.json"),
            (101, "fp8", "thor_u_cutlass_tactics.json"),
            (110, "bf16", "thor_sm110_bf16_v2_v3_h10_tactics.json"),
            (110, "fp8", "thor_sm110_fp8_native_v2_v3_h10_tactics.json"),
        ]
        for sm, precision, filename in cases:
            with self.subTest(sm=sm, precision=precision), tempfile.TemporaryDirectory() as root:
                root = pathlib.Path(root)
                path = root / "configs" / "pi05" / filename
                path.parent.mkdir(parents=True)
                path.touch()
                with mock.patch.object(_pi05_tactics, "cuda_sm", return_value=sm):
                    selected = _pi05_tactics.select_pi05_tactics("cuda:0", precision, root)
                self.assertEqual(selected, path)

    def test_int8_has_no_persisted_tactics(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(
            _pi05_tactics, "cuda_sm", return_value=87
        ):
            self.assertIsNone(
                _pi05_tactics.select_pi05_tactics("cuda:0", "int8", pathlib.Path(root))
            )

    def test_hidden_override_takes_precedence(self):
        override = pathlib.Path("custom.json")
        self.assertEqual(
            _pi05_tactics.select_pi05_tactics(
                "not-a-cuda-device", "bf16", pathlib.Path("."), override=override
            ),
            override,
        )


if __name__ == "__main__":
    unittest.main()
