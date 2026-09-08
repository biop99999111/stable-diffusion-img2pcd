"""No GPU/models required; verify the exact BasicSR legacy import contract."""
import importlib
import sys
import types
import unittest
from unittest.mock import patch

from backends import _patch_basicsr_torchvision


LEGACY = "torchvision.transforms.functional_tensor"
PUBLIC = "torchvision.transforms.functional"


class BasicSRCompatTests(unittest.TestCase):
    def test_missing_legacy_uses_public_function_and_is_idempotent(self):
        public = types.ModuleType(PUBLIC)
        public.rgb_to_grayscale = lambda image, num_output_channels=1: (image, num_output_channels)
        def load(name):
            if name in sys.modules:
                return sys.modules[name]
            if name == LEGACY:
                raise ModuleNotFoundError(name, name=LEGACY)
            self.assertEqual(name, PUBLIC)
            return public
        with patch.dict(sys.modules):
            sys.modules.pop(LEGACY, None)
            with patch.object(importlib, "import_module", side_effect=load):
                _patch_basicsr_torchvision()
                first = sys.modules[LEGACY]
                _patch_basicsr_torchvision()
                self.assertIs(first, sys.modules[LEGACY])
            # Exactly the import in BasicSR's degradations.py.
            from torchvision.transforms.functional_tensor import rgb_to_grayscale
            self.assertIs(rgb_to_grayscale, public.rgb_to_grayscale)
            self.assertEqual(rgb_to_grayscale("tensor", 3), ("tensor", 3))

    def test_existing_legacy_is_preserved(self):
        legacy = types.ModuleType(LEGACY)
        with patch.dict(sys.modules, {LEGACY: legacy}):
            _patch_basicsr_torchvision()
            self.assertIs(sys.modules[LEGACY], legacy)

    def test_unrelated_import_failure_is_not_masked(self):
        error = ModuleNotFoundError("torch is missing", name="torch")
        with patch.object(importlib, "import_module", side_effect=error):
            with self.assertRaises(ModuleNotFoundError) as result:
                _patch_basicsr_torchvision()
        self.assertIs(result.exception, error)


if __name__ == "__main__":
    unittest.main()
