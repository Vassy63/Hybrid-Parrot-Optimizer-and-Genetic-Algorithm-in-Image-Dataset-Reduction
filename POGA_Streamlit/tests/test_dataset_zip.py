import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

path=Path(__file__).resolve().parents[1]/'scripts/prepare_dataset.py'
spec=importlib.util.spec_from_file_location('prepare_dataset',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DatasetArchiveTests(unittest.TestCase):
    def test_excludes_unlabeled_prediction_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'intel.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('seg_train/seg_train/forest/1.jpg',b'example')
                z.writestr('seg_test/seg_test/forest/2.jpg',b'example')
                z.writestr('seg_pred/seg_pred/3.jpg',b'example')
            self.assertEqual(module.extract_intel(archive,root/'out'),2)
            self.assertFalse((root/'out/seg_pred').exists())

    def test_rejects_traversal_under_a_permitted_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);archive=root/'intel.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('seg_train/../../outside.jpg',b'example')
            with self.assertRaises(ValueError):
                module.extract_intel(archive,root/'out')
            self.assertFalse((root/'outside.jpg').exists())
