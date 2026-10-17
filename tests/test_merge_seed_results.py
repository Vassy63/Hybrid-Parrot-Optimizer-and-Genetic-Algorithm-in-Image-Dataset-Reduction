import contextlib
import csv
import io
import json
import statistics
import tempfile
import unittest
from pathlib import Path

from scripts.merge_seed_results import METHODS, MODELS, merge, read_seed


class MergeSeedResultsTest(unittest.TestCase):
    def write_seed(self, root, seed):
        folder = root / f'seed{seed}'
        folder.mkdir()
        metadata = dict(seed=seed, source_sha='test-snapshot', epochs=20,
                        batch_size=64, split_seed=42, csv='experiment_results.csv')
        (folder / 'DONE.json').write_text(json.dumps(metadata), encoding='utf-8')
        rows = []
        for method in METHODS:
            for model in MODELS:
                rows.append(dict(Method=method, Model=model, Seed=seed,
                                 Test_Acc=80 + seed / 10, Val_Acc=81, Macro_F1=80,
                                 DR_Percent=70, ARR_Percent=99, Train_Time_Sec=10,
                                 Opt_Time_Sec=1, Speedup=2, Fitness_Total=0.5))
        with (folder / metadata['csv']).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return folder

    def test_complete_merge_and_sample_std(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            for seed in range(1, 31):
                self.write_seed(root, seed)
            with contextlib.redirect_stdout(io.StringIO()):
                merge(root, root / 'merged')
            def read(name):
                with (root / 'merged' / name).open(encoding='utf-8') as handle:
                    return list(csv.DictReader(handle))
            rows = read('experiment_results_30seeds.csv')
            self.assertEqual(len(rows), 990)
            summary = read('summary_30seeds.csv')
            self.assertTrue(all(int(r['Test_Acc_count']) == 30 for r in summary))
            self.assertAlmostEqual(float(summary[0]['Test_Acc_std']), statistics.stdev([80+s/10 for s in range(1,31)]))
            pvalues = read('wilcoxon_C1_30seeds.csv')
            self.assertTrue(all(float(r['P_Two_Sided']) == 1 for r in pvalues))

    def test_missing_duplicate_and_incompatible_results_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            folder = self.write_seed(root, 1)
            with self.assertRaisesRegex(ValueError, 'Expected seeds'):
                merge(root, root / 'merged')
            with self.assertRaisesRegex(ValueError, 'Different source'):
                read_seed(folder, 'other-source')
            duplicate_root = root / 'duplicate'
            duplicate_root.mkdir()
            self.write_seed(duplicate_root, 1)
            with self.assertRaisesRegex(ValueError, 'Duplicate seed'):
                merge(root, root / 'merged')


if __name__ == '__main__':
    unittest.main()
