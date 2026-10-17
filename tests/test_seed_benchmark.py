import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import run_poga


class SeedBenchmarkTest(unittest.TestCase):
    def test_measured_full_baseline_and_paired_training_seeds(self):
        calls = []
        rng_seeds = []

        def train(config, *loaders):
            seed = rng_seeds[-1]
            calls.append((seed, config.seed, config.model_name, config.run_dir,
                          config.subset_indices_path))
            acc = 80 + seed if config.subset_indices_path is None else 70 + seed
            return dict(test_acc=acc, best_val_acc=acc, macro_f1=acc,
                        total_time_seconds=100 if config.subset_indices_path is None else 50,
                        test_metrics=dict(macro_precision=acc, macro_recall=acc))

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            argv = ['run_poga.py', '--method', 'random', '--model', 'all',
                    '--seeds', '7', '11', '--benchmark_seeds', '--save_dir', tmp]
            with patch.object(sys, 'argv', argv), \
                 patch.object(run_poga, 'extract_and_cache_features', return_value=(np.eye(6), np.arange(6))), \
                 patch.object(run_poga, 'create_dataloaders', return_value=(None, None, None, list(range(6)))), \
                 patch.object(run_poga, 'train_single_model', side_effect=train), \
                 patch.object(run_poga, 'set_seed', side_effect=rng_seeds.append), \
                 patch.object(run_poga.GeneticAlgorithm, 'evaluate_population', return_value=np.ones(20)), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_poga.main()
            with next(Path(tmp).rglob('experiment_results.csv')).open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(calls), 12)
        self.assertEqual({c[0] for c in calls}, {7, 11})
        self.assertTrue(all(c[1] == 42 for c in calls), 'Train/val split must stay fixed')
        self.assertEqual(len({c[3] for c in calls}), 12, 'Each model needs its own logs')
        for row in rows:
            seed = int(row['Seed'])
            if row['Method'] == 'Full Dataset (B0)':
                self.assertEqual(float(row['Test_Acc']), 80 + seed)
            else:
                self.assertAlmostEqual(float(row['ARR_Percent']), round((70 + seed) / (80 + seed) * 100, 2))
                self.assertEqual(float(row['Speedup']), 2.0)


if __name__ == '__main__':
    unittest.main()
