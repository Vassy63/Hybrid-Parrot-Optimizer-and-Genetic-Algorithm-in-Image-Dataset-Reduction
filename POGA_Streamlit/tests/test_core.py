import csv
import io
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from poga_demo.settings import DemoSettings, minimum_retained
from poga_demo.constraints import enforce_keep_floor
from poga_demo.results import paired_comparison, read_results, summarize


class SettingsTests(unittest.TestCase):
    def test_budget_never_rounds_above_seventy_percent(self):
        for n in [120, 599, 600, 601, 11228]:
            for dr in [0, 10, 35, 69.5, 70]:
                m = minimum_retained(n, dr)
                self.assertLessEqual(100 * (1 - m / n), dr + 1e-10)

    def test_rejects_impossible_or_unsupported_settings(self):
        for change in [{'reduction_percent': 71}, {'epochs': 0}, {'population_size': 5},
                       {'model': 'unknown'}, {'num_samples': 0}, {'time_limit_minutes': 0}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                DemoSettings(**change).validate()

    def test_defaults_run_both_hybrid_phases(self):
        cfg = DemoSettings()
        cfg.validate()
        self.assertGreaterEqual(cfg.iterations, 2 * cfg.phase_size)
        self.assertEqual(cfg.seed, 42)


class ConstraintTests(unittest.TestCase):
    def test_repairs_maximum_compression_and_preserves_classes(self):
        labels = np.tile(np.arange(6), 100)
        mask = np.zeros(600, dtype=int)
        mask[:2] = 1
        fixed = enforce_keep_floor(mask, labels, 180, np.random.RandomState(42))
        self.assertGreaterEqual(int(fixed.sum()), 180)
        self.assertEqual(set(labels[fixed == 1]), set(range(6)))
        self.assertTrue(np.array_equal(mask[:3], [1, 1, 0]), 'input must not be modified')

    def test_zero_reduction_keeps_every_image(self):
        labels = np.tile(np.arange(6), 20)
        fixed = enforce_keep_floor(np.zeros(120), labels, 120, np.random.RandomState(1))
        self.assertEqual(int(fixed.sum()), 120)

    def test_repair_is_repeatable_and_rejects_invalid_masks(self):
        labels = np.tile(np.arange(6), 20)
        a = enforce_keep_floor(np.zeros(120), labels, 36, np.random.RandomState(7))
        b = enforce_keep_floor(np.zeros(120), labels, 36, np.random.RandomState(7))
        self.assertTrue(np.array_equal(a, b))
        for bad in [np.zeros(119), np.full(120, 2)]:
            with self.assertRaises(ValueError):
                enforce_keep_floor(bad, labels, 36, np.random.RandomState(7))


class ResultTests(unittest.TestCase):
    def test_paired_comparison_uses_the_measured_baseline(self):
        r = paired_comparison(90.0, 87.0, 100.0, 40.0)
        self.assertAlmostEqual(r['ARR_Percent'], 96.6666666667)
        self.assertAlmostEqual(r['Delta_Acc'], -3.0)
        self.assertEqual(r['Speedup'], 2.5)
        self.assertIsNone(paired_comparison(0, 1, 0, 0)['ARR_Percent'])

    def test_summary_uses_sample_std_and_rejects_duplicate_keys(self):
        text = 'Method,Model,Seed,Test_Acc\nS1,resnet18,1,80\nS1,resnet18,2,82\n'
        rows = read_results(io.StringIO(text))
        agg = summarize(rows)[0]
        self.assertEqual(agg['Test_Acc_mean'], 81)
        self.assertAlmostEqual(agg['Test_Acc_std'], math.sqrt(2))
        with self.assertRaises(ValueError):
            read_results(io.StringIO(text + 'S1,resnet18,1,85\n'))

    def test_invalid_numeric_results_are_not_silently_shown(self):
        with self.assertRaises(ValueError):
            read_results(io.StringIO('Method,Model,Seed,Test_Acc\nS1,resnet18,1,nan\n'))


if __name__ == '__main__':
    unittest.main()
