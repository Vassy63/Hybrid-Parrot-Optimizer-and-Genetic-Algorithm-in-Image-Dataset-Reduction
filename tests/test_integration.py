"""Kiem tra tinh tuong thich sau khi tich hop (commit 5).

Cac kiem tra nay khong yeu cau dataset lon, CUDA hay chay thuc su thuat toan.
Muc tieu: xac nhan cau hinh, import, rang buoc demo va du lieu 30 seed sau tich hop.
"""
import csv
import importlib
import importlib.util
import io
import math
import sys
import unittest
from pathlib import Path

# Them thu muc goc repo vao path de import bo loi
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Thu muc POGA_Streamlit
POGA_STREAMLIT = REPO_ROOT / 'POGA_Streamlit'
sys.path.insert(0, str(POGA_STREAMLIT / 'src'))

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


class TestCoreImport(unittest.TestCase):
    """Kiem tra import va cau hinh cua bo chay thuc nghiem."""

    @unittest.skipUnless(TORCH_AVAILABLE, "torch chua duoc cai dat tren may nay")
    def test_configs_import(self):
        """configs.config phai import duoc va co cac class chinh."""
        from configs.config import BaselineConfig, POGAConfig, GAConfig, POConfig, FitnessConfig
        cfg = POGAConfig()
        self.assertAlmostEqual(
            cfg.fitness.w_div + cfg.fitness.w_cov + cfg.fitness.w_bal + cfg.fitness.w_com, 1.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch chua duoc cai dat tren may nay")
    def test_fitness_target_ratio_default_none(self):
        """FitnessConfig.target_ratio mac dinh phai la None (khong rang buoc so anh)."""
        from configs.config import FitnessConfig
        fc = FitnessConfig()
        self.assertIsNone(fc.target_ratio)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch chua duoc cai dat tren may nay")
    def test_fitness_target_ratio_soft_constraint(self):
        """compute_compression voi target_ratio la rang buoc mem, khong bat buoc."""
        import numpy as np
        from optimization.fitness import compute_compression
        # Khong co target_ratio: khuyen khich nen toi da
        sol = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        com_no_target = compute_compression(sol)
        self.assertAlmostEqual(com_no_target, 0.8)
        # Co target_ratio = 0.2: dat diem cao nhat khi ratio == target
        com_at_target = compute_compression(sol, target_ratio=0.2)
        self.assertAlmostEqual(com_at_target, 1.0)
        # Co target_ratio: lech khoi muc tieu thi bi phat
        com_off_target = compute_compression(sol, target_ratio=0.5)
        self.assertLess(com_off_target, 1.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch chua duoc cai dat tren may nay")
    def test_run_poga_entry_point_exists(self):
        """run_poga.py phai co ham main() va cac import can thiet."""
        import run_poga
        self.assertTrue(callable(getattr(run_poga, 'main', None)))
        self.assertTrue(hasattr(run_poga, 'GeneticAlgorithm'))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch chua duoc cai dat tren may nay")
    def test_no_demo_side_effect_on_core_fitness(self):
        """Import cac module demo khong duoc thay doi ham fitness cua bo loi."""
        fitness_mod = importlib.import_module('optimization.fitness')
        original_fitness = fitness_mod.fitness
        # Import demo (settings khong chay side effect)
        from poga_demo.settings import DemoSettings  # noqa: F401
        # Ham fitness phai van la ham goc
        self.assertIs(fitness_mod.fitness, original_fitness,
                      "Import demo khong duoc monkey patch fitness cua bo loi.")

    def test_fitness_py_exists_at_core(self):
        """Bo loi phai chua optimization/fitness.py (kiem tra tep, khong can import torch)."""
        self.assertTrue((REPO_ROOT / 'optimization' / 'fitness.py').is_file())
        self.assertTrue((REPO_ROOT / 'configs' / 'config.py').is_file())
        self.assertTrue((REPO_ROOT / 'run_poga.py').is_file())
        self.assertTrue((REPO_ROOT / 'train.py').is_file())

    def test_config_py_has_target_ratio_field(self):
        """FitnessConfig trong config.py phai khai bao truong target_ratio."""
        content = (REPO_ROOT / 'configs' / 'config.py').read_text(encoding='utf-8', errors='replace')
        self.assertIn('target_ratio', content)
        self.assertIn('FitnessConfig', content)

    def test_fitness_py_soft_constraint_formula(self):
        """optimization/fitness.py phai chua cong thuc F_Com voi target_ratio."""
        content = (REPO_ROOT / 'optimization' / 'fitness.py').read_text(
            encoding='utf-8', errors='replace')
        self.assertIn('target_ratio', content)
        self.assertIn('compute_compression', content)
        self.assertIn('compute_raw_fitness', content)


class TestDemoConstraints(unittest.TestCase):
    """Kiem tra rang buoc so anh va bao toan lop trong demo."""

    def test_minimum_retained_never_exceeds_dr_limit(self):
        """minimum_retained dam bao DR thuc te <= reduction_percent."""
        from poga_demo.settings import minimum_retained
        for n in [120, 599, 600, 601, 11228]:
            for dr in [0, 10, 35, 69.5, 70]:
                m = minimum_retained(n, dr)
                actual_dr = 100 * (1 - m / n)
                self.assertLessEqual(actual_dr, dr + 1e-9,
                    f"n={n}, dr={dr}: actual_dr={actual_dr:.4f} > limit")

    def test_minimum_retained_rejects_over_70_percent(self):
        """minimum_retained phai bac bo reduction_percent > 70."""
        from poga_demo.settings import minimum_retained
        with self.assertRaises(ValueError):
            minimum_retained(600, 71)
        with self.assertRaises(ValueError):
            minimum_retained(600, 100)

    def test_enforce_keep_floor_class_preservation(self):
        """enforce_keep_floor phai bao toan it nhat mot anh moi lop."""
        import numpy as np
        from poga_demo.constraints import enforce_keep_floor
        labels = np.tile(np.arange(6), 100)
        mask = np.zeros(600, dtype=int)
        rng = np.random.RandomState(0)
        fixed = enforce_keep_floor(mask, labels, 180, rng)
        self.assertGreaterEqual(int(fixed.sum()), 180)
        selected_labels = labels[fixed == 1]
        self.assertEqual(set(selected_labels.tolist()), set(range(6)))

    def test_enforce_keep_floor_does_not_modify_input(self):
        """enforce_keep_floor khong duoc sua doi mask dau vao."""
        import numpy as np
        from poga_demo.constraints import enforce_keep_floor
        labels = np.tile(np.arange(6), 20)
        mask = np.zeros(120, dtype=int)
        original = mask.copy()
        enforce_keep_floor(mask, labels, 36, np.random.RandomState(42))
        self.assertTrue(np.array_equal(mask, original), "Mask dau vao bi sua doi.")

    def test_demo_settings_validate_rejects_high_dr(self):
        """DemoSettings.validate phai bac bo reduction_percent > 70."""
        from poga_demo.settings import DemoSettings
        with self.assertRaises(ValueError):
            DemoSettings(reduction_percent=71).validate()

    def test_demo_settings_default_is_70_percent(self):
        """Gia tri mac dinh cua reduction_percent phai la 70 (rut gon toi da)."""
        from poga_demo.settings import DemoSettings
        self.assertEqual(DemoSettings.reduction_percent, 70)


class TestResults30Seeds(unittest.TestCase):
    """Kiem tra doc va tong hop bo ket qua 30 seed."""

    def _csv_path(self, name):
        return POGA_STREAMLIT / 'assets' / 'results_30seeds' / name

    def test_experiment_results_csv_readable(self):
        """experiment_results_30seeds.csv phai doc duoc va co cac cot can thiet."""
        from poga_demo.results import read_results
        path = self._csv_path('experiment_results_30seeds.csv')
        self.assertTrue(path.is_file(), f"Khong tim thay {path}")
        rows = read_results(path)
        self.assertGreater(len(rows), 0)
        self.assertIn('Method', rows[0])
        self.assertIn('Test_Acc', rows[0])

    def test_summary_csv_has_30_seeds_per_method(self):
        """Bo ket qua 30 seed phai co it nhat 30 ban ghi cho mot phuong phap."""
        from poga_demo.results import read_results
        path = self._csv_path('experiment_results_30seeds.csv')
        rows = read_results(path)
        seeds_by_method = {}
        for r in rows:
            m = r['Method']
            seeds_by_method.setdefault(m, set()).add(r['Seed'])
        max_seeds = max(len(s) for s in seeds_by_method.values())
        self.assertGreaterEqual(max_seeds, 30,
            f"Khong co phuong phap nao co 30 seed. Max: {max_seeds}")

    def test_results_not_modified_by_integration(self):
        """Kiem tra nhanh: cac gia tri trong CSV khong chua NaN hoac inf."""
        from poga_demo.results import read_results
        path = self._csv_path('experiment_results_30seeds.csv')
        rows = read_results(path)
        numeric_fields = ['Test_Acc', 'DR_Percent']
        for row in rows:
            for field in numeric_fields:
                val = row.get(field)
                if val is not None:
                    self.assertTrue(math.isfinite(float(val)),
                        f"Gia tri {field}={val} khong hop le trong bo ket qua 30 seed.")

    def test_summarize_produces_mean_and_std(self):
        """summarize() phai tinh trung binh va do lech chuan mau."""
        from poga_demo.results import read_results, summarize
        text = 'Method,Model,Seed,Test_Acc\nC1,resnet18,1,80\nC1,resnet18,2,82\n'
        rows = read_results(io.StringIO(text))
        agg = summarize(rows)[0]
        self.assertEqual(agg['Test_Acc_mean'], 81)
        self.assertAlmostEqual(agg['Test_Acc_std'], math.sqrt(2))


class TestStartupAndLaunch(unittest.TestCase):
    """Kiem tra khoi chay ung dung va cac lenh ho tro."""

    def test_start_demo_bat_exists(self):
        """start_demo.bat phai ton tai trong POGA_Streamlit/."""
        self.assertTrue((POGA_STREAMLIT / 'start_demo.bat').is_file())

    def test_setup_demo_bat_exists(self):
        """setup_demo.bat phai ton tai trong POGA_Streamlit/."""
        self.assertTrue((POGA_STREAMLIT / 'setup_demo.bat').is_file())

    def test_requirements_txt_exists(self):
        """requirements.txt phai ton tai trong POGA_Streamlit/ va chua cac goi chinh."""
        req = POGA_STREAMLIT / 'requirements.txt'
        self.assertTrue(req.is_file())
        content = req.read_text(encoding='utf-8')
        self.assertIn('streamlit', content)
        self.assertIn('torch', content)

    def test_app_py_exists(self):
        """POGA_Streamlit/src/app.py phai ton tai."""
        self.assertTrue((POGA_STREAMLIT / 'src' / 'app.py').is_file())

    def test_settings_project_root_resolves_to_core(self):
        """PROJECT_ROOT trong settings.py phai tro vao thu muc chua bo loi."""
        from poga_demo.settings import PROJECT_ROOT
        core_files = ['configs/config.py', 'optimization/fitness.py', 'train.py']
        for f in core_files:
            self.assertTrue((PROJECT_ROOT / f).is_file(),
                f"Bo loi thieu tep: {PROJECT_ROOT / f}")

    def test_poga_root_distinct_from_project_root(self):
        """POGA_ROOT phai la thu muc POGA_Streamlit, khac CORE_ROOT."""
        from poga_demo.settings import POGA_ROOT, CORE_ROOT
        self.assertNotEqual(POGA_ROOT, CORE_ROOT)
        self.assertTrue((POGA_ROOT / 'src' / 'app.py').is_file())
        self.assertIn('POGA_Streamlit', str(POGA_ROOT))


if __name__ == '__main__':
    unittest.main()