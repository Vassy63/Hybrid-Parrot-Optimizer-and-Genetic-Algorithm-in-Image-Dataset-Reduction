import ast
import json
import contextlib
import csv
import io
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path


class DistributedNotebooksTest(unittest.TestCase):
    def test_run_cell_archives_resumes_and_obeys_budget(self):
        from scripts.merge_seed_results import METHODS, MODELS, read_seed
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'notebooks/kaggle_30seeds_person1.ipynb').read_text(encoding='utf-8'))
        source = ''.join(notebook['cells'][-1]['source'])
        # Redirect only filesystem roots; run the actual notebook orchestration.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            work = Path(tmp).resolve()
            source = source.replace('/kaggle/working', work.as_posix()).replace('/kaggle/input', (work/'input').as_posix())
            namespace = dict(Path=Path, PERSON=1, SEEDS=list(range(1,11)), SEEDS_TO_RUN=[1],
                             MAX_SESSION_HOURS=10.5, AVAILABLE_GPU_HOURS=30,
                             WORK_DIR=work, SOURCE_SHA='test', VERSIONS={}, TRAIN_DIR='train', TEST_DIR='test',
                             read_seed=read_seed, sys=__import__('sys'), json=json)

            class Process:
                returncode = 0

                def __init__(self, command, **kwargs):
                    if command[command.index('--epochs')+1] != '20':
                        raise AssertionError('All benchmark models must use 20 epochs')
                    folder = Path(command[command.index('--save_dir')+1]) / 'run_poga_all_test'
                    folder.mkdir()
                    for method in range(11):
                        for model in MODELS:
                            weights = folder / 'model_runs' / f'method{method}_seed1' / model / 'weights' / f'{model}_best.pth'
                            weights.parent.mkdir(parents=True)
                            weights.write_bytes(b'weights')
                    with (folder / 'experiment_results.csv').open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=['Method', 'Model', 'Seed', 'Test_Acc', 'Macro_F1', 'Train_Time_Sec', 'ARR_Percent', 'Speedup'])
                        writer.writeheader()
                        for method in METHODS:
                            for model in MODELS:
                                writer.writerow(dict(Method=method, Model=model, Seed=1, Test_Acc=80,
                                                     Macro_F1=80, Train_Time_Sec=10, ARR_Percent=99, Speedup=2))

                def poll(self):
                    return 0

            with patch('subprocess.Popen', Process), contextlib.redirect_stdout(io.StringIO()):
                exec(source, namespace)
            self.assertTrue((work / 'person1_seed1.zip').exists())
            with zipfile.ZipFile(work / 'person1_seed1.zip') as archive:
                self.assertEqual(sum(name.endswith('.pth') for name in archive.namelist()), 33)
            with patch('subprocess.Popen', side_effect=AssertionError('Completed seed must not run again')), contextlib.redirect_stdout(io.StringIO()):
                exec(source, namespace)
                namespace['SEEDS_TO_RUN'] = [2]
                namespace['AVAILABLE_GPU_HOURS'] = 1
                exec(source, namespace)
            self.assertFalse((work / 'person1/seed2/DONE.json').exists())

    def test_three_shards_use_shared_input_zip_without_embedded_code(self):
        root = Path(__file__).resolve().parents[1]
        all_seeds = []
        setup_cells = []
        for person in range(1, 4):
            path = root / 'notebooks' / f'kaggle_30seeds_person{person}.ipynb'
            self.assertTrue(path.exists(), str(path))
            notebook = json.loads(path.read_text(encoding='utf-8'))
            assignments = {}
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                self.assertNotIn('CODE_B64', source)
                tree = ast.parse(source)
                compile(tree, str(path), 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                for node in tree.body:
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                        try:
                            assignments[node.targets[0].id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
            seeds = assignments['SEEDS']
            self.assertEqual(len(seeds), 10)
            all_seeds.extend(seeds)
            setup_cells.append(''.join(notebook['cells'][2]['source']))
            self.assertEqual(assignments['SOURCE_ZIP'], '')
            config = {}
            exec(''.join(notebook['cells'][1]['source']), config)
            self.assertEqual(config['SEEDS_TO_RUN'], seeds[:5])
        self.assertEqual(sorted(all_seeds), list(range(1, 31)))
        self.assertEqual(setup_cells[0], setup_cells[1])
        self.assertEqual(setup_cells[1], setup_cells[2])
        with zipfile.ZipFile(root / 'notebooks/KhoaLuanDatasetReduction_30seeds_source.zip') as archive:
            self.assertIn('scripts/merge_seed_results.py', archive.namelist())
            for name in archive.namelist():
                self.assertTrue(name.endswith('.py'))
                compile(archive.read(name), name, 'exec')

    def test_setup_loads_uploaded_zip_and_rejects_unsafe_archive(self):
        import os
        import shutil
        import sys
        from types import SimpleNamespace
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'notebooks/kaggle_30seeds_person1.ipynb').read_text(encoding='utf-8'))
        source = ''.join(notebook['cells'][2]['source'])
        fake_torch = SimpleNamespace(__version__='test', cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: 'test-GPU'))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            work = Path(tmp).resolve()
            inputs = work / 'input'
            inputs.mkdir()
            shutil.copy(root / 'notebooks/KhoaLuanDatasetReduction_30seeds_source.zip', inputs)
            for split in ('seg_train', 'seg_test'):
                for name in ('buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'):
                    (inputs / 'intel' / split / split / name).mkdir(parents=True)
            source = source.replace('/kaggle/input', inputs.as_posix()).replace('/kaggle/temp', (work/'runtime').as_posix())
            namespace = {'SOURCE_ZIP': ''}
            original_cwd = os.getcwd()
            try:
                with patch.dict(sys.modules, torch=fake_torch), patch.object(sys, 'path', sys.path[:]), patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                    exec(source, namespace)
                    self.assertTrue((namespace['WORK_DIR'] / 'run_poga.py').is_file())
                    self.assertEqual(Path(namespace['TRAIN_DIR']).parts[-2:], ('seg_train', 'seg_train'))
                    malicious = inputs / 'unsafe.zip'
                    with zipfile.ZipFile(malicious, 'w') as archive:
                        archive.writestr('../../escape.py', 'bad')
                    namespace['SOURCE_ZIP'] = str(malicious)
                    with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
                        exec(source, namespace)
            finally:
                os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
