import ast
import base64
import json
import contextlib
import csv
import io
import tempfile
import unittest
from unittest.mock import patch
import zlib
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
                             INCLUDE_WEIGHTS=False, MAX_SESSION_HOURS=10.5, AVAILABLE_GPU_HOURS=30,
                             WORK_DIR=work, SOURCE_SHA='test', VERSIONS={}, TRAIN_DIR='train', TEST_DIR='test',
                             read_seed=read_seed, sys=__import__('sys'), json=json)

            class Process:
                returncode = 0

                def __init__(self, command, **kwargs):
                    folder = Path(command[command.index('--save_dir')+1]) / 'run_poga_all_test'
                    folder.mkdir()
                    (folder / 'weights.pth').write_bytes(b'weights')
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
            with patch('subprocess.Popen', side_effect=AssertionError('Completed seed must not run again')), contextlib.redirect_stdout(io.StringIO()):
                exec(source, namespace)
                namespace['SEEDS_TO_RUN'] = [2]
                namespace['AVAILABLE_GPU_HOURS'] = 1
                exec(source, namespace)
            self.assertFalse((work / 'person1/seed2/DONE.json').exists())

    def test_three_self_contained_shards(self):
        root = Path(__file__).resolve().parents[1]
        all_seeds = []
        payloads = []
        for person in range(1, 4):
            path = root / 'notebooks' / f'kaggle_30seeds_person{person}.ipynb'
            self.assertTrue(path.exists(), str(path))
            notebook = json.loads(path.read_text(encoding='utf-8'))
            assignments = {}
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
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
            payload = json.loads(zlib.decompress(base64.b64decode(assignments['CODE_B64'])))
            payloads.append(payload)
            for relative, content in payload.items():
                compile(content, relative, 'exec')
            self.assertIn('--benchmark_seeds', payload['run_poga.py'])
            self.assertIn('scripts/merge_seed_results.py', payload)
        self.assertEqual(sorted(all_seeds), list(range(1, 31)))
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[1], payloads[2])


if __name__ == '__main__':
    unittest.main()
