import ast
import json
from pathlib import Path
import unittest


class CpuSmokeNotebookTest(unittest.TestCase):
    def test_cpu_profile_runs_real_pipeline_and_checks_full_export(self):
        path = Path(__file__).resolve().parents[1] / 'notebooks/kaggle_cpu_smoke_test.ipynb'
        self.assertTrue(path.exists())
        notebook = json.loads(path.read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        ast.parse(code)
        self.assertNotIn('CODE_B64', code)
        self.assertIn("os.environ['CUDA_VISIBLE_DEVICES'] = '-1'", code)
        self.assertIn("'--epochs', '1'", code)
        self.assertIn("'--num_samples', '120'", code)
        self.assertIn("'--model', 'all'", code)
        self.assertIn("'--method', 'all'", code)
        self.assertIn('testzip()', code)
        self.assertIn('CPU_SMOKE_PASS', code)
        self.assertNotIn('assert torch.cuda.is_available()', code)
        self.assertNotEqual(notebook['metadata'].get('accelerator'), 'GPU')


if __name__ == '__main__':
    unittest.main()
