import importlib.util
import sys
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Streamlit chưa được cài trong Python kiểm tra')
class StreamlitTests(unittest.TestCase):
    def test_initial_ui_and_slider_limit(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'src/app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        slider = next(item for item in app.slider if item.label == 'Rút gọn tối đa (%)')
        self.assertEqual(slider.max, 70)
        slider.set_value(0).run()
        self.assertFalse(app.exception)

    def test_simplified_sidebar_uses_fixed_defaults_when_starting_a_job(self):
        from streamlit.testing.v1 import AppTest
        from poga_demo.settings import PROJECT_ROOT
        app = AppTest.from_file(str(PROJECT_ROOT / 'src/app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.sidebar.text_input), 0)
        self.assertEqual(len(app.sidebar.select_slider), 0)
        self.assertFalse(any(item.label == 'Cấu hình bắt đầu' for item in app.sidebar.selectbox))
        self.assertEqual(len(app.get('file_uploader')), 0)
        with tempfile.TemporaryDirectory() as temp, \
             patch('poga_demo.jobs.JobManager.start', return_value=Path(temp)) as start:
            next(b for b in app.button if b.label == 'Chạy demo').click().run()
            self.assertFalse(app.exception)
            cfg = start.call_args.args[0]
            self.assertEqual(cfg.time_limit_minutes, 15)
            self.assertEqual(cfg.source_dir, str(PROJECT_ROOT))
            self.assertEqual(cfg.train_dir, str(PROJECT_ROOT / 'datasets/intel/seg_train/seg_train'))
            self.assertEqual(cfg.test_dir, str(PROJECT_ROOT / 'datasets/intel/seg_test/seg_test'))

    def test_archive_model_filter_updates_all_comparison_charts(self):
        from streamlit.testing.v1 import AppTest
        from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
        from poga_demo.settings import PROJECT_ROOT
        app = AppTest.from_file(str(PROJECT_ROOT / 'src/app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        for model in ['resnet18', 'mobilenet_v3', 'densenet121']:
            app.selectbox(key='archive_model').set_value(model).run()
            self.assertFalse(app.exception)
            charts = app.get('vega_lite_chart')
            self.assertEqual(len(charts), 4)
            for chart in charts:
                data = convert_arrow_bytes_to_pandas_df(chart.proto.datasets[0].data.data)
                self.assertEqual(set(data['Model']), {model})
                self.assertEqual(set(data['Count']), {30})


if __name__ == '__main__':
    unittest.main()
