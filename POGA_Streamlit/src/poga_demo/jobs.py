import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .settings import PROJECT_ROOT

TERMINAL = {'succeeded', 'failed', 'cancelled', 'time_limit'}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {} if default is None else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        for attempt in range(12):
            try:
                os.replace(temp, path)
                return
            except PermissionError as exc:
                # Windows readers/scanners can briefly deny atomic replacement.
                if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 11:
                    raise
                time.sleep(min(0.02 * (2 ** attempt), 0.25))
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass  # Cleanup must not hide the original write/replace error.


def environment_info():
    deps = {name: importlib.util.find_spec(name) is not None for name in
            ['streamlit','torch','torchvision','numpy','pandas','matplotlib','scipy','tqdm','PIL']}
    info = {'python': sys.executable, 'version': sys.version.split()[0], 'dependencies': deps,
            'cuda_available': False, 'gpu': None}
    if deps['torch']:
        try:
            import torch
            info.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
            if torch.cuda.is_available():
                info.update(gpu=torch.cuda.get_device_name(0), vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
        except Exception as exc:
            info['torch_error'] = str(exc)
    return info


def read_events(folder, limit=500):
    path = Path(folder) / 'events.jsonl'
    if not path.exists():
        return []
    # Avoid loading an unbounded training log into every Streamlit refresh.
    with path.open('rb') as handle:
        size = handle.seek(0, 2)
        handle.seek(max(0, size - 1024 * 1024))
        chunk = handle.read().decode('utf-8', errors='replace')
    events = []
    for line in chunk.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    return events[-limit:]


class JobManager:
    def __init__(self, output_root=None):
        self.output_root = Path(output_root or PROJECT_ROOT / 'outputs')
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._process = None
        self._folder = None

    def active(self):
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self, settings, mode='train'):
        settings.validate(paths=mode == 'train')
        if mode not in ['train', 'diagnose']:
            raise ValueError('Loại công việc không hợp lệ.')
        with self._lock:
            if self.active():
                raise ValueError('Một phiên đang chạy. Hãy chờ hoặc dừng phiên hiện tại.')
            name = datetime.datetime.now().strftime('demo_%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]
            folder = self.output_root / name
            folder.mkdir()
            write_json(folder / 'config.json', {'mode': mode, 'settings': settings.to_dict()})
            write_json(folder / 'status.json', {'state': 'starting', 'message': 'Đang khởi động', 'elapsed': 0, 'started_at': time.time()})
            command = [sys.executable, '-u', '-B', '-m', 'poga_demo.worker', '--job', str(folder)]
            env = dict(os.environ)
            env['PYTHONPATH'] = str(PROJECT_ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
            env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            env['TORCH_HOME'] = str(PROJECT_ROOT / 'cache' / 'torch')
            env['MPLCONFIGDIR'] = str(PROJECT_ROOT / 'cache' / 'matplotlib')
            try:
                with (folder / 'console.log').open('w', encoding='utf-8') as log:
                    self._process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log,
                        stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except OSError as exc:
                write_json(folder / 'status.json', {'state':'failed', 'message':str(exc), 'elapsed':0})
                raise
            self._folder = folder
            threading.Thread(target=self._watch, args=(self._process, folder), daemon=True).start()
            return folder

    def _watch(self, process, folder):
        exit_code = process.wait()
        with self._lock:
            state = read_json(folder / 'status.json')
            if state.get('state') not in TERMINAL:
                state.update(state='failed', message=f'Tiến trình kết thúc với mã {exit_code}. Xem console.log.', exit_code=exit_code)
                write_json(folder / 'status.json', state)

    def cancel(self):
        with self._lock:
            if not self.active():
                return False
            self._process.terminate()
            self._process.wait(timeout=10)
            state = read_json(self._folder / 'status.json')
            state.update(state='cancelled', message='Đã dừng theo yêu cầu. Kết quả chưa hoàn tất.')
            write_json(self._folder / 'status.json', state)
            return True

    def folders(self):
        return sorted(self.output_root.glob('demo_*'), key=lambda p: p.name, reverse=True)

