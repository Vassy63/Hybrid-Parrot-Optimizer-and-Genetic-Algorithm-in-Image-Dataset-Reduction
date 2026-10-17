"""Regenerate the three self-contained Kaggle notebooks from this checkout."""
import base64
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
files = ['__init__.py', 'train.py', 'run_poga.py', 'scripts/merge_seed_results.py']
for package in ('configs', 'data', 'models', 'optimization', 'training', 'utils'):
    files.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / package).glob('*.py')))
payload = {name: (ROOT / name).read_text(encoding='utf-8') for name in files}
encoded = base64.b64encode(zlib.compress(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8'))).decode('ascii')

SETUP = '''import os, sys, json, base64, zlib, hashlib, platform
from pathlib import Path
import torch

assert torch.cuda.is_available(), "Bật GPU trong Notebook options trước khi chạy."
WORK_DIR = Path('/kaggle/working/KhoaLuanDatasetReduction')
WORK_DIR.mkdir(parents=True, exist_ok=True)
CODE_B64 = __PAYLOAD__
source_bytes = zlib.decompress(base64.b64decode(CODE_B64))
SOURCE_SHA = hashlib.sha256(source_bytes).hexdigest()
for name, content in json.loads(source_bytes).items():
    destination = WORK_DIR / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding='utf-8')
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))
os.environ['PYTHONPATH'] = str(WORK_DIR)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from scripts.merge_seed_results import read_seed, METHODS, MODELS

matches = {'seg_train': [], 'seg_test': []}
classes = {'buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'}
for root, dirs, _ in os.walk('/kaggle/input'):
    if classes.issubset(dirs):
        for kind in matches:
            if kind in Path(root).parts:
                matches[kind].append(root)
assert all(len(paths) == 1 for paths in matches.values()), f'Chỉ thêm một dataset Intel với đủ train/test: {matches}'
TRAIN_DIR, TEST_DIR = matches['seg_train'][0], matches['seg_test'][0]
VERSIONS = {'python': platform.python_version(), 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(0)}
print('GPU:', VERSIONS, '\\nSource SHA256:', SOURCE_SHA)
print('Train:', TRAIN_DIR, '\\nTest:', TEST_DIR)
'''

RUN = '''import csv, subprocess, time, zipfile

OUTPUT_ROOT = Path(f'/kaggle/working/person{PERSON}')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def archive_seed(seed):
    folder = OUTPUT_ROOT / f'seed{seed}'
    archive = Path(f'/kaggle/working/person{PERSON}_seed{seed}.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in folder.rglob('*'):
            if path.is_file() and (INCLUDE_WEIGHTS or path.suffix != '.pth'):
                output.write(path, path.relative_to(OUTPUT_ROOT))
    print('ZIP:', archive)

# Thêm các ZIP personN_seedS.zip đã tải từ phiên trước vào Kaggle Input.
for archive in sorted(Path('/kaggle/input').rglob(f'person{PERSON}_seed*.zip')):
    with zipfile.ZipFile(archive) as saved:
        for member in saved.infolist():
            target = (OUTPUT_ROOT / member.filename).resolve()
            if not target.is_relative_to(OUTPUT_ROOT.resolve()):
                raise ValueError(f'Unsafe archive path: {member.filename}')
        saved.extractall(OUTPUT_ROOT)

assert len(SEEDS) == 10 and len(set(SEEDS)) == 10
assert set(SEEDS_TO_RUN).issubset(SEEDS)
session_started = time.monotonic()
session_budget = min(MAX_SESSION_HOURS, AVAILABLE_GPU_HOURS - 0.5)
assert session_budget > 0, 'Quota còn lại quá thấp, cập nhật AVAILABLE_GPU_HOURS.'
estimated_seed_hours = 6.5  # Ước tính ban đầu; cập nhật từ seed thực tế trong phiên.
for seed in SEEDS_TO_RUN:
    seed_dir = OUTPUT_ROOT / f'seed{seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    marker = seed_dir / 'DONE.json'
    if marker.exists():
        read_seed(seed_dir, SOURCE_SHA)
        print(f'Seed {seed}: đã hoàn tất, bỏ qua.')
        archive_seed(seed)
        continue
    elapsed_hours = (time.monotonic() - session_started) / 3600
    if elapsed_hours + estimated_seed_hours > session_budget:
        print(f'Dừng trước seed {seed}: còn {session_budget-elapsed_hours:.2f}h trong ngân sách phiên, dự kiến seed cần {estimated_seed_hours:.2f}h.')
        break
    command = [sys.executable, '-u', 'run_poga.py', '--method', 'all', '--model', 'all',
               '--epochs', '20', '--batch_size', '64', '--seeds', str(seed),
               '--benchmark_seeds', '--population_size', '20', '--total_iterations', '100',
               '--phase_size', '10', '--exchange_k', '3',
               '--train_dir', TRAIN_DIR, '--test_dir', TEST_DIR, '--save_dir', str(seed_dir)]
    started = time.monotonic()
    print(f'Bắt đầu seed {seed}:', ' '.join(command), flush=True)
    # Log đầy đủ vào file; chỉ in đầu ra định kỳ để notebook không phình vì tqdm.
    with (seed_dir / 'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=WORK_DIR, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(30)
            print(f'Seed {seed}: {(time.monotonic()-started)/60:.1f} phút; xem {seed_dir}/console.log', flush=True)
    if process.returncode:
        print((seed_dir / 'console.log').read_text(encoding='utf-8', errors='replace')[-8000:])
        raise RuntimeError(f'Seed {seed} lỗi; chưa đánh dấu hoàn tất.')
    result_files = sorted(seed_dir.glob('run_poga_all_*/experiment_results.csv'))
    result = result_files[-1]
    metadata = dict(seed=seed, epochs=20, batch_size=64, split_seed=42,
                    training_seed=seed, optimization_seed=seed,
                    source_sha=SOURCE_SHA, versions=VERSIONS,
                    elapsed_seconds=round(time.monotonic()-started, 2),
                    csv=result.relative_to(seed_dir).as_posix())
    marker.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        read_seed(seed_dir, SOURCE_SHA)
    except Exception:
        marker.unlink()
        raise
    archive_seed(seed)
    estimated_seed_hours = max(estimated_seed_hours, metadata['elapsed_seconds'] / 3600 * 1.2)
    print(f'Seed {seed}: đủ 33 kết quả, đã lưu ZIP.', flush=True)
print('Hoàn tất các seed được chọn. Tải ZIP trong /kaggle/working trước khi kết thúc phiên.')
'''

for person in range(1, 4):
    seeds = list(range((person - 1) * 10 + 1, person * 10 + 1))
    title = f'''# Benchmark 30 seed — Người {person}: seed {seeds[0]}–{seeds[-1]}

20 epoch × 11 phương pháp (Full B0 + 10 phương pháp rút gọn) × 3 CNN × 10 seed = **330 lượt huấn luyện/người**.

1. Import notebook vào Kaggle, bật **GPU** và **Internet**; thêm `puneet6060/intel-image-classification`.
2. Chạy các ô theo thứ tự. Mã nguồn đã nhúng giống nhau trong cả ba notebook.
3. Có thể chọn vài seed mỗi phiên bằng `SEEDS_TO_RUN`. Seed đang dở sẽ chạy lại; seed có `DONE.json` hợp lệ sẽ bỏ qua.
4. Sau mỗi seed, tải `person{person}_seedS.zip` trong `/kaggle/working`. Lần sau thêm ZIP đó vào Input rồi chạy lại.
5. ZIP mặc định gồm CSV, log, biểu đồ, mask và metadata; **không chứa trọng số .pth**. Đổi `INCLUDE_WEIGHTS=True` nếu cần.

Split train/validation cố định ở 42; seed tối ưu và seed huấn luyện thay đổi theo seed thực nghiệm.
B0 được huấn luyện thực tế cùng seed, epoch và GPU với các phương pháp còn lại.
Mỗi seed chạy tuần tự, chưa có ước lượng thời gian bảo đảm; có thể cần nhiều phiên Kaggle.
Xem `notebooks/HUONG_DAN_30_SEEDS.md` trong repository để gộp kết quả.
'''
    config = f'''PERSON = {person}
SEEDS = {seeds!r}
SEEDS_TO_RUN = SEEDS[:]  # Ví dụ SEEDS[:2] để chạy 2 seed trong phiên này.
INCLUDE_WEIGHTS = False
MAX_SESSION_HOURS = 10.5  # Dừng trước seed tiếp theo; không ngắt seed đang chạy.
AVAILABLE_GPU_HOURS = 30.0  # Điền quota thực tế còn lại trong Kaggle trước mỗi phiên.
print('Người', PERSON, '| Seeds được chọn:', SEEDS_TO_RUN)
'''
    cells = []
    for cell_type, content in [('markdown', title), ('code', config), ('code', SETUP.replace('__PAYLOAD__', repr(encoded))), ('code', RUN)]:
        cell = dict(cell_type=cell_type, metadata={}, source=content.splitlines(keepends=True))
        if cell_type == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python'), accelerator='GPU'), nbformat=4, nbformat_minor=4)
    destination = ROOT / 'notebooks' / f'kaggle_30seeds_person{person}.ipynb'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(destination)
