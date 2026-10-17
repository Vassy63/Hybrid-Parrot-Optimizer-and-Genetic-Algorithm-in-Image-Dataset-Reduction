"""Build three Kaggle notebooks and their shared source ZIP."""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
files = ['__init__.py', 'train.py', 'run_poga.py', 'scripts/merge_seed_results.py']
for package in ('configs', 'data', 'models', 'optimization', 'training', 'utils'):
    files.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / package).glob('*.py')))
source_zip = ROOT / 'notebooks' / 'KhoaLuanDatasetReduction_30seeds_source.zip'
source_zip.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(source_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(files):
        entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(entry, (ROOT / name).read_text(encoding='utf-8').encode('utf-8'))
print(source_zip)

SETUP = '''import os, sys, json, hashlib, platform, zipfile, shutil
from pathlib import Path
import torch

assert torch.cuda.is_available(), "Bật GPU trong Notebook options trước khi chạy."
# Mã nguồn và cache ở /kaggle/temp; /kaggle/working chỉ dành cho kết quả.
input_root = Path('/kaggle/input')

def source_roots(directory):
    return sorted(path.parent for path in directory.rglob('run_poga.py') if (path.parent / 'train.py').is_file())

def contains_source(path):
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            return any(name.endswith('run_poga.py') and name[:-len('run_poga.py')] + 'train.py' in names for name in names)
    except zipfile.BadZipFile:
        return False

if SOURCE_ZIP:
    selected = Path(SOURCE_ZIP)
    if not selected.exists():
        raise FileNotFoundError(f'SOURCE_ZIP không tồn tại: {selected}. Kiểm tra đường dẫn trong Kaggle Input.')
    archives = [] if selected.is_dir() else [selected]
    roots = source_roots(selected) if selected.is_dir() else []
else:
    archives = [path for path in sorted(input_root.rglob('*.zip')) if contains_source(path)]
    roots = source_roots(input_root) if not archives else []
if archives:
    if len(archives) != 1:
        raise ValueError('Có nhiều ZIP mã nguồn. Điền SOURCE_ZIP bằng một đường dẫn sau:\\n' + '\\n'.join(map(str, archives)))
    source_archive = archives[0]
    archive_sha = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    extracted = Path('/kaggle/temp') / f'poga_source_{archive_sha[:16]}'
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_archive) as archive:
        for entry in archive.infolist():
            if not (extracted / entry.filename).resolve().is_relative_to(extracted.resolve()):
                raise ValueError(f'Unsafe archive path: {entry.filename}')
        archive.extractall(extracted)
    roots = source_roots(extracted)
if len(roots) != 1:
    available = '\\n'.join(map(str, sorted(input_root.iterdir()))) if input_root.exists() else '(Input không tồn tại)'
    raise ValueError(f'Tìm thấy {len(roots)} bộ mã nguồn: {list(map(str, roots))}. Thêm ZIP có run_poga.py và train.py vào Input, hoặc điền SOURCE_ZIP là đường dẫn ZIP/thư mục source. Input hiện có:\\n{available}')
source_root = roots[0]
print('Source được chọn:', source_root)
sources = {path.relative_to(source_root).as_posix(): path.read_text(encoding='utf-8')
           for path in sorted(source_root.rglob('*.py')) if '__pycache__' not in path.parts}
assert 'scripts/merge_seed_results.py' in sources and '--benchmark_seeds' in sources['run_poga.py'], 'Source ZIP cũ; hãy upload ZIP 30seeds mới.'
SOURCE_SHA = hashlib.sha256(json.dumps(sources, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
WORK_DIR = Path('/kaggle/temp') / f'KhoaLuanDatasetReduction_{SOURCE_SHA[:16]}'
shutil.copytree(source_root, WORK_DIR, dirs_exist_ok=True)
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
    metadata, _ = read_seed(folder, SOURCE_SHA)
    run_dir = folder / Path(metadata['csv']).parent
    # Mỗi seed phải có checkpoint cho 11 phương pháp × 3 CNN.
    checkpoints = list((run_dir / 'model_runs').glob('*/*/weights/*_best.pth'))
    if len(checkpoints) != 33:
        raise ValueError(f'Seed {seed} thiếu checkpoint: {len(checkpoints)}/33. ZIP cũ bỏ trọng số không đủ để khôi phục đầy đủ; cần chạy lại seed.')
    archive = Path(f'/kaggle/working/person{PERSON}_seed{seed}.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in folder.rglob('*'):
            if path.is_file():
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
2. Thêm ZIP `KhoaLuanDatasetReduction_30seeds_source.zip` vào Input. Cả ba người dùng cùng ZIP. Chạy các ô theo thứ tự; không có mã nguồn nhúng.
3. Đợt đầu mặc định chọn 5 seed đầu bằng `SEEDS_TO_RUN = SEEDS[:5]` (165 lượt CNN/người). Đợt sau đổi thành `SEEDS[5:]`. Có thể cần nhiều phiên cho mỗi đợt.
4. Sau mỗi seed, tải `person{person}_seedS.zip` trong `/kaggle/working`. Lần sau thêm ZIP đó vào Input rồi chạy lại.
5. ZIP chứa **đầy đủ kết quả, kể cả checkpoint .pth**. Source và cache nằm ở `/kaggle/temp`; `/kaggle/working` dành cho kết quả và ZIP.

Split train/validation cố định ở 42; seed tối ưu và seed huấn luyện thay đổi theo seed thực nghiệm.
B0 được huấn luyện thực tế cùng seed, epoch và GPU với các phương pháp còn lại.
Mỗi seed chạy tuần tự, chưa có ước lượng thời gian bảo đảm; có thể cần nhiều phiên Kaggle.
Xem `notebooks/HUONG_DAN_30_SEEDS.md` trong repository để gộp kết quả.
'''
    config = f'''PERSON = {person}
SEEDS = {seeds!r}
SEEDS_TO_RUN = SEEDS[:5]  # Đợt 1: 5 seed đầu. Đợt 2: đổi thành SEEDS[5:].
SOURCE_ZIP = ''  # Tự tìm theo nội dung ZIP; có thể điền đường dẫn ZIP hoặc thư mục source đã giải nén.
MAX_SESSION_HOURS = 10.5  # Dừng trước seed tiếp theo; không ngắt seed đang chạy.
AVAILABLE_GPU_HOURS = 30.0  # Điền quota thực tế còn lại trong Kaggle trước mỗi phiên.
print('Người', PERSON, '| Seeds được chọn:', SEEDS_TO_RUN)
'''
    cells = []
    for cell_type, content in [('markdown', title), ('code', config), ('code', SETUP), ('code', RUN)]:
        cell = dict(cell_type=cell_type, metadata={}, source=content.splitlines(keepends=True))
        if cell_type == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python'), accelerator='GPU'), nbformat=4, nbformat_minor=4)
    destination = ROOT / 'notebooks' / f'kaggle_30seeds_person{person}.ipynb'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(destination)

# Dùng chung bộ nạp source và vòng chạy/xuất ZIP với notebook thật.
cpu_setup = SETUP.replace('assert torch.cuda.is_available(), "Bật GPU trong Notebook options trước khi chạy."',
                          'assert not torch.cuda.is_available(), "CPU smoke test phải chạy với Accelerator=None."')
cpu_setup = cpu_setup.replace("torch.cuda.get_device_name(0)", "'CPU'")
cpu_run = RUN.replace("Path(f'/kaggle/working/person{PERSON}')", "Path('/kaggle/working/cpu_smoke')")
cpu_run = cpu_run.replace("f'/kaggle/working/person{PERSON}_seed{seed}.zip'", "f'/kaggle/working/cpu_smoke_seed{seed}.zip'")
cpu_run = cpu_run.replace("f'person{PERSON}_seed*.zip'", "'cpu_smoke_seed*.zip'")
cpu_run = cpu_run.replace("'--epochs', '20'", "'--epochs', '1', '--num_samples', '120'")
cpu_run = cpu_run.replace("'--population_size', '20', '--total_iterations', '100'", "'--population_size', '6', '--total_iterations', '2'")
cpu_run = cpu_run.replace("'--phase_size', '10', '--exchange_k', '3'", "'--phase_size', '1', '--exchange_k', '1'")
cpu_run = cpu_run.replace('seed=seed, epochs=20', 'seed=seed, epochs=1')
cpu_run = cpu_run.replace('estimated_seed_hours = 6.5', 'estimated_seed_hours = 0.1')

cpu_validator = '''# Kiểm tra riêng profile 1 epoch; không dùng kết quả này trong benchmark 20 epoch.
import csv, math
def read_seed(folder, source_sha=None):
    folder = Path(folder)
    metadata = json.loads((folder / 'DONE.json').read_text(encoding='utf-8'))
    assert metadata['source_sha'] == source_sha, 'Source snapshot không khớp.'
    assert metadata['epochs'] == 1 and metadata['split_seed'] == 42
    result_path = (folder / metadata['csv']).resolve()
    assert result_path.is_relative_to(folder.resolve())
    with result_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = {(method, model, 1) for method in METHODS for model in MODELS}
    assert len(rows) == 33 and {(r['Method'], r['Model'], int(r['Seed'])) for r in rows} == expected
    assert all(math.isfinite(float(row[key])) for row in rows
               for key in ('Test_Acc', 'Val_Acc', 'Macro_F1', 'Train_Time_Sec', 'ARR_Percent', 'Speedup'))
    return metadata, rows
'''

cpu_verify = '''# Kiểm tra checkpoint, log, ZIP và khôi phục; không khởi chạy huấn luyện mới.
import tempfile
import numpy as np
metadata, rows = read_seed(OUTPUT_ROOT / 'seed1', SOURCE_SHA)
run_dir = OUTPUT_ROOT / 'seed1' / Path(metadata['csv']).parent
subsets = sorted((run_dir / 'subsets').glob('subset_*_seed1.npy'))
assert len(subsets) == 10
for path in subsets:
    mask = np.load(path, allow_pickle=False)
    assert mask.shape == (120,) and set(np.unique(mask)).issubset({0, 1}) and mask.sum() > 0
assert (run_dir / 'convergence_histories.csv').is_file()
checkpoints = sorted((run_dir / 'model_runs').glob('*/*/weights/*_best.pth'))
assert len(checkpoints) == 33
for path in checkpoints:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 1 and checkpoint['model_state_dict']
logs = list((run_dir / 'model_runs').glob('*/*/logs/run_metadata.json'))
assert len(logs) == 33
for path in logs:
    config = json.loads(path.read_text(encoding='utf-8'))['config']
    assert config['device'] == 'cpu' and config['epochs'] == 1 and config['num_samples'] == 120
assert len(list((run_dir / 'model_runs').glob('*/*/logs/training_log.csv'))) == 33
assert len(list((run_dir / 'model_runs').glob('*/*/plots/*_curve.png'))) == 33
archive = Path('/kaggle/working/cpu_smoke_seed1.zip')
with zipfile.ZipFile(archive) as saved:
    assert saved.testzip() is None, 'ZIP lỗi CRC.'
    assert sum(name.endswith('.pth') for name in saved.namelist()) == 33
    with tempfile.TemporaryDirectory(dir='/kaggle/temp') as restored:
        saved.extractall(restored)  # ZIP vừa được tạo từ output do notebook kiểm soát.
        read_seed(Path(restored) / 'seed1', SOURCE_SHA)
report = dict(status='CPU_SMOKE_PASS', device='cpu', samples=120, epochs=1,
              methods=11, models=3, result_rows=len(rows), subsets=len(subsets), checkpoints=len(checkpoints),
              zip_crc='PASS', restore='PASS', source_sha=SOURCE_SHA,
              scope='CPU tiny-data pipeline only; GPU/CUDA, full-data runtime and quota not tested')
(OUTPUT_ROOT / 'CPU_SMOKE_REPORT.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
print('Chạy lại ô benchmark ngay phía trên để kiểm tra bỏ qua seed đã xong và nén lại ZIP.')
'''

cpu_cells = [
    ('markdown', '''# Kiểm tra pipeline trên Kaggle CPU — chưa cần bật GPU

**Accelerator: None**. Bật Internet, thêm cùng ZIP source và Intel dataset như notebook thật.
Chạy Run All: nạp source (kể cả ZIP đổi tên), tìm dataset, trích đặc trưng thật,
chạy toàn bộ 11 phương pháp × 3 CNN trên 120 ảnh train, 1 seed, 1 epoch và ngân sách tối ưu rất nhỏ.
Sau đó kiểm tra đủ 33 dòng CSV, 33 checkpoint, log, biểu đồ, ZIP CRC và giải nén khôi phục.
Kết quả chỉ dùng kiểm tra kỹ thuật, không dùng báo cáo độ chính xác hay trộn với benchmark 20 epoch.

Cần tải trọng số ImageNet nếu chưa cache. Chưa có thời gian chạy đảm bảo trên CPU Kaggle.
`CPU_SMOKE_PASS` xác nhận đường chạy CPU với dữ liệu nhỏ; không xác nhận CUDA,
VRAM, thời gian chạy toàn bộ dataset hoặc quota GPU.
Source/cache ở `/kaggle/temp`; kết quả ở `/kaggle/working/cpu_smoke` và `cpu_smoke_seed1.zip`.
'''),
    ('code', '''import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
SOURCE_ZIP = ''  # Tự tìm ZIP source trong Input; có thể điền đường dẫn ZIP/thư mục.
PERSON = 0
SEEDS = list(range(1, 11))
SEEDS_TO_RUN = [1]
MAX_SESSION_HOURS = 10.5
AVAILABLE_GPU_HOURS = 30.0  # Chỉ tái sử dụng cơ chế giới hạn phiên; không bật GPU.
'''),
    ('code', cpu_setup), ('code', cpu_validator + '\n' + cpu_run), ('code', cpu_verify),
]
cells = []
for cell_type, content in cpu_cells:
    cell = dict(cell_type=cell_type, metadata={}, source=content.splitlines(keepends=True))
    if cell_type == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)
cpu_notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python')), nbformat=4, nbformat_minor=4)
destination = ROOT / 'notebooks/kaggle_cpu_smoke_test.ipynb'
destination.write_text(json.dumps(cpu_notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(destination)
