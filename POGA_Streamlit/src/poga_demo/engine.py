"""Adapter around the supplied engine; this module imports torch only in a worker."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .constraints import enforce_keep_floor
from .jobs import write_json
from .results import paired_comparison
from .settings import minimum_retained, PROJECT_ROOT


def run_demo(cfg, folder, emit):
    sys.path.insert(0, str(Path(cfg.source_dir).resolve()))
    import numpy as np
    import torch
    from configs import BaselineConfig, POGAConfig
    from data import create_dataloaders, extract_and_cache_features
    from optimization import precompute_distance_matrix
    import importlib
    fitness_module = importlib.import_module('optimization.fitness')
    ga_module = importlib.import_module('optimization.genetic_algorithm')
    po_module = importlib.import_module('optimization.parrot_optimizer')
    hybrids = importlib.import_module('optimization.hybrid_poga')
    train_module = importlib.import_module('train')
    from training import BaselineTrainer
    from utils import set_seed

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.device != 'auto':
        if cfg.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('PyTorch chưa nhận CUDA. Hãy cài bản PyTorch CUDA hoặc chọn CPU.')
        device = cfg.device
    base = BaselineConfig(train_dir=cfg.train_dir, test_dir=cfg.test_dir, save_dir=str(folder),
        model_name=cfg.model, epochs=cfg.epochs, batch_size=cfg.batch_size, seed=42,
        num_samples=cfg.num_samples, device=device, run_id=folder.name, run_dir=str(folder / 'model_runs' / 'b0'))
    emit('data', 'Đang tạo train / validation / test', device=device)
    full_loaders = create_dataloaders(base)
    train_loader, val_loader, test_loader, classes = full_loaders
    if test_loader is None:
        raise ValueError('Demo cần tập test độc lập; không dùng validation thay test.')
    if len(classes) != 6 or len(train_loader.dataset) < 120 or not len(val_loader.dataset) or not len(test_loader.dataset):
        raise ValueError('Dữ liệu không đủ: cần sáu lớp và train/validation/test không rỗng.')
    test_base = getattr(test_loader.dataset, 'dataset', test_loader.dataset)
    if getattr(test_base, 'classes', classes) != classes:
        raise ValueError('Danh sách lớp giữa train và test không khớp.')
    underlying = train_loader.dataset.dataset
    indices = train_loader.dataset.indices
    labels_expected = np.array([underlying.samples[i][1] for i in indices])
    if len(np.unique(labels_expected)) != len(classes):
        raise ValueError('Mẫu train thiếu lớp. Tăng số ảnh demo hoặc đổi cách lấy mẫu.')
    records = []
    hasher = hashlib.sha256()
    for index in indices:
        path, label = underlying.samples[index]
        stat = Path(path).stat()
        record = {'path': str(Path(path).resolve()), 'label': int(label), 'class': classes[label]}
        records.append(record)
        hasher.update(json.dumps([record['path'], label, stat.st_size, stat.st_mtime_ns]).encode())
    import torchvision
    hasher.update(json.dumps([42, base.image_size, 'resnet18-imagenet-default', torchvision.__version__]).encode())
    cache = PROJECT_ROOT / 'cache' / 'features' / hasher.hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    # A fingerprint names the exact ordered train files and extractor configuration.
    emit('features', 'Đang nạp hoặc trích xuất đặc trưng ResNet-18',
         train_count=len(records), val_count=len(val_loader.dataset), test_count=len(test_loader.dataset))
    features, labels = extract_and_cache_features(base, cache_dir=str(cache))
    if features.shape != (len(records), 512) or not np.array_equal(labels, labels_expected) or not np.isfinite(features).all():
        raise ValueError(f'Cache đặc trưng không khớp mẫu train: {cache}. Đổi tên folder cache này rồi chạy lại.')
    write_json(cache / 'manifest.json', {'fingerprint': hasher.hexdigest(), 'train_count': len(records),
        'split_seed': 42, 'classes': classes, 'torchvision': torchvision.__version__})

    n = len(labels)
    floor = minimum_retained(n, cfg.reduction_percent)
    rng = np.random.RandomState(cfg.seed)
    def repair(mask):
        return enforce_keep_floor(mask, labels, floor, rng)
    ga_module.repair_solution = repair
    po_module.repair_solution = repair
    poga = POGAConfig()
    poga.ga.population_size = poga.po.population_size = cfg.population_size
    poga.total_iterations = cfg.iterations
    poga.phase_size = cfg.phase_size
    poga.num_classes = len(classes)
    poga.ga.init_ratio = cfg.keep_ratio
    poga.fitness.target_ratio = cfg.keep_ratio if cfg.keep_ratio < 1 else None
    target_ratio = poga.fitness.target_ratio
    original_raw = fitness_module.compute_raw_fitness
    def raw_history(*args, **kwargs):
        kwargs.setdefault('target_ratio', target_ratio)
        return original_raw(*args, **kwargs)
    ga_module.compute_raw_fitness = po_module.compute_raw_fitness = raw_history
    # Respect an explicit CPU choice, including the fitness matrix.
    tensor_cache = {}
    def tensor_for(matrix):
        if id(matrix) not in tensor_cache:
            tensor_cache.clear()
            tensor_cache[id(matrix)] = torch.from_numpy(matrix).to(device)
        return tensor_cache[id(matrix)]
    fitness_module._get_dist_tensor = tensor_for
    emit('optimization', 'Đang tính khoảng cách và lựa chọn tập con', minimum_retained=floor)
    matrix = precompute_distance_matrix(features, metric='cosine')
    original_fitness = fitness_module.fitness
    eval_count, best, last_emit = 0, float('-inf'), 0.0
    def tracked_fitness(*args, **kwargs):
        nonlocal eval_count, best, last_emit
        value = original_fitness(*args, **kwargs)
        eval_count += 1
        best = max(best, value)
        now = time.monotonic()
        if now - last_emit > .4:
            emit('optimization', 'Đang đánh giá nghiệm', evaluations=eval_count, fitness=best)
            last_emit = now
        return value
    ga_module.fitness = po_module.fitness = tracked_fitness
    set_seed(cfg.seed)
    started_opt = time.perf_counter()
    history = []
    if cfg.reduction_percent == 0:
        mask = np.ones(n, dtype=int)
    elif cfg.method in ['random', 'stratified']:
        mask = hybrids.random_selection(n, floor, cfg.seed) if cfg.method == 'random' else hybrids.stratified_random(labels, floor, cfg.seed)
        mask = repair(mask)
    else:
        initial = np.array([repair(row) for row in (rng.rand(cfg.population_size, n) < cfg.keep_ratio).astype(int)])
        ga = ga_module.GeneticAlgorithm(poga, matrix, labels, len(classes))
        initial_fitness = ga.evaluate_population(initial)
        if cfg.method == 'ga':
            population, _, history = ga.run(initial, initial_fitness, steps=cfg.iterations, total_gens=cfg.iterations, disable_early_stop=True)
            mask = population[0]
        elif cfg.method == 'po':
            po = po_module.ParrotOptimizer(poga, matrix, labels, len(classes))
            mask, _, history = po.run(initial, initial_fitness, steps=cfg.iterations, total_iters=cfg.iterations, disable_early_stop=True)
        else:
            methods = {'s1': hybrids.run_s1_alternating_ga_po, 's2': hybrids.run_s2_alternating_po_ga,
                's3': hybrids.run_s3_block_ga_po, 's4': hybrids.run_s4_block_po_ga,
                'p1': hybrids.run_p1_parallel_merge, 'c1': hybrids.run_c1_cooperative_exchange}
            mask, _, history = methods[cfg.method](poga, matrix, labels, len(classes), initial, initial_fitness)
        mask = repair(mask)
    opt_seconds = time.perf_counter() - started_opt
    raw = raw_history(mask, matrix, labels, len(classes))
    fitness_value = original_fitness(mask, matrix, labels, len(classes), poga.fitness)
    selected = int(mask.sum())
    dr = 100 * (1 - selected / n)
    if selected < floor or dr > cfg.reduction_percent + 1e-8:
        raise RuntimeError('Tập con vi phạm giới hạn rút gọn.')
    (folder / 'subsets').mkdir(exist_ok=True)
    mask_path = folder / 'subsets' / f'subset_{cfg.method}_seed{cfg.seed}.npy'
    np.save(mask_path, mask, allow_pickle=False)
    for i, record in enumerate(records):
        record['selected'] = bool(mask[i])
    write_json(folder / 'image_manifest.json', records)
    if not history:
        history = [{'total': fitness_value, 'div': float(raw[0]), 'cov': float(raw[1]), 'bal': float(raw[2]), 'com': float(raw[3])}]
    with (folder / 'convergence_histories.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Iteration','total','div','cov','bal','com'])
        writer.writeheader()
        writer.writerows([{'Iteration': i + 1, **h} for i, h in enumerate(history)])
    emit('subset', 'Đã chọn tập con', selected=selected, original=n, actual_dr=dr,
         target_dr=cfg.reduction_percent, fitness=fitness_value)

    # Keep all training behavior/checkpointing from the original engine.
    training_label = ''
    class MonitoredLoader:
        def __init__(self, loader, stage):
            self.loader, self.stage = loader, stage
        def __len__(self):
            return len(self.loader)
        def __getattr__(self, name):
            return getattr(self.loader, name)
        def __iter__(self):
            last = 0
            for i, batch in enumerate(self.loader, 1):
                now = time.monotonic()
                if now - last > .7:
                    emit('training', f'{training_label} · {self.stage}', batch=i, batches=len(self.loader))
                    last = now
                yield batch

    class ProgressTrainer(BaselineTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_val_acc = float('-inf')  # Save the first epoch even if accuracy is zero.
            self.train_loader = MonitoredLoader(self.train_loader, 'huấn luyện')
            self.val_loader = MonitoredLoader(self.val_loader, 'validation')
            self.test_loader = MonitoredLoader(self.test_loader, 'test')
        def train_epoch(self, epoch):
            loss, acc = super().train_epoch(epoch)
            emit('epoch', f'{training_label} · epoch {epoch}/{self.config.epochs}',
                 model_label=training_label, epoch=epoch, train_loss=float(loss), train_acc=float(acc) * 100)
            return loss, acc
        def validate(self, epoch):
            loss, acc = super().validate(epoch)
            emit('validation', f'{training_label} · validation epoch {epoch}',
                 model_label=training_label, epoch=epoch, val_loss=float(loss), val_acc=float(acc) * 100)
            return loss, acc
    train_module.BaselineTrainer = ProgressTrainer

    baseline = None
    if cfg.run_baseline:
        training_label = 'B0'
        emit('training', 'Đang huấn luyện B0 trên tập train demo đầy đủ')
        set_seed(cfg.seed)
        baseline = train_module.train_single_model(base, *full_loaders)
    if selected == n and baseline is not None:
        reduced = baseline
        emit('training', 'Tập con trùng tập B0; sử dụng cùng kết quả đo')
    else:
        training_label = cfg.method.upper()
        reduced_cfg = copy.copy(base)
        reduced_cfg.subset_indices_path = str(mask_path)
        reduced_cfg.run_id = f'{folder.name}_{cfg.method}'
        reduced_cfg.run_dir = str(folder / 'model_runs' / cfg.method)
        reduced_loaders = create_dataloaders(reduced_cfg)
        emit('training', 'Đang huấn luyện CNN trên tập con')
        set_seed(cfg.seed)
        reduced = train_module.train_single_model(reduced_cfg, *reduced_loaders)

    rows = []
    def row_for(method, result, count, reduction, optimization_time, score):
        metrics = result['test_metrics']
        row = {'Method': method, 'Model': cfg.model, 'Seed': cfg.seed, 'Subset_Size_M': count,
            'DR_Percent': reduction, 'Test_Acc': float(result['test_acc']), 'Val_Acc': float(result['best_val_acc']),
            'Macro_F1': float(result['macro_f1']), 'Train_Time_Sec': result['total_time_seconds'],
            'Opt_Time_Sec': optimization_time, 'Fitness_Total': score}
        if baseline is not None:
            row.update(paired_comparison(float(baseline['test_acc']), row['Test_Acc'],
                baseline['total_time_seconds'], row['Train_Time_Sec']))
        else:
            row.update(ARR_Percent=None, Delta_Acc=None, Speedup=None)
        return row
    if baseline is not None:
        rows.append(row_for('Full Dataset (B0)', baseline, n, 0, 0, 0))
    rows.append(row_for(cfg.method.upper(), reduced, selected, dr, opt_seconds, fitness_value))
    with (folder / 'experiment_results.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary = {'demo': True, 'rows': rows, 'classes': classes, 'train_before': n, 'train_after': selected,
        'val_count': len(val_loader.dataset), 'test_count': len(test_loader.dataset), 'actual_dr': dr,
        'requested_max_dr': cfg.reduction_percent, 'fitness': fitness_value,
        'fitness_components': dict(zip(['Diversity','Coverage','Balance','Target_Com'], map(float, raw))),
        'opt_seconds': opt_seconds, 'split_seed': 42, 'cnn_seed': cfg.seed, 'device': device,
        'reused_full_result': selected == n and baseline is not None,
        'note': 'Một seed minh họa; speedup chỉ tính thời gian huấn luyện. Không dùng để suy luận thống kê.'}
    write_json(folder / 'summary.json', summary)
    emit('done', 'Hoàn tất tối ưu, huấn luyện và đánh giá', actual_dr=dr, selected=selected,
         test_acc=float(reduced['test_acc']), macro_f1=float(reduced['macro_f1']))
