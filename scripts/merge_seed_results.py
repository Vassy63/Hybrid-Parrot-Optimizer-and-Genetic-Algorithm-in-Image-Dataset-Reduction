"""Merge completed seed folders from the three Kaggle notebooks."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

MODELS = ('resnet18', 'mobilenet_v3', 'densenet121')
METHODS = ('Full Dataset (B0)', 'Random Selection (B1)', 'Stratified Random (B2)',
           'GA-only', 'PO-only', 'S1 (Alt GA-PO)', 'S2 (Alt PO-GA)',
           'S3 (Block GA-PO)', 'S4 (Block PO-GA)', 'P1 (Parallel Merge)',
           'C1 (Coop Exchange)')


def read_seed(folder, source_sha=None):
    folder = Path(folder)
    metadata = json.loads((folder / 'DONE.json').read_text(encoding='utf-8'))
    if source_sha is not None and metadata['source_sha'] != source_sha:
        raise ValueError(f'Different source snapshot: {folder}')
    if metadata['epochs'] != 20 or metadata['batch_size'] != 64 or metadata['split_seed'] != 42:
        raise ValueError(f'Different protocol: {folder}')
    csv_path = (folder / metadata['csv']).resolve()
    if not csv_path.is_relative_to(folder.resolve()):
        raise ValueError(f'CSV path outside seed folder: {folder}')
    with csv_path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    expected = {(method, model, metadata['seed']) for method in METHODS for model in MODELS}
    keys = [(r['Method'], r['Model'], int(r['Seed'])) for r in rows]
    if len(keys) != 33 or set(keys) != expected:
        raise ValueError(f'Incomplete or duplicate experiment rows: {folder}')
    for row in rows:
        for key in ('Test_Acc', 'Macro_F1', 'Train_Time_Sec', 'ARR_Percent', 'Speedup'):
            if not math.isfinite(float(row[key])):
                raise ValueError(f'Invalid {key}: {folder}')
    return metadata, rows


def merge(root, output):
    from scipy.stats import wilcoxon

    records, seen, fingerprint = [], set(), None
    for marker in sorted(Path(root).rglob('DONE.json')):
        metadata, rows = read_seed(marker.parent, fingerprint)
        fingerprint = metadata['source_sha']
        seed = metadata['seed']
        if seed in seen:
            raise ValueError(f'Duplicate seed {seed}: {marker}')
        seen.add(seed)
        records.extend(rows)
    if seen != set(range(1, 31)):
        raise ValueError(f'Expected seeds 1..30; missing={sorted(set(range(1,31))-seen)}, extra={sorted(seen-set(range(1,31)))}')
    numeric = ['Test_Acc', 'Val_Acc', 'Macro_F1', 'DR_Percent', 'ARR_Percent',
               'Train_Time_Sec', 'Opt_Time_Sec', 'Speedup', 'Fitness_Total']
    records.sort(key=lambda r: (r['Method'], r['Model'], int(r['Seed'])))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def write_csv(name, rows):
        with (output / name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv('experiment_results_30seeds.csv', records)
    summary = []
    for method in METHODS:
        for model in MODELS:
            group = [r for r in records if r['Method'] == method and r['Model'] == model]
            row = dict(Method=method, Model=model)
            for metric in numeric:
                values = [float(r[metric]) for r in group]
                row.update({f'{metric}_count': len(values), f'{metric}_mean': statistics.mean(values),
                            f'{metric}_std': statistics.stdev(values)})
            summary.append(row)
    write_csv('summary_30seeds.csv', summary)
    comparisons = []
    for model in MODELS:
        table = {(r['Method'], int(r['Seed'])): float(r['Test_Acc']) for r in records if r['Model'] == model}
        target = [table[('C1 (Coop Exchange)', seed)] for seed in range(1, 31)]
        for method in METHODS:
            if method == 'C1 (Coop Exchange)':
                continue
            comparator = [table[(method, seed)] for seed in range(1, 31)]
            diff = [a-b for a, b in zip(target, comparator)]
            p = 1.0 if all(d == 0 for d in diff) else float(wilcoxon(target, comparator, alternative='two-sided').pvalue)
            comparisons.append(dict(Model=model, Target='C1 (Coop Exchange)', Comparator=method,
                                    N=30, Mean_Delta_Acc=statistics.mean(diff), P_Two_Sided=p))
    write_csv('wilcoxon_C1_30seeds.csv', comparisons)
    print(f'Merged {len(seen)} seeds, {len(records)} rows into {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='Directory containing the extracted result ZIPs')
    parser.add_argument('--output', default='outputs/merged_30seeds')
    args = parser.parse_args()
    merge(args.root, args.output)
