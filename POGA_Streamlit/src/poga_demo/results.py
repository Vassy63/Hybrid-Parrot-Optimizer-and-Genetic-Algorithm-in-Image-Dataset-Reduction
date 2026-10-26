import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

NUMERIC = ['Test_Acc','Val_Acc','Macro_F1','DR_Percent','ARR_Percent','Delta_Acc',
           'Train_Time_Sec','Speedup','Opt_Time_Sec','Fitness_Total','Subset_Size_M']


def read_results(source):
    close = isinstance(source, (str, Path))
    handle = open(source, encoding='utf-8-sig', newline='') if close else source
    try:
        reader = csv.DictReader(handle)
        if not {'Method','Model','Seed','Test_Acc'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV thiếu Method, Model, Seed hoặc Test_Acc.')
        rows, keys = [], set()
        for row in reader:
            key = (row['Method'], row['Model'], row['Seed'])
            if key in keys:
                raise ValueError('CSV có tổ hợp phương pháp–CNN–seed trùng nhau.')
            keys.add(key)
            for name in NUMERIC:
                if row.get(name, '') not in ['', 'N/A']:
                    value = float(row[name])
                    if not math.isfinite(value):
                        raise ValueError(f'CSV chứa giá trị không hữu hạn ở {name}.')
                    row[name] = value
                elif name in row:
                    row[name] = None
            rows.append(row)
        if not rows:
            raise ValueError('CSV không có kết quả.')
        return rows
    finally:
        if close:
            handle.close()


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['Method'], row['Model']].append(row)
    result = []
    for (method, model), values in groups.items():
        record = {'Method': method, 'Model': model, 'Count': len(values)}
        for name in NUMERIC:
            numbers = [float(v[name]) for v in values if v.get(name) is not None]
            if numbers:
                record[name + '_mean'] = statistics.mean(numbers)
                record[name + '_std'] = statistics.stdev(numbers) if len(numbers) > 1 else 0
        result.append(record)
    return result


def paired_comparison(full_acc, subset_acc, full_seconds, subset_seconds):
    return {
        'ARR_Percent': subset_acc / full_acc * 100 if full_acc > 0 else None,
        'Delta_Acc': subset_acc - full_acc,
        'Speedup': full_seconds / subset_seconds if subset_seconds > 0 else None,
    }
