import numpy as np


def enforce_keep_floor(mask, labels, min_keep, rng):
    raw = np.asarray(mask)
    labels = np.asarray(labels)
    if raw.ndim != 1 or labels.ndim != 1 or len(raw) != len(labels) or not len(raw):
        raise ValueError('Mask và nhãn phải là hai vector cùng chiều dài.')
    if not np.isin(raw, [0, 1]).all() or not 1 <= min_keep <= len(raw):
        raise ValueError('Mask nhị phân hoặc số ảnh tối thiểu không hợp lệ.')
    fixed = raw.astype(np.int64, copy=True)
    # Restore absent classes before filling the remaining retention budget.
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        if not fixed[candidates].any():
            fixed[int(rng.choice(candidates))] = 1
    missing = int(min_keep - fixed.sum())
    if missing > 0:
        candidates = np.flatnonzero(fixed == 0)
        fixed[rng.choice(candidates, size=missing, replace=False)] = 1
    return fixed
