import numpy as np
import torch

torch.set_num_threads(1)

def precompute_distance_matrix(features: np.ndarray, metric: str = 'cosine') -> np.ndarray:
    """
    Tính ma trận khoảng cách giữa tất cả các cặp vector đặc trưng (N x 512)
    và chuẩn hóa trực tiếp về [0, 1] bằng D_max.
    Sử dụng phép nhân ma trận BLAS float32 tốc độ cao (không phụ thuộc scipy/sklearn).
    """
    feats = features.astype(np.float32)
    if metric == 'cosine':
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1e-8, norms)
        f_norm = feats / norms
        sim = np.dot(f_norm, f_norm.T)
        dist = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float32)
    else:
        sq = np.sum(feats ** 2, axis=1, keepdims=True)
        dist_sq = np.maximum(sq + sq.T - 2.0 * np.dot(feats, feats.T), 0.0)
        dist = np.sqrt(dist_sq).astype(np.float32)

    np.fill_diagonal(dist, 0.0)
    d_max = float(np.max(dist))
    if d_max > 1e-8:
        dist /= d_max
    return dist

_DIST_TENSOR_CACHE = {}

def _get_dist_tensor(dist_matrix: np.ndarray) -> torch.Tensor:
    key = id(dist_matrix)
    if key not in _DIST_TENSOR_CACHE:
        _DIST_TENSOR_CACHE.clear()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _DIST_TENSOR_CACHE[key] = torch.from_numpy(dist_matrix).to(device)
    return _DIST_TENSOR_CACHE[key]

def compute_diversity(dist_matrix: np.ndarray, selected_idx: np.ndarray) -> float:
    """F_Div in [0, 1]: Khoảng cách trung bình giữa các mẫu được chọn."""
    M = len(selected_idx)
    if M < 2:
        return 0.0
    dist_pt = _get_dist_tensor(dist_matrix)
    idx_pt = torch.as_tensor(selected_idx, dtype=torch.long, device=dist_pt.device)
    sub_matrix = torch.index_select(torch.index_select(dist_pt, 0, idx_pt), 1, idx_pt)
    sum_dist = torch.sum(sub_matrix).item()
    return float(sum_dist / (M * (M - 1)))

def compute_coverage(dist_matrix: np.ndarray, selected_idx: np.ndarray) -> float:
    """F_Cov in [0, 1]: Độ bao phủ không gian đặc trưng (1 - avg_min_dist)."""
    if len(selected_idx) < 1:
        return 0.0
    dist_pt = _get_dist_tensor(dist_matrix)
    idx_pt = torch.as_tensor(selected_idx, dtype=torch.long, device=dist_pt.device)
    min_dists = torch.index_select(dist_pt, 1, idx_pt).min(dim=1)[0]
    avg_min_dist = torch.mean(min_dists).item()
    return float(max(0.0, 1.0 - avg_min_dist))

def compute_balance(labels: np.ndarray, selected_idx: np.ndarray, num_classes: int) -> float:
    """F_Bal in [0, 1]: Độ cân bằng phân bố lớp so với tập gốc."""
    if len(selected_idx) == 0:
        return 0.0
    N = len(labels)
    M = len(selected_idx)
    orig_ratios = np.bincount(labels, minlength=num_classes) / N
    sel_ratios = np.bincount(labels[selected_idx], minlength=num_classes) / M
    deviation = np.sqrt(np.mean((sel_ratios - orig_ratios) ** 2))
    return float(1.0 - deviation)

def compute_compression(solution_vector: np.ndarray) -> float:
    """F_Com in [0, 1]: Tỷ lệ nén dữ liệu (1 - M/N)."""
    M = np.sum(solution_vector)
    N = len(solution_vector)
    return float(1.0 - (M / N))

def compute_raw_fitness(solution: np.ndarray, dist_matrix: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Trả về 4 thành phần [Div, Cov, Bal, Com] đều thuộc [0, 1]."""
    selected_idx = np.where(solution == 1)[0]
    M = len(selected_idx)
    if M < 2:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dist_pt = _get_dist_tensor(dist_matrix)
    idx_pt = torch.as_tensor(selected_idx, dtype=torch.long, device=dist_pt.device)
    cols = torch.index_select(dist_pt, 1, idx_pt)
    div = float(torch.index_select(cols, 0, idx_pt).sum().item() / (M * (M - 1)))
    cov = float(max(0.0, 1.0 - cols.min(dim=1)[0].mean().item()))
    bal = compute_balance(labels, selected_idx, num_classes)
    com = compute_compression(solution)
    return np.array([div, cov, bal, com], dtype=np.float64)

def fitness(solution: np.ndarray, dist_matrix: np.ndarray, labels: np.ndarray, num_classes: int, weights) -> float:
    """Tính điểm Fitness tổng hợp F(X) = alpha*Div + beta*Cov + gamma*Bal + delta*Com."""
    selected_idx = np.where(solution == 1)[0]
    if len(selected_idx) < 2:
        return 0.0
    raw = compute_raw_fitness(solution, dist_matrix, labels, num_classes)
    w = np.array([weights.w_div, weights.w_cov, weights.w_bal, weights.w_com], dtype=np.float64)
    return float(np.sum(raw * w))

def repair_solution(solution: np.ndarray) -> np.ndarray:
    """Đảm bảo tập con có ít nhất 2 mẫu (M >= 2) để tính được F_Div."""
    sol = solution.copy()
    M = int(np.sum(sol))
    if M < 2:
        zero_idx = np.where(sol == 0)[0]
        need = min(2 - M, len(zero_idx))
        if need > 0:
            sol[np.random.choice(zero_idx, size=need, replace=False)] = 1
    return sol
