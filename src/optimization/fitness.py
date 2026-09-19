import numpy as np

def compute_raw_fitness(ind, dist_matrix, labels, num_classes, dist_max=1.0):
    """
    Tính 4 giá trị mục tiêu (chưa chuẩn hóa) của một cá thể.
    Đã tối ưu hóa siêu tốc bằng Vectorization (phép nhân ma trận).
    """
    M = np.sum(ind)
    N = len(labels)
    
    if M == 0:
        return 0.0, 0.0, 0.0, 1.0
        
    # 1. Compression Score (Com)
    com = 1.0 - (M / N)
    
    # 2. Diversity (Div)
    if M == 1:
        div = 0.0
    else:
        # Tối ưu siêu tốc: X.T @ dist_matrix @ X
        ind_float = ind.astype(np.float32)
        total_dist_sum = np.dot(ind_float, np.dot(dist_matrix, ind_float))
        sum_dist_upper = total_dist_sum / 2.0
        # Cập nhật: Thêm hệ số 2 theo công thức mới
        div = (2.0 / (M * (M - 1))) * (sum_dist_upper / dist_max)
        
    # 3. Coverage (Cov)
    selected_idx = np.where(ind == 1)[0]
    min_dists = np.min(dist_matrix[:, selected_idx], axis=1)
    # Cập nhật: Thêm / dist_max theo công thức mới
    cov = 1.0 - np.mean(min_dists / dist_max)
    
    # 4. Class Balance (Bal)
    sum_sq_diff = 0
    N_c_total = np.bincount(labels, minlength=num_classes)
    for c in range(num_classes):
        N_c = N_c_total[c]
        M_c = np.sum(labels[selected_idx] == c)
        ratio_S = M_c / M
        ratio_D = N_c / N
        sum_sq_diff += (ratio_S - ratio_D) ** 2
        
    bal = 1.0 - np.sqrt(sum_sq_diff / num_classes)
    
    return div, cov, bal, com

from tqdm import tqdm

def estimate_global_bounds(dist_matrix, labels, num_classes, dist_max, n_samples=1000):
    """
    Ước lượng ngưỡng (min, max) cho mỗi hàm mục tiêu.
    """
    N = len(labels)
    samples_raw = np.zeros((n_samples, 4))
    
    for i in tqdm(range(n_samples), desc="[*] Khảo sát Global Bounds", leave=False):
        prob = np.random.uniform(0.05, 0.5)
        ind = (np.random.rand(N) < prob).astype(int)
        
        if np.sum(ind) < 2:
            ind[np.random.choice(N, 2, replace=False)] = 1
            
        samples_raw[i] = compute_raw_fitness(ind, dist_matrix, labels, num_classes, dist_max)
        
    global_min = np.min(samples_raw, axis=0)
    global_max = np.max(samples_raw, axis=0)
    
    for i in range(4):
        if global_max[i] == global_min[i]:
            global_max[i] = global_min[i] + 1e-6
            
    return global_min, global_max

def fitness(ind, dist_matrix, labels, num_classes, config_fit, global_min, global_max, dist_max=1.0):
    """
    Tính fitness tổng hợp.
    """
    raw_fit = compute_raw_fitness(ind, dist_matrix, labels, num_classes, dist_max)
    raw_fit = np.array(raw_fit)
    
    norm_fit = (raw_fit - global_min) / (global_max - global_min)
    norm_fit = np.clip(norm_fit, 0.0, 1.0)
    
    total_fit = (
        config_fit.alpha * norm_fit[0] +
        config_fit.beta * norm_fit[1] +
        config_fit.gamma * norm_fit[2] +
        config_fit.delta * norm_fit[3]
    )
    
    return total_fit
