import os
import sys
import time
import json
import csv
import argparse
import datetime

# 1. Khống chế số luồng C-level để tránh lỗi cạn kiệt tài nguyên Windows (Error #1450)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 2. Xử lý bảng mã tiếng Việt cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import concurrent.futures
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

import torch
torch.set_num_threads(1)

from src.config import config
from src.optimization.fitness import fitness, compute_raw_fitness, estimate_global_bounds
from src.evaluation.classifier import train_and_predict
from src.evaluation.evaluator import evaluate_predictions, evaluate_subset
import src.data.feature_extractor as feature_extractor

class GeneticAlgorithmReducer:
    """
    Bộ tối ưu hóa rút gọn dữ liệu ảnh sử dụng Giải thuật Di truyền (GA).
    """
    def __init__(self, dist_matrix, labels, num_classes, global_min, global_max, dist_max, args):
        self.dist_matrix = dist_matrix
        self.labels = labels
        self.num_classes = num_classes
        self.global_min = global_min
        self.global_max = global_max
        self.dist_max = dist_max
        self.args = args
        self.N = dist_matrix.shape[0]
        self.fitness_config = config.fitness

    def evaluate_individual(self, ind):
        return fitness(
            ind, self.dist_matrix, self.labels, 
            self.num_classes, self.fitness_config, 
            self.global_min, self.global_max, self.dist_max
        )

    def evaluate_population(self, population):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            scores = list(executor.map(self.evaluate_individual, population))
        return np.array(scores)

    def tournament_selection(self, population, fitness_scores):
        idx = np.random.choice(self.args.pop_size, size=self.args.tournament_size, replace=False)
        best = idx[np.argmax(fitness_scores[idx])]
        return population[best].copy()

    def crossover(self, p1, p2):
        if np.random.rand() < self.args.crossover_rate:
            mask = (np.random.rand(self.N) < 0.5).astype(int)
            c1 = np.where(mask, p1, p2)
            c2 = np.where(mask, p2, p1)
            return c1, c2
        return p1.copy(), p2.copy()

    def mutate(self, ind, gen):
        p_max = self.args.mutation_rate_max
        p_min = self.args.mutation_rate_min
        p_mut = max(p_min, p_max - (p_max - p_min) * (gen / max(1, self.args.generations)))
        mask = (np.random.rand(self.N) < p_mut).astype(int)
        return np.where(mask, 1 - ind, ind)

    def run(self):
        print("\n[*] Khởi tạo quần thể ban đầu...")
        # Khởi tạo với tỷ lệ chọn ~ init_ratio (mặc định 0.3)
        population = (np.random.rand(self.args.pop_size, self.N) < self.args.init_ratio).astype(int)
        fitness_scores = self.evaluate_population(population)

        history = []
        best_idx = np.argmax(fitness_scores)
        best_overall_sol = population[best_idx].copy()
        best_overall_fit = fitness_scores[best_idx]
        stagnation_count = 0

        print(f"[*] Bắt đầu tiến hóa GA ({self.args.generations} thế hệ, Pop Size: {self.args.pop_size})...")

        for gen in tqdm(range(1, self.args.generations + 1), desc="GA Generations"):
            t_gen_start = time.time()
            
            # Sắp xếp và giữ lại Elitism
            sorted_idx = np.argsort(fitness_scores)[::-1]
            new_pop = [population[sorted_idx[i]].copy() for i in range(self.args.elitism_count)]

            # Sinh sản & Đột biến
            while len(new_pop) < self.args.pop_size:
                p1 = self.tournament_selection(population, fitness_scores)
                p2 = self.tournament_selection(population, fitness_scores)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1, gen)
                c2 = self.mutate(c2, gen)
                new_pop.append(c1)
                if len(new_pop) < self.args.pop_size:
                    new_pop.append(c2)

            population = np.array(new_pop)
            fitness_scores = self.evaluate_population(population)

            # Thống kê thế hệ
            cur_best_idx = np.argmax(fitness_scores)
            cur_best_fit = fitness_scores[cur_best_idx]
            mean_fit = np.mean(fitness_scores)

            if cur_best_fit > best_overall_fit + 1e-6:
                best_overall_fit = cur_best_fit
                best_overall_sol = population[cur_best_idx].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1

            # 4 hàm mục tiêu thô của cá thể tốt nhất
            raw_fit = compute_raw_fitness(best_overall_sol, self.dist_matrix, self.labels, self.num_classes, self.dist_max)
            gen_time = time.time() - t_gen_start

            history.append({
                'generation': gen,
                'best_fitness': float(best_overall_fit),
                'mean_fitness': float(mean_fit),
                'f_diversity': float(raw_fit[0]),
                'f_coverage': float(raw_fit[1]),
                'f_balance': float(raw_fit[2]),
                'f_compression': float(raw_fit[3]),
                'time_seconds': float(gen_time)
            })

            # Dừng sớm nếu không cải thiện
            if stagnation_count >= self.args.patience:
                print(f"\n[!] Early Stopping tại thế hệ {gen} do không cải thiện sau {stagnation_count} vòng.")
                break

        return best_overall_sol, best_overall_fit, history

def plot_all_results(history, train_features, train_labels, best_solution, classes, run_dir, run_id, conf_matrices=None):
    """Vẽ toàn bộ các biểu đồ kết quả."""
    os.makedirs(run_dir, exist_ok=True)
    selected_idx = np.where(best_solution == 1)[0]
    subset_labels = train_labels[selected_idx]
    
    # -------------------------------------------------------------
    # 1. BIỂU ĐỒ FITNESS CHÍNH: fitness_curve.png
    # -------------------------------------------------------------
    gens = [h['generation'] for h in history]
    best_fits = [h['best_fitness'] for h in history]
    mean_fits = [h['mean_fitness'] for h in history]
    
    f_com = [h['f_compression'] for h in history]
    f_cov = [h['f_coverage'] for h in history]
    f_div = [h['f_diversity'] for h in history]
    f_bal = [h['f_balance'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Đồ thị 1: Hội tụ Fitness Tổng (Best vs Mean)
    axes[0].plot(gens, best_fits, label='Best Fitness', color='#1f77b4', linewidth=2.2)
    axes[0].plot(gens, mean_fits, label='Mean Fitness', color='#ff7f0e', linestyle='--', linewidth=1.8)
    axes[0].set_title('Tiến hóa Điểm Thích nghi (Fitness Evolution)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Thế hệ (Generations)', fontsize=10)
    axes[0].set_ylabel('Điểm Fitness Tổng hợp', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)

    # Đồ thị 2: 4 Hàm mục tiêu thành phần
    axes[1].plot(gens, f_com, label='F_Com (Độ nén)', color='#2ca02c', linewidth=2)
    axes[1].plot(gens, f_cov, label='F_Cov (Bao phủ)', color='#d62728', linewidth=1.8)
    axes[1].plot(gens, f_div, label='F_Div (Đa dạng)', color='#9467bd', linewidth=1.8)
    axes[1].plot(gens, f_bal, label='F_Bal (Cân bằng)', color='#bcbd22', linewidth=1.8)
    axes[1].set_title('Động thái 4 Hàm Mục tiêu Thành phần', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Thế hệ (Generations)', fontsize=10)
    axes[1].set_ylabel('Giá trị hàm mục tiêu', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=9)

    plt.suptitle(f"GA Fitness Analysis | Run ID: {run_id} | Final Best Fitness: {best_fits[-1]:.4f}", fontsize=13, y=0.98)
    plt.tight_layout()
    fitness_plot_path = os.path.join(run_dir, "fitness_curve.png")
    plt.savefig(fitness_plot_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ Fitness tại: {fitness_plot_path}")

    # -------------------------------------------------------------
    # 2. BIỂU ĐỒ PHÂN BỐ LỚP: class_distribution.png
    # -------------------------------------------------------------
    orig_counts = np.bincount(train_labels, minlength=len(classes))
    sub_counts = np.bincount(subset_labels, minlength=len(classes))
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, orig_counts, width, label='Tập gốc (Full)', color='#4c72b0')
    ax.bar(x + width/2, sub_counts, width, label='Tập rút gọn (GA Subset)', color='#55a868')
    ax.set_ylabel('Số lượng mẫu ảnh', fontsize=11)
    ax.set_title(f'So sánh Phân bố 6 Lớp Ảnh | Run ID: {run_id}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    offset = max(orig_counts) * 0.015
    for i, v in enumerate(orig_counts):
        ax.text(i - width/2, v + offset, str(v), ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(sub_counts):
        ax.text(i + width/2, v + offset, str(v), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    dist_plot_path = os.path.join(run_dir, "class_distribution.png")
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ phân bố lớp tại: {dist_plot_path}")

    # -------------------------------------------------------------
    # 3. BIỂU ĐỒ KHÔNG GIAN 2D: feature_space_pca.png
    # -------------------------------------------------------------
    try:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(train_features)
        
        plt.figure(figsize=(10, 7))
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c='#cccccc', label='Ảnh bị loại bỏ', alpha=0.4, s=15)
        plt.scatter(features_2d[selected_idx, 0], features_2d[selected_idx, 1], 
                    c='#d62728', label='Ảnh GA chọn giữ lại', alpha=0.8, s=25, edgecolors='black', linewidth=0.3)
        plt.title(f'Không gian Đặc trưng 2D (PCA) | Run ID: {run_id}', fontsize=12, fontweight='bold')
        plt.xlabel(f'PCA-1 ({pca.explained_variance_ratio_[0]*100:.1f}% phương sai)', fontsize=10)
        plt.ylabel(f'PCA-2 ({pca.explained_variance_ratio_[1]*100:.1f}% phương sai)', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        pca_plot_path = os.path.join(run_dir, "feature_space_pca.png")
        plt.savefig(pca_plot_path, dpi=300)
        plt.close()
        print(f"[*] Đã lưu biểu đồ không gian đặc trưng tại: {pca_plot_path}")
    except Exception as e:
        print(f"[!] Bỏ qua vẽ PCA do: {e}")

    # -------------------------------------------------------------
    # 4. MA TRẬN NHẦM LẪN: confusion_matrices.png
    # -------------------------------------------------------------
    if conf_matrices:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        for idx, (model_name, cm) in enumerate(conf_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[idx])
            axes[idx].set_title(f'Confusion Matrix: {model_name.upper()} (GA Subset)', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=9)
            axes[idx].set_ylabel('True Label', fontsize=9)
        plt.suptitle(f'Đánh giá Phân loại trên Tập Kiểm thử | Run ID: {run_id}', fontsize=13, y=0.98)
        plt.tight_layout()
        cm_plot_path = os.path.join(run_dir, "confusion_matrices.png")
        plt.savefig(cm_plot_path, dpi=300)
        plt.close()
        print(f"[*] Đã lưu ma trận nhầm lẫn tại: {cm_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Standalone GA Reduction Pipeline for Image Dataset")
    parser.add_argument("--generations", type=int, default=50, help="Số thế hệ GA")
    parser.add_argument("--pop_size", type=int, default=20, help="Kích thước quần thể")
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="Tỷ lệ lai ghép")
    parser.add_argument("--mutation_rate_max", type=float, default=0.03, help="Tỷ lệ đột biến max")
    parser.add_argument("--mutation_rate_min", type=float, default=0.001, help="Tỷ lệ đột biến min")
    parser.add_argument("--tournament_size", type=int, default=4, help="Kích thước Tournament Selection")
    parser.add_argument("--elitism_count", type=int, default=3, help="Số lượng cá thể tinh hoa giữ lại")
    parser.add_argument("--init_ratio", type=float, default=0.3, help="Tỷ lệ chọn mẫu ban đầu (~30%)")
    parser.add_argument("--patience", type=int, default=15, help="Số thế hệ dừng sớm nếu không cải thiện")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mini_test", type=int, default=None, help="Chạy thử nghiệm nhanh trên N mẫu ảnh")
    parser.add_argument("--run_name", type=str, default=None, help="Tên gợi nhớ cho lần chạy")
    parser.add_argument("--skip_eval", action="store_true", help="Bỏ qua bước train đánh giá 4 bộ phân loại ML")
    args = parser.parse_args()

    np.random.seed(args.seed)
    start_total_time = time.time()

    # Tạo Run ID và thư mục riêng biệt cho lần chạy
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"run_ga_{args.run_name}_" if args.run_name else "run_ga_"
    run_id = f"{prefix}{timestamp}"
    run_dir = os.path.join(config.paths.RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"==================================================")
    print(f"QUY TRÌNH RÚT GỌN DỮ LIỆU BẰNG GIẢI THUẬT DI TRUYỀN (GA)")
    print(f"Mã lần chạy (Run ID): {run_id}")
    print(f"Thư mục lưu kết quả : {run_dir}")
    print(f"Số thế hệ           : {args.generations}")
    print(f"Kích thước quần thể : {args.pop_size}")
    print(f"Seed                : {args.seed}")
    print(f"==================================================")

    # Nạp hoặc trích xuất đặc trưng
    train_feat_path = os.path.join(config.paths.FEATURES_DIR, "train_features.npy")
    if not os.path.exists(train_feat_path):
        print("[*] Không tìm thấy file đặc trưng, tiến hành trích xuất từ ResNet-18...")
        feature_extractor.main()

    print("[*] Đang nạp đặc trưng từ thư mục outputs/features/...")
    train_features = np.load(train_feat_path)
    train_labels = np.load(os.path.join(config.paths.FEATURES_DIR, "train_labels.npy"))
    test_features = np.load(os.path.join(config.paths.FEATURES_DIR, "test_features.npy"))
    test_labels = np.load(os.path.join(config.paths.FEATURES_DIR, "test_labels.npy"))

    # Chế độ mini_test
    if args.mini_test is not None:
        n_samples = min(args.mini_test, len(train_labels))
        print(f"[*] MINI TEST MODE: Lấy {n_samples} mẫu train ngẫu nhiên...")
        idx = np.random.choice(len(train_labels), size=n_samples, replace=False)
        train_features = train_features[idx]
        train_labels = train_labels[idx]

    N = len(train_labels)
    num_classes = config.num_classes
    classes = config.classes

    print(f"[*] Kích thước tập dữ liệu train: {N} mẫu | Số lớp: {num_classes}")

    # Tính ma trận khoảng cách Euclid theo đúng yêu cầu tài liệu
    print("[*] Đang tính ma trận khoảng cách Euclid (đa luồng CPU)...")
    dist_matrix = pairwise_distances(train_features, metric='euclidean', n_jobs=-1).astype(np.float32)
    dist_max = float(np.max(dist_matrix))
    print(f"[*] dist_max (Euclid) = {dist_max:.4f}")

    # Ước lượng Global Bounds
    print("[*] Đang ước lượng Global Bounds cho 4 hàm mục tiêu...")
    global_min, global_max = estimate_global_bounds(
        dist_matrix, train_labels, num_classes, dist_max, n_samples=config.fitness.global_bounds_samples
    )

    # Chạy GA
    ga = GeneticAlgorithmReducer(dist_matrix, train_labels, num_classes, global_min, global_max, dist_max, args)
    best_solution, best_fitness, history = ga.run()

    # Lưu file tập con .npy
    subset_path = os.path.join(run_dir, "best_subset_ga.npy")
    np.save(subset_path, best_solution)
    print(f"\n[*] Đã lưu nghiệm tập con tối ưu tại: {subset_path}")

    # Đánh giá độ nén
    M, compression_ratio = evaluate_subset(best_solution)
    print(f"[*] Kích thước tập rút gọn: {M} / {N} ảnh")
    print(f"[*] Tỷ lệ nén dữ liệu    : {compression_ratio * 100:.2f}% (Loại bỏ {N - M} ảnh)")

    # Lưu fitness_history.csv
    csv_history_path = os.path.join(run_dir, "fitness_history.csv")
    with open(csv_history_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'generation', 'best_fitness', 'mean_fitness', 
            'f_diversity', 'f_coverage', 'f_balance', 'f_compression', 'time_seconds'
        ])
        writer.writeheader()
        writer.writerows(history)
    print(f"[*] Đã lưu lịch sử Fitness tại: {csv_history_path}")

    # Đánh giá trên 4 bộ phân loại Downstream đã được loại bỏ theo yêu cầu

    # Vẽ toàn bộ biểu đồ (Fitness, Class Dist, PCA 2D)
    print("\n[*] Đang vẽ và lưu các biểu đồ báo cáo...")
    plot_all_results(history, train_features, train_labels, best_solution, classes, run_dir, run_id, conf_matrices=None)

    total_time = time.time() - start_total_time

    # Lưu run_metadata.json
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "total_time_seconds": round(total_time, 2),
        "command_args": vars(args),
        "results_summary": {
            "original_samples": N,
            "reduced_samples": int(M),
            "compression_ratio_percent": round(compression_ratio * 100, 2),
            "best_fitness": round(float(best_fitness), 4),
            "total_generations_run": len(history)
        }
    }
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"[*] Đã lưu metadata cấu hình tại: {metadata_path}")

    # Báo cáo tổng kết ra console
    print("\n=================================================================================")
    print(f"HOÀN THÀNH QUY TRÌNH GA REDUCTION TRONG {total_time:.1f} GIÂY")
    print(f"Mã lần chạy (Run ID)   : {run_id}")
    print(f"Thư mục chứa kết quả   : {run_dir}")
    print(f"Kích thước tập ảnh gốc : {N} mẫu")
    print(f"Kích thước tập rút gọn : {M} mẫu (Tỷ lệ nén: {compression_ratio * 100:.2f}%)")
    print(f"Điểm Fitness cao nhất  : {best_fitness:.4f}")
    print("=================================================================================")

if __name__ == "__main__":
    main()
