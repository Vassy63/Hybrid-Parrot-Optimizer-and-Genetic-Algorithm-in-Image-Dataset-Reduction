import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import copy
import time
import csv
import json
import math
import datetime
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from configs import BaselineConfig, POGAConfig
from data import create_dataloaders, extract_and_cache_features
from models.model import SUPPORTED_MODELS
from train import train_single_model
from utils import set_seed
from optimization import (
    precompute_distance_matrix,
    compute_raw_fitness,
    fitness,
    repair_solution,
    GeneticAlgorithm,
    ParrotOptimizer,
    run_s1_alternating_ga_po,
    run_s2_alternating_po_ga,
    run_s3_block_ga_po,
    run_s4_block_po_ga,
    run_p1_parallel_merge,
    run_c1_cooperative_exchange,
    random_selection,
    stratified_random,
)

# المرجع chuẩn Full Dataset (Upper Bound B0) đã chạy thực nghiệm trên 11,228 Train / 2,806 Val / 3,000 Test
FULL_DATASET_BASELINE = {
    "resnet18": {"test_acc": 90.27, "val_acc": 90.34, "macro_precision": 90.38, "macro_recall": 90.46, "macro_f1": 90.41, "time_s": 265.16},
    "mobilenet_v3": {"test_acc": 88.93, "val_acc": 90.34, "macro_precision": 89.26, "macro_recall": 89.28, "macro_f1": 89.25, "time_s": 174.44},
    "densenet121": {"test_acc": 90.57, "val_acc": 91.80, "macro_precision": 90.71, "macro_recall": 90.76, "macro_f1": 90.72, "time_s": 301.34},
}


def wilcoxon_two_sided(x: np.ndarray, y: np.ndarray) -> float:
    """Tính p-value kiểm định cặp có dấu Wilcoxon Signed-Rank Test (hỗ trợ cả scipy hoặc thuần toán học)."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.abs(diff) > 1e-12]
    n = len(diff)
    if n == 0:
        return 1.0
    try:
        from scipy.stats import wilcoxon
        _, p_val = wilcoxon(x, y, alternative='two-sided')
        return float(p_val)
    except Exception:
        abs_d = np.abs(diff)
        order = np.argsort(abs_d)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, n + 1, dtype=float)
        w_plus = float(np.sum(ranks[diff > 0]))
        w_minus = float(np.sum(ranks[diff < 0]))
        w = min(w_plus, w_minus)
        mean_w = n * (n + 1) / 4.0
        std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        if std_w < 1e-12:
            return 1.0
        z = abs(w - mean_w) / std_w
        return float(math.erfc(z / math.sqrt(2.0)))


def plot_convergence_curves(convergence_rows, save_path: str):
    """Vẽ biểu đồ so sánh tốc độ hội tụ Fitness giữa các kiến trúc."""
    if not convergence_rows:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    methods = []
    for r in convergence_rows:
        if r["Method"] not in methods:
            methods.append(r["Method"])

    plt.figure(figsize=(12, 6.5))
    styles = {
        "GA-only": ("gray", "--", 1.6),
        "PO-only": ("black", ":", 1.6),
        "S1 (Alt GA-PO)": ("#1f77b4", "-", 2.0),
        "S2 (Alt PO-GA)": ("#ff7f0e", "-", 2.0),
        "S3 (Block GA-PO)": ("#2ca02c", "-.", 1.8),
        "S4 (Block PO-GA)": ("#9467bd", "-.", 1.8),
        "P1 (Parallel Merge)": ("#8c564b", "--", 2.0),
        "C1 (Coop Exchange)": ("#d62728", "-", 2.5),
    }
    for m in methods:
        m_rows = [r for r in convergence_rows if r["Method"] == m]
        max_iter = max(r["Iteration"] for r in m_rows)
        xs, means, stds = [], [], []
        for it in range(1, max_iter + 1):
            vals = [r["Fitness_Total"] for r in m_rows if r["Iteration"] == it]
            if vals:
                xs.append(it)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
        color, ls, lw = styles.get(m, (None, "-", 2.0))
        xs_arr, m_arr, s_arr = np.array(xs), np.array(means), np.array(stds)
        plt.plot(xs_arr, m_arr, label=m, color=color, linestyle=ls, linewidth=lw)
        if np.any(s_arr > 0):
            plt.fill_between(xs_arr, m_arr - s_arr, m_arr + s_arr, color=color, alpha=0.12)

    plt.title("So sánh Tốc độ Hội tụ: Sequential (S1–S4) vs Parallel (P1) vs Cooperative (C1)", fontsize=13, fontweight="bold")
    plt.xlabel("Vòng lặp / Thế hệ tương đương (Equal FEs Budget)", fontsize=11)
    plt.ylabel("Điểm Thích nghi Tổng hợp (Fitness Score)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9.5, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[*] Đã lưu biểu đồ hội tụ tại: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Khung Thực nghiệm Hybrid PO-GA Rút gọn Tập dữ liệu Ảnh & Đánh giá trên ResNet-18, MobileNetV3, DenseNet-121"
    )
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "s1", "s2", "s3", "s4", "p1", "c1", "ga", "po", "random", "stratified"],
                        help="Phương pháp rút gọn dữ liệu (mặc định: 'all' chạy toàn bộ S1-S4, P1, C1, GA, PO, Random, Stratified)")
    parser.add_argument("--model", type=str, default="all",
                        choices=SUPPORTED_MODELS + ["all"],
                        help="Mô hình CNN dùng để đánh giá tập con: resnet18, mobilenet_v3, densenet121 hoặc 'all'")
    parser.add_argument("--population_size", type=int, default=20, help="Kích thước quần thể GA/PO (mặc định: 20)")
    parser.add_argument("--total_iterations", type=int, default=100, help="Tổng số thế hệ/vòng lặp (mặc định: 100 -> 2,000 FEs)")
    parser.add_argument("--phase_size", type=int, default=10, help="Độ dài mỗi pha xen kẽ S1/S2 hoặc trao đổi C1")
    parser.add_argument("--exchange_k", type=int, default=3, help="Số cá thể ưu tú trao đổi trong C1")
    parser.add_argument("--epochs", type=int, default=5, help="Số epochs huấn luyện CNN đánh giá tập con")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size huấn luyện CNN")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Danh sách hạt giống thực nghiệm (ví dụ: --seeds 42 123 456 789 2026)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Giới hạn số lượng ảnh để kiểm tra nhanh (Fast / Mini-test mode)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Chỉ chạy tối ưu hóa PO-GA và xuất tập con + biểu đồ hội tụ, bỏ qua bước huấn luyện CNN")
    parser.add_argument("--train_dir", type=str, default="data/seg_train/seg_train")
    parser.add_argument("--test_dir", type=str, default="data/seg_test/seg_test")
    parser.add_argument("--save_dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"run_poga_{args.method}_{timestamp}"

    base_cfg = BaselineConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=42,
        run_id=session_id
    )
    session_dir = base_cfg.run_dir
    subsets_dir = os.path.join(session_dir, "subsets")
    plots_dir = os.path.join(session_dir, "plots")
    os.makedirs(subsets_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    poga_cfg = POGAConfig()
    poga_cfg.ga.population_size = args.population_size
    poga_cfg.po.population_size = args.population_size
    poga_cfg.total_iterations = args.total_iterations
    poga_cfg.phase_size = args.phase_size
    poga_cfg.exchange_k = args.exchange_k

    print("=" * 90)
    print("KHUNG THỰC NGHIỆM HYBRID PO-GA RÚT GỌN DỮ LIỆU & ĐÁNH GIÁ CNN (CNTT-KLCN140)")
    print(f"Thư mục phiên chạy : {session_dir}")
    print(f"Thiết bị           : {base_cfg.device}")
    print(f"Phương pháp        : {args.method.upper()} | Models: {args.model.upper()} | Seeds: {args.seeds}")
    print("Hàm Fitness        : F = 0.1*Div + 0.3*Cov + 0.2*Bal + 0.4*Com (4 thành phần nguyên bản)")
    print("=" * 90)

    # 1. Trích xuất đặc trưng ResNet-18 (512 chiều) và chuẩn hóa ma trận khoảng cách Cosine
    features, labels = extract_and_cache_features(base_cfg)
    N = len(labels)
    poga_cfg.num_classes = len(np.unique(labels))
    print(f"[*] Đang tính ma trận khoảng cách Cosine chuẩn hóa ({N} x {N})...")
    t0 = time.time()
    dist_matrix = precompute_distance_matrix(features, metric="cosine")
    print(f"[*] Hoàn thành ma trận khoảng cách trong {time.time() - t0:.2f}s.")

    all_opt_map = {
        "S1 (Alt GA-PO)": "s1",
        "S2 (Alt PO-GA)": "s2",
        "S3 (Block GA-PO)": "s3",
        "S4 (Block PO-GA)": "s4",
        "P1 (Parallel Merge)": "p1",
        "C1 (Coop Exchange)": "c1",
        "GA-only": "ga",
        "PO-only": "po",
    }
    if args.method == "all":
        selected_opt = all_opt_map
        run_baselines = True
    elif args.method in ("random", "stratified"):
        selected_opt = {}
        run_baselines = True
    else:
        key_to_name = {v: k for k, v in all_opt_map.items()}
        selected_opt = {key_to_name[args.method]: args.method}
        run_baselines = False

    # 2. Khởi tạo quần thể ban đầu cố định cho từng seed để đảm bảo công bằng tuyệt đối
    ga_helper = GeneticAlgorithm(poga_cfg, dist_matrix, labels, poga_cfg.num_classes)
    seed_init_pops = {}
    for s in args.seeds:
        rng = np.random.RandomState(s)
        np.random.seed(s)
        raw_pop = (rng.rand(poga_cfg.ga.population_size, N) < poga_cfg.ga.init_ratio).astype(int)
        ipop = np.array([repair_solution(ind) for ind in raw_pop])
        ifits = ga_helper.evaluate_population(ipop)
        seed_init_pops[s] = (ipop, ifits)

    models_to_eval = SUPPORTED_MODELS if args.model == "all" else [args.model]
    convergence_rows = []
    experiment_rows = []
    subset_sizes = []

    # Ghi nhận mốc chuẩn Full Dataset (B0) vào bảng kết quả
    if not args.skip_eval:
        for m_name in models_to_eval:
            b0 = FULL_DATASET_BASELINE[m_name]
            for r_idx, s in enumerate(args.seeds, 1):
                experiment_rows.append({
                    "Method": "Full Dataset (B0)", "Run": r_idx, "Seed": s, "Model": m_name,
                    "Subset_Size_M": N, "DR_Percent": 0.0,
                    "Test_Acc": b0["test_acc"], "Val_Acc": b0["val_acc"],
                    "Macro_Precision": b0["macro_precision"], "Macro_Recall": b0["macro_recall"], "Macro_F1": b0["macro_f1"],
                    "ARR_Percent": 100.0, "Delta_Acc": 0.0,
                    "Train_Time_Sec": b0["time_s"], "Speedup": 1.0, "Opt_Time_Sec": 0.0,
                    "Fitness_Total": 0.0, "Fitness_Div": 0.0, "Fitness_Cov": 1.0, "Fitness_Bal": 1.0, "Fitness_Com": 0.0
                })

    def evaluate_and_record(method_label: str, m_key: str, run_idx: int, seed_val: int, best_sol: np.ndarray, opt_time: float):
        M = int(np.sum(best_sol))
        dr_pct = round((1.0 - M / N) * 100.0, 2)
        raw_f = compute_raw_fitness(best_sol, dist_matrix, labels, poga_cfg.num_classes)
        fit_tot = fitness(best_sol, dist_matrix, labels, poga_cfg.num_classes, poga_cfg.fitness)

        subset_file = os.path.join(subsets_dir, f"subset_{m_key}_seed{seed_val}.npy")
        np.save(subset_file, best_sol, allow_pickle=False)

        if args.skip_eval:
            experiment_rows.append({
                "Method": method_label, "Run": run_idx, "Seed": seed_val, "Model": "optimization_only",
                "Subset_Size_M": M, "DR_Percent": dr_pct,
                "Test_Acc": np.nan, "Val_Acc": np.nan,
                "Macro_Precision": np.nan, "Macro_Recall": np.nan, "Macro_F1": np.nan,
                "ARR_Percent": np.nan, "Delta_Acc": np.nan,
                "Train_Time_Sec": 0.0, "Speedup": np.nan, "Opt_Time_Sec": round(opt_time, 2),
                "Fitness_Total": round(fit_tot, 4), "Fitness_Div": round(float(raw_f[0]), 4),
                "Fitness_Cov": round(float(raw_f[1]), 4), "Fitness_Bal": round(float(raw_f[2]), 4),
                "Fitness_Com": round(float(raw_f[3]), 4)
            })
            return

        # Tạo DataLoader từ tập con đã chọn
        sub_cfg = copy.copy(base_cfg)
        sub_cfg.subset_indices_path = subset_file
        sub_cfg.seed = 42
        train_loader, val_loader, test_loader, classes = create_dataloaders(sub_cfg)

        for m_name in models_to_eval:
            set_seed(42)
            m_cfg = copy.copy(sub_cfg)
            m_cfg.model_name = m_name
            m_cfg.run_id = f"{m_key}_seed{seed_val}_{m_name}"
            m_cfg.run_dir = os.path.join(session_dir, "model_runs", f"{m_key}_seed{seed_val}")
            os.makedirs(m_cfg.run_dir, exist_ok=True)

            res = train_single_model(m_cfg, train_loader, val_loader, test_loader, classes)
            t_metrics = res["test_metrics"]
            test_acc = float(res["test_acc"])
            val_acc = float(res["best_val_acc"]) if res["best_val_acc"] is not None else test_acc
            prec = float(t_metrics.get("macro_precision", 0.0))
            rec = float(t_metrics.get("macro_recall", 0.0))
            f1 = float(res["macro_f1"])
            t_train = float(res["total_time_seconds"])

            b0 = FULL_DATASET_BASELINE[m_name]
            arr = round((test_acc / max(1e-8, b0["test_acc"])) * 100.0, 2)
            delta_acc = round(test_acc - b0["test_acc"], 2)
            speedup = round(b0["time_s"] / max(1e-4, t_train), 2)

            experiment_rows.append({
                "Method": method_label, "Run": run_idx, "Seed": seed_val, "Model": m_name,
                "Subset_Size_M": M, "DR_Percent": dr_pct,
                "Test_Acc": test_acc, "Val_Acc": val_acc,
                "Macro_Precision": prec, "Macro_Recall": rec, "Macro_F1": f1,
                "ARR_Percent": arr, "Delta_Acc": delta_acc,
                "Train_Time_Sec": t_train, "Speedup": speedup, "Opt_Time_Sec": round(opt_time, 2),
                "Fitness_Total": round(fit_tot, 4), "Fitness_Div": round(float(raw_f[0]), 4),
                "Fitness_Cov": round(float(raw_f[1]), 4), "Fitness_Bal": round(float(raw_f[2]), 4),
                "Fitness_Com": round(float(raw_f[3]), 4)
            })

    # 3. Chạy các phương pháp tối ưu hóa (S1-S4, P1, C1, GA, PO)
    for method_label, m_key in selected_opt.items():
        print(f"\n==================== {method_label} ====================")
        for r_idx, s in enumerate(args.seeds, 1):
            np.random.seed(s)
            ipop, ifits = seed_init_pops[s]
            ipop, ifits = ipop.copy(), ifits.copy()

            t_start = time.time()
            if m_key == "s1":
                best_sol, _, hist = run_s1_alternating_ga_po(poga_cfg, dist_matrix, labels, poga_cfg.num_classes, ipop, ifits)
            elif m_key == "s2":
                best_sol, _, hist = run_s2_alternating_po_ga(poga_cfg, dist_matrix, labels, poga_cfg.num_classes, ipop, ifits)
            elif m_key == "s3":
                best_sol, _, hist = run_s3_block_ga_po(poga_cfg, dist_matrix, labels, poga_cfg.num_classes, ipop, ifits)
            elif m_key == "s4":
                best_sol, _, hist = run_s4_block_po_ga(poga_cfg, dist_matrix, labels, poga_cfg.num_classes, ipop, ifits)
            elif m_key == "p1":
                best_sol, _, hist = run_p1_parallel_merge(poga_cfg, dist_matrix, labels, poga_cfg.num_classes, ipop, ifits)
            elif m_key == "c1":
                best_sol, _, hist = run_c1_cooperative_exchange(poga_cfg, dist_matrix, labels, poga_cfg.num_classes, ipop, ifits)
            elif m_key == "ga":
                ga = GeneticAlgorithm(poga_cfg, dist_matrix, labels, poga_cfg.num_classes)
                pop_out, _, hist = ga.run(ipop, ifits, steps=poga_cfg.total_iterations, total_gens=poga_cfg.total_iterations, disable_early_stop=True)
                best_sol = pop_out[0]
            elif m_key == "po":
                po = ParrotOptimizer(poga_cfg, dist_matrix, labels, poga_cfg.num_classes)
                best_sol, _, hist = po.run(ipop, ifits, steps=poga_cfg.total_iterations, total_iters=poga_cfg.total_iterations, disable_early_stop=True)
            opt_time = time.time() - t_start

            subset_sizes.append(int(np.sum(best_sol)))
            for it_idx, h_item in enumerate(hist, 1):
                convergence_rows.append({
                    "Method": method_label, "Run": r_idx, "Seed": s, "Iteration": it_idx,
                    "Fitness_Total": h_item["total"], "Fitness_Div": h_item["div"],
                    "Fitness_Cov": h_item["cov"], "Fitness_Bal": h_item["bal"], "Fitness_Com": h_item["com"]
                })

            evaluate_and_record(method_label, m_key, r_idx, s, best_sol, opt_time)

    # 4. Chạy các Baseline lấy mẫu ngẫu nhiên (B1: Random Selection, B2: Stratified Random)
    if run_baselines:
        target_m = int(np.median(subset_sizes)) if subset_sizes else int(round(N * poga_cfg.ga.init_ratio))
        b_map = {}
        if args.method in ("all", "random"):
            b_map["Random Selection (B1)"] = ("random", lambda s: random_selection(N, target_m, seed=s))
        if args.method in ("all", "stratified"):
            b_map["Stratified Random (B2)"] = ("stratified", lambda s: stratified_random(labels, target_m, seed=s))

        for b_label, (b_key, b_fn) in b_map.items():
            print(f"\n==================== {b_label} (M={target_m}) ====================")
            for r_idx, s in enumerate(args.seeds, 1):
                t_start = time.time()
                best_sol = b_fn(s)
                opt_time = time.time() - t_start
                evaluate_and_record(b_label, b_key, r_idx, s, best_sol, opt_time)

    # 5. Lưu kết quả CSV, vẽ biểu đồ hội tụ và xuất báo cáo thống kê (Mean±Std, Median, IQR, Wilcoxon)
    exp_csv = os.path.join(session_dir, "experiment_results.csv")
    if experiment_rows:
        with open(exp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(experiment_rows[0].keys()))
            writer.writeheader()
            writer.writerows(experiment_rows)
        print(f"\n[*] Đã lưu bảng kết quả thực nghiệm tại: {exp_csv}")

    if convergence_rows:
        conv_csv = os.path.join(session_dir, "convergence_histories.csv")
        with open(conv_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(convergence_rows[0].keys()))
            writer.writeheader()
            writer.writerows(convergence_rows)
        plot_convergence_curves(convergence_rows, os.path.join(plots_dir, "convergence_comparison.png"))

    # Tính thống kê mô tả & Wilcoxon
    desc_rows = []
    methods_present = []
    for r in experiment_rows:
        if r["Method"] not in methods_present:
            methods_present.append(r["Method"])
    models_present = []
    for r in experiment_rows:
        if r["Model"] not in models_present:
            models_present.append(r["Model"])

    for m in methods_present:
        for mdl in models_present:
            sub = [r for r in experiment_rows if r["Method"] == m and r["Model"] == mdl]
            if not sub:
                continue
            accs = np.array([r["Test_Acc"] for r in sub], dtype=float)
            f1s = np.array([r["Macro_F1"] for r in sub], dtype=float)
            fits = np.array([r["Fitness_Total"] for r in sub], dtype=float)
            drs = np.array([r["DR_Percent"] for r in sub], dtype=float)
            arrs = np.array([r["ARR_Percent"] for r in sub], dtype=float)
            spds = np.array([r["Speedup"] for r in sub], dtype=float)

            q75_a, q25_a = np.nanpercentile(accs, [75, 25]) if not np.all(np.isnan(accs)) else (np.nan, np.nan)
            q75_f, q25_f = np.nanpercentile(fits, [75, 25]) if not np.all(np.isnan(fits)) else (np.nan, np.nan)

            desc_rows.append({
                "Method": m,
                "Model": mdl,
                "DR_Percent": round(float(np.nanmean(drs)), 2),
                "Fitness_Mean_Std": f"{np.nanmean(fits):.4f} ± {np.nanstd(fits):.4f}",
                "Fitness_Median_IQR": f"{np.nanmedian(fits):.4f} [{q75_f - q25_f:.4f}]",
                "Test_Acc_Mean_Std": f"{np.nanmean(accs):.2f} ± {np.nanstd(accs):.2f}" if not np.all(np.isnan(accs)) else "N/A",
                "Test_Acc_Median_IQR": f"{np.nanmedian(accs):.2f} [{q75_a - q25_a:.2f}]" if not np.all(np.isnan(accs)) else "N/A",
                "Macro_F1_Mean_Std": f"{np.nanmean(f1s):.2f} ± {np.nanstd(f1s):.2f}" if not np.all(np.isnan(f1s)) else "N/A",
                "ARR_Percent": round(float(np.nanmean(arrs)), 2) if not np.all(np.isnan(arrs)) else np.nan,
                "Speedup": round(float(np.nanmean(spds)), 2) if not np.all(np.isnan(spds)) else np.nan,
            })

    if desc_rows:
        desc_csv = os.path.join(session_dir, "descriptive_statistics.csv")
        with open(desc_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(desc_rows[0].keys()))
            writer.writeheader()
            writer.writerows(desc_rows)
        print(f"[*] Đã lưu bảng thống kê (Mean±Std, Median, IQR) tại: {desc_csv}")

        # Xuất báo cáo kiểm định Wilcoxon Signed-Rank Test
        preferred_targets = ["C1 (Coop Exchange)", "P1 (Parallel Merge)", "S1 (Alt GA-PO)", "S3 (Block GA-PO)"]
        target_m = next((m for m in preferred_targets if m in methods_present), methods_present[0])
        md_lines = [
            "# Báo cáo Thống kê Thực nghiệm & Kiểm định Wilcoxon Signed-Rank Test",
            f"\n* **Thư mục phiên chạy:** `{session_dir}`",
            f"* **Phương pháp đối chứng (Target):** `{target_m}`\n",
            "## 1. Thống kê Mô tả Đa Seed (Mean ± Std, Median [IQR])",
            "| Phương pháp | Mô hình | DR (%) | Fitness (Mean ± Std) | Fitness (Median [IQR]) | Test Acc (Mean ± Std %) | Test Acc (Median [IQR] %) | Macro F1 (%) | ARR (%) | Speedup (x) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]
        for r in desc_rows:
            md_lines.append(
                f"| {r['Method']} | {r['Model']} | {r['DR_Percent']} | {r['Fitness_Mean_Std']} | {r['Fitness_Median_IQR']} | "
                f"{r['Test_Acc_Mean_Std']} | {r['Test_Acc_Median_IQR']} | {r['Macro_F1_Mean_Std']} | {r['ARR_Percent']} | {r['Speedup']} |"
            )

        competitors = [m for m in methods_present if m != target_m and m != "Full Dataset (B0)"]
        if competitors:
            md_lines.append(f"\n## 2. Kiểm định Wilcoxon Signed-Rank Test (p-value so với `{target_m}`)")
            md_lines.append("| Phương pháp đối sánh | " + " | ".join(models_present) + " |")
            md_lines.append("| :--- | " + " | ".join([":---:" for _ in models_present]) + " |")
            for comp in competitors:
                cells = []
                for mdl in models_present:
                    metric_key = "Fitness_Total" if mdl == "optimization_only" else "Test_Acc"
                    v_tar = [r[metric_key] for r in experiment_rows if r["Method"] == target_m and r["Model"] == mdl]
                    v_cmp = [r[metric_key] for r in experiment_rows if r["Method"] == comp and r["Model"] == mdl]
                    k_len = min(len(v_tar), len(v_cmp))
                    if k_len < 2:
                        cells.append("N/A")
                    else:
                        pval = wilcoxon_two_sided(np.array(v_tar[:k_len]), np.array(v_cmp[:k_len]))
                        star = " *" if pval < 0.05 else ""
                        cells.append(f"{pval:.4f}{star}")
                md_lines.append(f"| {comp} | " + " | ".join(cells) + " |")

        stat_md_path = os.path.join(session_dir, "statistical_significance.md")
        with open(stat_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"[*] Đã lưu báo cáo kiểm định Wilcoxon tại: {stat_md_path}")

    print("\n" + "=" * 90)
    print(f"HOÀN TẤT TOÀN BỘ THỰC NGHIỆM! Kết quả được lưu tại: {session_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
