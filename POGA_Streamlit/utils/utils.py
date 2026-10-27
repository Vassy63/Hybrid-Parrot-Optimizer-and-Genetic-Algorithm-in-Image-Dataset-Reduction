import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import json
import csv
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    """Cố định seed để đảm bảo khả năng tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_model_params(model: nn.Module) -> Tuple[int, int]:
    """Trả về (tổng số tham số, số tham số có thể huấn luyện)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def compute_classification_metrics(
    y_true: List[int], 
    y_pred: List[int], 
    classes: List[str]
) -> Dict[str, Any]:
    """Tính Confusion Matrix, Precision, Recall, F1-Score thuần NumPy."""
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = {}
    macro_p, macro_r, macro_f1 = [], [], []
    total_samples = len(y_true)

    for i, cls_name in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls_name] = {
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "support": int(support)
        }
        macro_p.append(precision)
        macro_r.append(recall)
        macro_f1.append(f1)

    accuracy = np.trace(cm) / total_samples if total_samples > 0 else 0.0

    return {
        "confusion_matrix": cm,
        "accuracy": round(accuracy * 100, 2),
        "macro_precision": round(float(np.mean(macro_p)) * 100, 2),
        "macro_recall": round(float(np.mean(macro_r)) * 100, 2),
        "macro_f1": round(float(np.mean(macro_f1)) * 100, 2),
        "per_class": per_class,
        "total_samples": total_samples
    }

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    save_path: str,
    model_name: str = "",
    acc: float = 0.0
):
    """Vẽ và lưu ma trận nhầm lẫn (Confusion Matrix) dạng Heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    title = f"Confusion Matrix: {model_name.upper()} | Acc: {acc:.2f}%" if model_name else "Confusion Matrix"
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=10, fontweight='bold')

    plt.ylabel('True Label (Nhãn thực tế)', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Label (Nhãn dự đoán)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Đã lưu ma trận nhầm lẫn tại: {save_path}")

def save_classification_report(metrics: Dict[str, Any], save_path: str):
    """Lưu báo cáo phân loại chi tiết (Precision, Recall, F1) ra file CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision_percent", "recall_percent", "f1_score_percent", "support"])
        for cls_name, stats in metrics["per_class"].items():
            writer.writerow([cls_name, stats["precision"], stats["recall"], stats["f1_score"], stats["support"]])
        writer.writerow([])
        writer.writerow(["macro_avg", metrics["macro_precision"], metrics["macro_recall"], metrics["macro_f1"], metrics["total_samples"]])
        writer.writerow(["accuracy", "", "", metrics["accuracy"], metrics["total_samples"]])
    print(f"[*] Đã lưu báo cáo phân loại chi tiết tại: {save_path}")

def plot_class_distribution(
    class_counts: Dict[str, int], 
    save_path: str, 
    title: str = "Class Distribution"
):
    """Vẽ biểu đồ cột phân bố số lượng ảnh từng lớp."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(9, 5))
    bars = plt.bar(classes, counts, color='#2b5c8f', edgecolor='black', alpha=0.85)
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Lớp (Class)', fontsize=11, fontweight='bold')
    plt.ylabel('Số lượng ảnh (Images)', fontsize=11, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(counts)*0.01 if counts else 1), f"{int(yval):,}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Đã lưu biểu đồ phân bố lớp tại: {save_path}")

def plot_training_curves(
    history: Dict[str, List[float]], 
    epochs: int, 
    save_path: str, 
    run_id: str = "", 
    has_val: bool = True
):
    """Vẽ và lưu biểu đồ hàm mất mát (Loss) và độ chính xác (Accuracy)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(13, 5))
    best_acc = max(history['val_acc']) * 100 if has_val and history['val_acc'] else max(history['train_acc']) * 100
    
    # Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', marker='o', color='#1f77b4')
    if has_val and 'val_loss' in history and history['val_loss']:
        plt.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', marker='s', color='#ff7f0e')
    plt.title('Loss vs. Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), [acc * 100 for acc in history['train_acc']], label='Train Acc', marker='o', color='#2ca02c')
    if has_val and 'val_acc' in history and history['val_acc']:
        plt.plot(range(1, epochs + 1), [acc * 100 for acc in history['val_acc']], label='Val Acc', marker='s', color='#d62728')
    plt.title('Accuracy (%) vs. Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    title_text = f"Learning Curves ({run_id})" if run_id else "Learning Curves"
    if has_val:
        title_text += f" | Best Val Acc: {best_acc:.2f}%"
    else:
        title_text += f" | Best Train Acc: {best_acc:.2f}%"
    plt.suptitle(title_text, fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Đã lưu biểu đồ huấn luyện tại: {save_path}")

def save_run_metadata(
    run_dir: str, 
    run_id: str, 
    config_dict: Dict[str, Any], 
    summary_stats: Dict[str, Any]
):
    """Lưu metadata thông số cấu hình và thống kê kết quả vào file JSON."""
    os.makedirs(run_dir, exist_ok=True)
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    data = {
        "run_id": run_id,
        "config": config_dict,
        "summary": summary_stats
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[*] Đã lưu thông tin cấu hình và kết quả tại: {metadata_path}")

def save_training_log_csv(
    run_dir: str, 
    history: Dict[str, List[float]], 
    epoch_times: List[float]
):
    """Lưu chi tiết chỉ số từng epoch vào file CSV."""
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "training_log.csv")
    epochs = len(history['train_loss'])
    has_val = 'val_loss' in history and len(history['val_loss']) > 0
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["epoch", "train_loss", "train_acc"]
        if has_val:
            header.extend(["val_loss", "val_acc"])
        header.append("time_seconds")
        writer.writerow(header)
        
        for i in range(epochs):
            row = [
                i + 1,
                f"{history['train_loss'][i]:.4f}",
                f"{history['train_acc'][i]:.4f}"
            ]
            if has_val:
                row.extend([
                    f"{history['val_loss'][i]:.4f}",
                    f"{history['val_acc'][i]:.4f}"
                ])
            row.append(f"{epoch_times[i]:.2f}")
            writer.writerow(row)
    print(f"[*] Đã lưu chi tiết log từng epoch tại: {csv_path}")

def generate_consolidated_report(
    benchmark_results: List[Dict[str, Any]], 
    classes: List[str], 
    save_dir: str
):
    """
    Xuất kết quả đối sánh sạch sẽ, trực quan và chuẩn hóa:
    1. benchmark_summary.csv: Bảng đối sánh tổng quan các mô hình.
    2. per_class_f1.csv: Bảng điểm F1 từng lớp theo từng mô hình.
    3. comparison_curves.png: Đồ thị so sánh trực tiếp Acc & Loss giữa các mô hình qua các Epoch.
    4. confusion_matrices.png: Các ma trận nhầm lẫn xếp ngang rõ nét.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_models = len(benchmark_results)
    if n_models == 0:
        return

    # 1. BẢNG 1: benchmark_summary.csv (Chuẩn bảng hình chữ nhật)
    summary_csv_path = os.path.join(save_dir, "benchmark_summary.csv")
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", 
            "params_million", 
            "test_accuracy_percent",
            "val_accuracy_percent", 
            "macro_f1_percent", 
            "train_accuracy_percent", 
            "latency_ms_per_image", 
            "training_time_seconds"
        ])
        for r in benchmark_results:
            writer.writerow([
                r["model"],
                r.get("params_m", "N/A"),
                r.get("test_acc", "N/A"),
                r.get("best_val_acc", "N/A"),
                r.get("macro_f1", "N/A"),
                r.get("final_train_acc", "N/A"),
                r.get("latency_ms", "N/A"),
                r.get("total_time_seconds", "N/A")
            ])
    print(f"[*] Đã lưu bảng tóm tắt đối sánh tại: {summary_csv_path}")

    # 2. BẢNG 2: per_class_f1.csv (Chuẩn bảng điểm F1 từng lớp)
    f1_csv_path = os.path.join(save_dir, "per_class_f1.csv")
    with open(f1_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["class_name"] + [r['model'] for r in benchmark_results]
        writer.writerow(header)
        
        for cls_name in classes:
            row = [cls_name]
            for r in benchmark_results:
                f1_val = r.get("test_metrics", {}).get("per_class", {}).get(cls_name, {}).get("f1_score", "N/A")
                row.append(f1_val)
            writer.writerow(row)
            
        macro_row = ["macro_avg"] + [r.get("macro_f1", "N/A") for r in benchmark_results]
        writer.writerow(macro_row)
    print(f"[*] Đã lưu bảng F1 từng lớp tại: {f1_csv_path}")

    # 3. ẢNH 1: comparison_curves.png (Đồ thị Loss & Accuracy đối sánh giữa các mô hình)
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    curves_png_path = os.path.join(plots_dir, "comparison_curves.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, r in enumerate(benchmark_results):
        model_name = r["model"]
        history = r["history"]
        epochs = len(history["train_acc"])
        color = colors[idx % len(colors)]

        # Đường Validation Acc
        if "val_acc" in history and history["val_acc"]:
            ax1.plot(range(1, epochs + 1), [a * 100 for a in history["val_acc"]], 
                     label=f"{model_name} (Val: {r.get('best_val_acc')}%)", 
                     marker='s', linewidth=2, color=color)
        else:
            ax1.plot(range(1, epochs + 1), [a * 100 for a in history["train_acc"]], 
                     label=f"{model_name} (Train)", marker='o', linewidth=2, color=color)

        # Đường Train Loss
        ax2.plot(range(1, epochs + 1), history["train_loss"], 
                 label=f"{model_name}", marker='o', linewidth=2, color=color)

    ax1.set_title("Validation Accuracy vs. Epochs", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_title("Training Loss vs. Epochs", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Model Learning Convergence Comparison (Full Dataset)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(curves_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Đã lưu biểu đồ hội tụ đối sánh tại: {curves_png_path}")

    # 4. ẢNH 2: confusion_matrices.png (Các ma trận nhầm lẫn xếp ngang rõ nét, tỉ lệ chuẩn vuông)
    cm_png_path = os.path.join(plots_dir, "confusion_matrices.png")
    fig_width = max(6.5 * n_models, 7)
    fig, axes = plt.subplots(1, n_models, figsize=(fig_width, 6.2))
    if n_models == 1:
        axes = [axes]

    for col_idx, r in enumerate(benchmark_results):
        ax = axes[col_idx]
        model_name = r["model"].upper()
        cm = r.get("test_metrics", {}).get("confusion_matrix")
        test_acc = r.get("test_acc", r.get("best_val_acc", 0.0))

        if cm is not None:
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f"{model_name}\nTest Acc: {test_acc}%", fontsize=12, fontweight='bold', pad=12)
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9.5)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(classes, fontsize=9.5)

            thresh = cm.max() / 2.0 if cm.max() > 0 else 1
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    val = cm[i, j]
                    color = "white" if val > thresh else "black"
                    ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=9.5, fontweight='bold')

            ax.set_ylabel('True Label', fontsize=10.5, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=10.5, fontweight='bold')

    plt.suptitle("Confusion Matrices Comparison (Independent Test Set)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Đã lưu ma trận nhầm lẫn đối sánh tại: {cm_png_path}")
