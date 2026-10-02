import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import copy
import json
import csv
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from configs import BaselineConfig
from data import create_dataloaders
from models import build_baseline_model
from models.model import SUPPORTED_MODELS
from training import BaselineTrainer
from utils import (
    set_seed, 
    plot_training_curves, 
    plot_class_distribution, 
    generate_consolidated_report,
    plot_confusion_matrix
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full Dataset Training & Evaluation (ResNet18, MobileNetV3, DenseNet121)"
    )
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=SUPPORTED_MODELS + ["all"],
                        help=f"Chọn mô hình huấn luyện: {SUPPORTED_MODELS} hoặc 'all' để chạy toàn bộ")
    parser.add_argument("--train_dir", type=str, default="data/seg_train/seg_train",
                        help="Đường dẫn đến thư mục ảnh train")
    parser.add_argument("--test_dir", type=str, default="data/seg_test/seg_test",
                        help="Đường dẫn đến thư mục ảnh test độc lập")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Tỷ lệ tách tập validation từ seg_train (mặc định: 0.2)")
    parser.add_argument("--save_dir", type=str, default="outputs",
                        help="Thư mục gốc lưu các runs")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Tên định danh tùy chọn cho lần chạy")
    parser.add_argument("--epochs", type=int, default=5, help="Số epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch (mặc định: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--fine_tune", action="store_true", 
                        help="Mở khóa tất cả các tầng để fine-tune toàn bộ mạng")
    parser.add_argument("--num_samples", type=int, default=None, 
                        help="Chạy nhanh trên số lượng ảnh mẫu nhỏ")
    parser.add_argument("--subset_indices_path", type=str, default=None,
                        help="Đường dẫn tới file .npy chứa tập con rút gọn (mask nhị phân hoặc danh sách chỉ số)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def train_single_model(config: BaselineConfig, train_loader, val_loader, test_loader, classes):
    """Huấn luyện và đánh giá 1 mô hình đơn lẻ."""
    print(f"\n==================================================")
    print(f"BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH: {config.model_name.upper()}")
    print(f"Device           : {config.device}")
    print(f"Epochs           : {config.epochs}")
    print(f"Batch size       : {config.batch_size}")
    print(f"Fine-tune        : {config.fine_tune}")
    print(f"==================================================")

    num_classes = len(classes)
    model = build_baseline_model(
        model_name=config.model_name,
        num_classes=num_classes,
        fine_tune=config.fine_tune
    )

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.lr, weight_decay=config.weight_decay)

    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        classes=classes
    )

    history, test_metrics = trainer.fit()

    # 1. Lưu biểu đồ Loss/Accuracy riêng của mô hình vào plots/
    plots_dir = os.path.join(config.run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    curve_path = os.path.join(plots_dir, f"{config.model_name}_curve.png")
    plot_training_curves(
        history=history,
        epochs=config.epochs,
        save_path=curve_path,
        run_id=config.model_name.upper(),
        has_val=(val_loader is not None)
    )

    # 2. Lưu Ma trận nhầm lẫn riêng của mô hình vào plots/
    cm = test_metrics.get("confusion_matrix")
    test_acc = test_metrics.get("accuracy", round(trainer.best_val_acc * 100, 2))
    if cm is not None:
        cm_path = os.path.join(plots_dir, f"{config.model_name}_confusion_matrix.png")
        plot_confusion_matrix(
            cm=cm,
            classes=classes,
            save_path=cm_path,
            model_name=config.model_name,
            acc=test_acc
        )

    return {
        "model": config.model_name,
        "params_m": round(trainer.total_params / 1e6, 2),
        "best_val_acc": round(trainer.best_val_acc * 100, 2) if val_loader else None,
        "test_acc": test_acc,
        "macro_f1": test_metrics.get("macro_f1"),
        "final_train_acc": round(history['train_acc'][-1] * 100, 2),
        "latency_ms": test_metrics.get("avg_inference_latency_ms"),
        "total_time_seconds": round(sum(trainer.epoch_times), 2),
        "run_dir": config.run_dir,
        "history": history,
        "test_metrics": test_metrics
    }

def main():
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"run_{args.run_name}_" if args.run_name else f"run_{args.model}_"
    session_run_id = f"{prefix}{timestamp}"
    session_run_dir = os.path.join(args.save_dir, session_run_id)
    os.makedirs(session_run_dir, exist_ok=True)

    print(f"==================================================")
    print(f"THỰC NGHIỆM ĐỐI SÁNH TRÊN TẬP GỐC (FULL DATASET)")
    print(f"Mã phiên chạy (Session ID): {session_run_id}")
    print(f"Thư mục lưu DUY NHẤT      : {session_run_dir}")
    print(f"Mô hình lựa chọn          : {args.model.upper()}")
    print(f"Số epochs                 : {args.epochs}")
    print(f"Batch size                : {args.batch_size}")
    print(f"==================================================")

    base_config = BaselineConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        val_split=args.val_split,
        save_dir=args.save_dir,
        model_name=args.model if args.model != "all" else "resnet18",
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fine_tune=args.fine_tune,
        num_samples=args.num_samples,
        subset_indices_path=args.subset_indices_path,
        seed=args.seed,
        run_id=session_run_id,
        run_dir=session_run_dir
    )

    # Nạp dữ liệu 1 lần dùng chung cho tất cả các mô hình
    train_loader, val_loader, test_loader, classes = create_dataloaders(base_config)

    # Lưu biểu đồ phân bố lớp ngay vào trong session_run_dir
    class_counts = {}
    if hasattr(train_loader.dataset, 'targets'):
        targets = train_loader.dataset.targets
        for i, cls_name in enumerate(classes):
            class_counts[cls_name] = sum(1 for t in targets if t == i)
    elif hasattr(train_loader.dataset, 'dataset'):
        indices = train_loader.dataset.indices
        targets = [train_loader.dataset.dataset.targets[i] for i in indices]
        for i, cls_name in enumerate(classes):
            class_counts[cls_name] = sum(1 for t in targets if t == i)
            
    plots_dir = os.path.join(session_run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    if class_counts:
        dist_save_path = os.path.join(plots_dir, "class_distribution.png")
        plot_class_distribution(
            class_counts, 
            dist_save_path, 
            title=f"Class Distribution: Train Dataset ({sum(class_counts.values()):,} images)"
        )

    models_to_run = SUPPORTED_MODELS if args.model == "all" else [args.model]
    benchmark_results = []

    for model_name in models_to_run:
        config = copy.copy(base_config)
        config.model_name = model_name
        
        res = train_single_model(config, train_loader, val_loader, test_loader, classes)
        benchmark_results.append(res)

    # Xuất file báo cáo tổng hợp và ảnh đối sánh ngay trong session_run_dir
    generate_consolidated_report(benchmark_results, classes, session_run_dir)

    print("\n" + "=" * 96)
    print("TỔNG HỢP KẾT QUẢ ĐỐI SÁNH CÁC MÔ HÌNH TRÊN TẬP GỐC (FULL DATASET)")
    print("=" * 96)
    print(f"{'Mô hình':<14} | {'Params (M)':<10} | {'Test Acc (%)':<13} | {'Val Acc (%)':<12} | {'Macro F1 (%)':<13} | {'Latency (ms)':<13} | {'Thời gian':<10}")
    print("-" * 96)
    for r in benchmark_results:
        test_acc_str = f"{r.get('test_acc', 'N/A')}%"
        val_acc_str = f"{r.get('best_val_acc', 'N/A')}%"
        print(f"{r['model']:<14} | {str(r['params_m'])+'M':<10} | {test_acc_str:<13} | {val_acc_str:<12} | {str(r['macro_f1'])+'%':<13} | {str(r['latency_ms'])+'ms':<13} | {str(r['total_time_seconds'])+'s':<10}")
    print("=" * 96)
    print(f"\n[*] TOÀN BỘ KẾT QUẢ ĐÃ ĐƯỢC GOM GỌN TRONG 1 THƯ MỤC DUY NHẤT:")
    print(f"    --> {session_run_dir}")
    print(f"    ├── benchmark_summary.csv     (Bảng tổng quan các chỉ số kỹ thuật)")
    print(f"    ├── per_class_f1.csv          (Bảng F1 từng lớp theo từng mô hình)")
    print(f"    ├── plots/                    (Toàn bộ biểu đồ so sánh, biểu đồ riêng, ma trận nhầm lẫn)")
    print(f"    ├── weights/                  (Các file checkpoint trọng số .pth)")
    print(f"    └── logs/                     (Nhật ký huấn luyện chi tiết từng epoch & metadata)")
    print("=" * 85)

if __name__ == "__main__":
    main()
