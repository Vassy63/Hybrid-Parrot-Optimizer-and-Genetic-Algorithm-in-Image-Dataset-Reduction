import os
import time
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import BaselineConfig
from utils.utils import (
    save_run_metadata, save_training_log_csv,
    compute_classification_metrics, plot_confusion_matrix,
    save_classification_report, count_model_params
)

class BaselineTrainer:
    """Class quản lý vòng lặp huấn luyện, đánh giá và lưu checkpoint mô hình."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: BaselineConfig,
        classes: List[str]
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.classes = classes
        self.device = torch.device(config.device)
        
        self.model.to(self.device)
        self.history: Dict[str, List[float]] = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        self.epoch_times: List[float] = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        self.total_params, self.trainable_params = count_model_params(self.model)
        
        # Thư mục con weights và logs nằm gọn bên trong run_dir duy nhất
        self.weights_dir = os.path.join(self.config.run_dir, "weights")
        self.logs_dir = os.path.join(self.config.run_dir, "logs")
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.best_model_path = os.path.join(self.weights_dir, f"{self.config.model_name}_best.pth")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.epochs} [Train]")
        for images, labels in train_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
            train_bar.set_postfix({'loss': f"{running_loss/total:.4f}", 'acc': f"{correct/total:.4f}"})
            
        return running_loss / total, correct / total

    def validate(self, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_bar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config.epochs} [Val]  ")
            for images, labels in val_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data).item()
                total += labels.size(0)
                
        return val_loss / total, correct / total

    def save_checkpoint(self, epoch: int, val_acc: float):
        torch.save({
            'run_id': self.config.run_id,
            'model_name': self.config.model_name,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'classes': self.classes,
            'timestamp': self.config.timestamp
        }, self.best_model_path)
        print(f"    [*] Đã lưu checkpoint tốt nhất (Val Acc: {val_acc*100:.2f}%) tại: {self.best_model_path}")

    def evaluate_test(self) -> Dict[str, Any]:
        """Tải trọng số tốt nhất và đánh giá độc lập trên Test Set (hoặc Val Set nếu không có Test Set)."""
        target_loader = self.test_loader if self.test_loader is not None else self.val_loader
        if not target_loader or not os.path.exists(self.best_model_path):
            return {}
            
        dataset_name = "tập kiểm thử độc lập (Test Set)" if self.test_loader is not None else "tập validation (Val Set)"
        print(f"\n[*] Đang thực hiện đánh giá chuyên sâu trên {dataset_name} bằng checkpoint tốt nhất...")
        checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        y_true = []
        y_pred = []
        inference_latencies = []
        
        with torch.no_grad():
            for images, labels in target_loader:
                images = images.to(self.device)
                
                t0 = time.time()
                outputs = self.model(images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                batch_ms = (time.time() - t0) * 1000.0
                inference_latencies.append(batch_ms / images.size(0))
                
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.tolist())
                y_pred.extend(preds.cpu().tolist())
                
        metrics = compute_classification_metrics(y_true, y_pred, self.classes)
        metrics["avg_inference_latency_ms"] = round(float(np.mean(inference_latencies)), 2)
        
        # Lưu classification report riêng của mô hình vào logs/
        cr_path = os.path.join(self.logs_dir, f"{self.config.model_name}_classification_report.csv")
        save_classification_report(metrics, cr_path)
        
        return metrics

    def fit(self) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        start_time = time.time()
        print(f"\n[*] Bắt đầu huấn luyện {self.config.epochs} epochs [Model: {self.config.model_name.upper()} | Run ID: {self.config.run_id}]...")
        print(f"[*] Tổng tham số: {self.total_params:,} (Trainable: {self.trainable_params:,})")

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            if self.val_loader:
                val_loss, val_acc = self.validate(epoch)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                epoch_duration = time.time() - t0
                self.epoch_times.append(epoch_duration)
                
                print(f"--> Epoch {epoch}: Train Acc = {train_acc*100:.2f}% (Loss: {train_loss:.4f}) | "
                      f"Val Acc = {val_acc*100:.2f}% (Loss: {val_loss:.4f}) | Time: {epoch_duration:.1f}s")
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, val_acc)
            else:
                epoch_duration = time.time() - t0
                self.epoch_times.append(epoch_duration)
                print(f"--> Epoch {epoch}: Train Acc = {train_acc*100:.2f}% (Loss: {train_loss:.4f}) | Time: {epoch_duration:.1f}s")

        total_time = time.time() - start_time
        
        # Lưu log chi tiết vào logs/
        log_csv = os.path.join(self.logs_dir, f"{self.config.model_name}_training_log.csv")
        save_training_log_csv(self.logs_dir, self.history, self.epoch_times)
        
        # Đánh giá chuyên sâu trên checkpoint tốt nhất
        test_metrics = self.evaluate_test()
        test_acc = test_metrics.get("accuracy")
        
        # Lưu metadata vào logs/
        summary_stats = {
            "model_name": self.config.model_name,
            "total_params": self.total_params,
            "total_params_million": round(self.total_params / 1e6, 2),
            "trainable_params": self.trainable_params,
            "total_training_time_seconds": round(total_time, 2),
            "best_validation_accuracy": round(self.best_val_acc * 100, 2),
            "best_epoch": self.best_epoch,
            "test_accuracy": test_acc,
            "final_train_loss": round(self.history['train_loss'][-1], 4),
            "final_train_acc": round(self.history['train_acc'][-1] * 100, 2),
            "macro_f1": test_metrics.get("macro_f1"),
            "macro_precision": test_metrics.get("macro_precision"),
            "macro_recall": test_metrics.get("macro_recall"),
            "avg_inference_latency_ms": test_metrics.get("avg_inference_latency_ms"),
            "classes": self.classes,
            "device_used": self.config.device
        }
        meta_path = os.path.join(self.logs_dir, f"{self.config.model_name}_metadata.json")
        save_run_metadata(self.logs_dir, self.config.run_id, asdict(self.config), summary_stats)

        print(f"\n==================================================")
        print(f"HOÀN THÀNH HUẤN LUYỆN TRONG {total_time:.1f} GIÂY")
        print(f"Mô hình                       : {self.config.model_name.upper()}")
        print(f"File checkpoint               : {self.best_model_path}")
        if self.val_loader:
            print(f"Độ chính xác Validation cao nhất: {self.best_val_acc*100:.2f}% (tại Epoch {self.best_epoch})")
        if test_acc is not None:
            print(f"Độ chính xác Kiểm thử (Test Acc): {test_acc:.2f}%")
            if "macro_f1" in test_metrics:
                print(f"Macro F1-Score (Test Set)       : {test_metrics['macro_f1']:.2f}%")
                print(f"Thời gian suy luận (Latency)    : {test_metrics['avg_inference_latency_ms']} ms/ảnh")
        print(f"==================================================")
        
        return self.history, test_metrics
