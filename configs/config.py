import os
import sys
import datetime
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple

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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@dataclass
class GAConfig:
    population_size: int = 20
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate_max: float = 0.0010
    mutation_rate_min: float = 0.0002
    tournament_size: int = 4
    elitism_count: int = 5
    init_ratio: float = 0.30
    early_stopping_patience: int = 15

@dataclass
class POConfig:
    population_size: int = 20
    iterations: int = 100
    levy_beta: float = 1.5
    early_stopping_patience: int = 30

@dataclass
class FitnessConfig:
    w_div: float = 0.15
    w_cov: float = 0.45
    w_bal: float = 0.20
    w_com: float = 0.20

@dataclass
class POGAConfig:
    ga: GAConfig = field(default_factory=GAConfig)
    po: POConfig = field(default_factory=POConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    total_iterations: int = 100
    split_ratio: float = 0.5
    phase_size: int = 10
    exchange_k: int = 3
    num_classes: int = 6

@dataclass
class BaselineConfig:
    """Cấu hình cho toàn bộ pipeline huấn luyện và rút gọn dữ liệu PO-GA."""
    train_dir: str = "data/seg_train/seg_train"
    test_dir: str = "data/seg_test/seg_test"
    val_split: float = 0.2
    save_dir: str = "outputs"
    model_name: str = "resnet18"
    run_name: Optional[str] = None
    subset_indices_path: Optional[str] = None
    
    # Model & Optimization
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    fine_tune: bool = False
    
    # Debug / Mini test
    num_samples: Optional[int] = None
    seed: int = 42
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run directory & identification
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_id: str = ""
    run_dir: str = ""

    def __post_init__(self):
        # Tự động nhận diện đường dẫn dữ liệu và thư mục lưu khi chạy trên Kaggle
        if os.path.exists("/kaggle/working"):
            if self.save_dir == "outputs":
                self.save_dir = "/kaggle/working/outputs"
            if not os.path.exists(self.train_dir) and os.path.exists("/kaggle/input"):
                for root, dirs, _ in os.walk("/kaggle/input"):
                    if "buildings" in dirs and "forest" in dirs:
                        if "seg_train" in root:
                            self.train_dir = root
                        elif "seg_test" in root:
                            self.test_dir = root
        if not self.run_id:
            prefix = f"run_{self.run_name}_" if self.run_name else f"run_{self.model_name}_"
            self.run_id = f"{prefix}{self.timestamp}"
        if not self.run_dir:
            self.run_dir = os.path.join(self.save_dir, self.run_id)

