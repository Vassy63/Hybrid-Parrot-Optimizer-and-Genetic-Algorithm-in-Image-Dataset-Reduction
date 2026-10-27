from dataclasses import asdict, dataclass
from pathlib import Path
import math

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    's1': 'S1 · Lai xen kẽ GA → PO', 's2': 'S2 · Lai xen kẽ PO → GA',
    's3': 'S3 · Lai chia khối GA → PO', 's4': 'S4 · Lai chia khối PO → GA',
    'p1': 'P1 · Lai song song hợp nhất', 'c1': 'C1 · Lai hợp tác đồng tiến hóa',
    'ga': 'GA · Giải thuật di truyền', 'po': 'PO · Tối ưu hóa đàn vẹt',
    'random': 'Random · Lấy mẫu ngẫu nhiên', 'stratified': 'Stratified · Lấy mẫu phân tầng',
}
MODELS = {'resnet18': 'ResNet-18', 'mobilenet_v3': 'MobileNetV3-Small', 'densenet121': 'DenseNet-121'}


def minimum_retained(n, reduction_percent):
    if not isinstance(n, int) or n < 1:
        raise ValueError('Số ảnh phải là số nguyên dương.')
    if not math.isfinite(float(reduction_percent)) or not 0 <= reduction_percent <= 70:
        raise ValueError('Tỷ lệ rút gọn phải nằm trong khoảng 0–70%.')
    return max(1, math.ceil(n * (100 - reduction_percent) / 100 - 1e-12))


@dataclass
class DemoSettings:
    source_dir: str = str(PROJECT_ROOT)
    train_dir: str = str(PROJECT_ROOT / 'datasets' / 'intel' / 'seg_train' / 'seg_train')
    test_dir: str = str(PROJECT_ROOT / 'datasets' / 'intel' / 'seg_test' / 'seg_test')
    method: str = 's1'
    model: str = 'resnet18'
    num_samples: int = 600
    reduction_percent: float = 70
    population_size: int = 10
    iterations: int = 20
    phase_size: int = 5
    epochs: int = 3
    batch_size: int = 64
    seed: int = 42
    time_limit_minutes: float = 15
    run_baseline: bool = True
    device: str = 'auto'

    @property
    def keep_ratio(self):
        return 1 - self.reduction_percent / 100

    def validate(self, paths=False):
        minimum_retained(self.num_samples, self.reduction_percent)
        if self.method not in METHODS or self.model not in MODELS:
            raise ValueError('Phương pháp hoặc mô hình CNN không được hỗ trợ.')
        if not 120 <= self.num_samples <= 11228:
            raise ValueError('Số ảnh demo phải nằm trong khoảng 120–11.228.')
        if not 6 <= self.population_size <= 40:
            raise ValueError('Quần thể phải có 6–40 cá thể, lớn hơn 5 cá thể tinh hoa.')
        if not 2 <= self.iterations <= 200 or not 1 <= self.phase_size <= self.iterations // 2:
            raise ValueError('Cần ít nhất hai pha tối ưu; độ dài pha không vượt nửa tổng vòng.')
        if not 1 <= self.epochs <= 30 or not 1 <= self.batch_size <= 128:
            raise ValueError('Epochs phải là 1–30, batch size phải là 1–128.')
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('Seed phải là số nguyên không âm nhỏ hơn 2^32.')
        if not 0 < self.time_limit_minutes <= 60:
            raise ValueError('Ngân sách thời gian phải lớn hơn 0 và tối đa 60 phút.')
        if self.device not in ['auto', 'cpu', 'cuda']:
            raise ValueError('Thiết bị không hợp lệ.')
        if paths:
            root = Path(self.source_dir)
            if not all((root / p).is_file() for p in ['configs/config.py','optimization/fitness.py','train.py']):
                raise ValueError('Đường dẫn source không chứa đầy đủ engine PO–GA.')
            for name, value in [('train', self.train_dir), ('test', self.test_dir)]:
                path = Path(value)
                if not path.is_dir() or len([p for p in path.iterdir() if p.is_dir()]) != 6:
                    raise ValueError(f'Thư mục {name} phải chứa đúng sáu thư mục lớp ảnh: {path}')
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        allowed = cls.__dataclass_fields__
        return cls(**{k: v for k, v in values.items() if k in allowed}).validate()
