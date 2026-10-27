from dataclasses import asdict, dataclass
from pathlib import Path
import math

# POGA_ROOT: thu muc POGA_Streamlit/ -- dung cho assets, outputs, cache cua demo.
# CORE_ROOT: thu muc goc chua bo loi PO-GA dung chung (configs/, optimization/, training/, ...).
# Hai hang so nay tach biet de demo khong phu thuoc vao ban copy loi ben trong POGA_Streamlit/.
POGA_ROOT = Path(__file__).resolve().parents[2]   # .../POGA_Streamlit
CORE_ROOT = POGA_ROOT.parent                       # thu muc goc chua bo loi chung

# Kiem tra: neu CORE_ROOT khong chua bo loi (vi du khi chay POGA_Streamlit doc lap),
# tra ve POGA_ROOT de giu kha nang tuong thich nguoc.
PROJECT_ROOT = CORE_ROOT if (CORE_ROOT / 'optimization' / 'fitness.py').is_file() else POGA_ROOT

METHODS = {
    's1': 'S1 \u00b7 Lai xen ke GA \u2192 PO', 's2': 'S2 \u00b7 Lai xen ke PO \u2192 GA',
    's3': 'S3 \u00b7 Lai chia khoi GA \u2192 PO', 's4': 'S4 \u00b7 Lai chia khoi PO \u2192 GA',
    'p1': 'P1 \u00b7 Lai song song hop nhat', 'c1': 'C1 \u00b7 Lai hop tac dong tien hoa',
    'ga': 'GA \u00b7 Giai thuat di truyen', 'po': 'PO \u00b7 Toi uu hoa dan vet',
    'random': 'Random \u00b7 Lay mau ngau nhien', 'stratified': 'Stratified \u00b7 Lay mau phan tang',
}
MODELS = {'resnet18': 'ResNet-18', 'mobilenet_v3': 'MobileNetV3-Small', 'densenet121': 'DenseNet-121'}


def minimum_retained(n, reduction_percent):
    if not isinstance(n, int) or n < 1:
        raise ValueError('So anh phai la so nguyen duong.')
    if not math.isfinite(float(reduction_percent)) or not 0 <= reduction_percent <= 70:
        raise ValueError('Ty le rut gon phai nam trong khoang 0-70%.')
    return max(1, math.ceil(n * (100 - reduction_percent) / 100 - 1e-12))


@dataclass
class DemoSettings:
    # source_dir tro vao bo loi PO-GA dung chung (PROJECT_ROOT).
    # train_dir / test_dir la duong dan den dataset anh Intel Image.
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
            raise ValueError('Phuong phap hoac mo hinh CNN khong duoc ho tro.')
        if not 120 <= self.num_samples <= 11228:
            raise ValueError('So anh demo phai nam trong khoang 120-11.228.')
        if not 6 <= self.population_size <= 40:
            raise ValueError('Quan the phai co 6-40 ca the, lon hon 5 ca the tinh hoa.')
        if not 2 <= self.iterations <= 200 or not 1 <= self.phase_size <= self.iterations // 2:
            raise ValueError('Can it nhat hai pha toi uu; do dai pha khong vuot nua tong vong.')
        if not 1 <= self.epochs <= 30 or not 1 <= self.batch_size <= 128:
            raise ValueError('Epochs phai la 1-30, batch size phai la 1-128.')
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('Seed phai la so nguyen khong am nho hon 2^32.')
        if not 0 < self.time_limit_minutes <= 60:
            raise ValueError('Ngan sach thoi gian phai lon hon 0 va toi da 60 phut.')
        if self.device not in ['auto', 'cpu', 'cuda']:
            raise ValueError('Thiet bi khong hop le.')
        if paths:
            root = Path(self.source_dir)
            if not all((root / p).is_file() for p in ['configs/config.py', 'optimization/fitness.py', 'train.py']):
                raise ValueError('Duong dan source khong chua day du engine PO-GA.')
            for name, value in [('train', self.train_dir), ('test', self.test_dir)]:
                path = Path(value)
                if not path.is_dir() or len([p for p in path.iterdir() if p.is_dir()]) != 6:
                    raise ValueError(f'Thu muc {name} phai chua dung sau thu muc lop anh: {path}')
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        allowed = cls.__dataclass_fields__
        return cls(**{k: v for k, v in values.items() if k in allowed}).validate()