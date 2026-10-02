import os
from typing import Tuple, List, Optional
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from configs.config import BaselineConfig

def get_transforms(image_size: Tuple[int, int] = (224, 224)):
    """Trả về transforms chuẩn hóa cho tập train và validation."""
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def create_dataloaders(config: BaselineConfig) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], List[str]]:
    """
    Nạp dữ liệu và khởi tạo 3 DataLoader chuẩn khoa học:
    - Train: (1 - val_split) của seg_train (~80%, có data augmentation)
    - Val: val_split của seg_train (~20%, dùng để chọn best epoch)
    - Test: 100% của seg_test (3,000 ảnh, độc lập hoàn toàn để đánh giá sau cùng)
    """
    train_transform, eval_transform = get_transforms(config.image_size)

    if not os.path.exists(config.train_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục train: {config.train_dir}.")
        
    full_train_dataset = datasets.ImageFolder(root=config.train_dir)
    classes = full_train_dataset.classes
    print(f"[*] Đã nhận diện {len(classes)} lớp: {classes}")
    print(f"[*] Tổng số ảnh gốc trong seg_train: {len(full_train_dataset)}")

    # Phân chia train/val với seed cố định để đảm bảo tính tái lập (Reproducibility)
    g = torch.Generator().manual_seed(config.seed)
    n_total = len(full_train_dataset)
    n_val = int(n_total * config.val_split)
    n_train = n_total - n_val

    # Đảm bảo chia ngẫu nhiên nhưng cố định theo seed
    perm = torch.randperm(n_total, generator=g).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    # Giới hạn mẫu nếu chạy chế độ fast debug
    if config.num_samples and config.num_samples < n_train:
        print(f"[*] Fast Mode: Giới hạn {config.num_samples} ảnh train.")
        train_indices = train_indices[:config.num_samples]
        val_indices = val_indices[:min(config.num_samples // 4, len(val_indices))]

    # Áp dụng tập con rút gọn (Coreset / Subset) nếu có truyền subset_indices_path
    if getattr(config, 'subset_indices_path', None) and os.path.exists(config.subset_indices_path):
        import numpy as np
        sub_arr = np.load(config.subset_indices_path)
        if sub_arr.ndim == 1 and set(np.unique(sub_arr)).issubset({0, 1}) and len(sub_arr) == len(train_indices):
            sel_pos = np.where(sub_arr == 1)[0]
            train_indices = [train_indices[i] for i in sel_pos]
        else:
            train_indices = [train_indices[int(i)] for i in sub_arr if int(i) < len(train_indices)]
        print(f"[*] Đã áp dụng tập con rút gọn từ '{config.subset_indices_path}': Giữ lại {len(train_indices):,} / {n_train:,} ảnh Train ({len(train_indices)/max(1,n_train)*100:.2f}%).")

    class CustomSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            path, target = self.dataset.samples[self.indices[idx]]
            sample = self.dataset.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target

    train_subset = CustomSubset(full_train_dataset, train_indices, train_transform)
    val_subset = CustomSubset(full_train_dataset, val_indices, eval_transform)

    print(f"[*] Phân chia tập huấn luyện: Train = {len(train_subset)} ảnh | Val = {len(val_subset)} ảnh")

    # Tập Test độc lập từ seg_test
    test_subset = None
    if config.test_dir and os.path.exists(config.test_dir):
        test_dataset = datasets.ImageFolder(root=config.test_dir, transform=eval_transform)
        if config.num_samples and config.num_samples < len(test_dataset):
            test_indices = torch.randperm(len(test_dataset), generator=g)[:min(config.num_samples // 3, len(test_dataset))].tolist()
            test_subset = CustomSubset(test_dataset, test_indices, eval_transform)
        else:
            test_subset = test_dataset
        print(f"[*] Tập kiểm thử độc lập (Test Set): {len(test_subset)} ảnh")

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    ) if test_subset is not None else None

    return train_loader, val_loader, test_loader, classes


def extract_and_cache_features(config: BaselineConfig, cache_dir: Optional[str] = None):
    """
    Trích xuất đặc trưng 512 chiều bằng ResNet-18 (ImageNet pretrained) trên tập Train split
    và lưu cache .npy để phục vụ các thuật toán tối ưu PO-GA, GA, PO, Random, Stratified.
    """
    import numpy as np
    from tqdm import tqdm
    from torchvision import models
    import torch.nn as nn

    if cache_dir is None:
        cache_dir = "features"
    os.makedirs(cache_dir, exist_ok=True)

    suffix = f"_sub{config.num_samples}" if config.num_samples else f"_train{int((1-config.val_split)*100)}"
    feat_name = f"train_features{suffix}.npy"
    label_name = f"train_labels{suffix}.npy"
    feat_path = os.path.join(cache_dir, feat_name)
    label_path = os.path.join(cache_dir, label_name)

    # Kiểm tra trong features/, outputs/features/, hoặc /kaggle/input
    candidate_dirs = [cache_dir, "features", os.path.join(config.save_dir, "features")]
    for cdir in candidate_dirs:
        fp = os.path.join(cdir, feat_name)
        lp = os.path.join(cdir, label_name)
        if os.path.exists(fp) and os.path.exists(lp):
            print(f"[*] Tải đặc trưng có sẵn từ: {fp}")
            return np.load(fp), np.load(lp)

    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if feat_name in files and label_name in files:
                kf = os.path.join(root, feat_name)
                kl = os.path.join(root, label_name)
                print(f"[*] Tải đặc trưng từ Kaggle Input: {kf}")
                return np.load(kf), np.load(kl)

    _, eval_transform = get_transforms(config.image_size)
    full_train_dataset = datasets.ImageFolder(root=config.train_dir)
    g = torch.Generator().manual_seed(config.seed)
    n_total = len(full_train_dataset)
    n_val = int(n_total * config.val_split)
    n_train = n_total - n_val
    perm = torch.randperm(n_total, generator=g).tolist()
    train_indices = perm[:n_train]
    if config.num_samples and config.num_samples < n_train:
        train_indices = train_indices[:config.num_samples]

    class EvalSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            path, target = self.dataset.samples[self.indices[idx]]
            return self.transform(self.dataset.loader(path)), target

    loader = DataLoader(EvalSubset(full_train_dataset, train_indices, eval_transform), batch_size=config.batch_size, shuffle=False, num_workers=0)
    device = torch.device(config.device)
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    extractor.eval()

    feats_list, labels_list = [], []
    print(f"[*] Đang trích xuất đặc trưng ResNet-18 (512-d) cho {len(train_indices):,} ảnh Train trên {device}...")
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting Features"):
            imgs = imgs.to(device)
            out = extractor(imgs).view(imgs.size(0), -1).cpu().numpy().astype(np.float32)
            feats_list.append(out)
            labels_list.append(lbls.numpy())

    features = np.vstack(feats_list)
    labels = np.concatenate(labels_list)
    np.save(feat_path, features, allow_pickle=False)
    np.save(label_path, labels, allow_pickle=False)
    print(f"[*] Đã lưu đặc trưng tại: {feat_path} {features.shape}")
    return features, labels


