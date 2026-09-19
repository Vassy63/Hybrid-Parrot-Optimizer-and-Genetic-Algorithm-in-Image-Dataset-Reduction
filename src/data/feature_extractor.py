import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from src.config import config

def extract_features(img_dir, output_features_path, output_labels_path, classes):
    print(f"[*] Đang xử lý thư mục: {img_dir}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Thư mục không tồn tại: {img_dir}. Vui lòng giải nén dataset.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Sử dụng thiết bị: {device}")

    # Load ResNet18
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity() # Bỏ classification head
    model.eval()
    model = model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    image_paths = []
    labels = []
    
    for cls_name in classes:
        cls_dir = os.path.join(img_dir, cls_name)
        if os.path.isdir(cls_dir):
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(cls_dir, img_name))
                    labels.append(class_to_idx[cls_name])
                    
    N = len(image_paths)
    print(f"[*] Tổng số ảnh tìm thấy: {N}")
    if N == 0:
        return
        
    features = np.zeros((N, 512), dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            if (i + 1) % 1000 == 0:
                print(f"    Đã xử lý {i + 1}/{N} ảnh...")
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = preprocess(img)
                batch_t = torch.unsqueeze(img_t, 0).to(device)
                feat = model(batch_t)
                features[i] = feat.cpu().numpy().flatten()
            except Exception as e:
                print(f"[!] Lỗi khi xử lý {img_path}: {e}")

    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    np.save(output_features_path, features)
    np.save(output_labels_path, labels)
    print(f"[*] Đã lưu {output_features_path} và {output_labels_path}")

def main():
    print("="*50)
    print("TRÍCH XUẤT ĐẶC TRƯNG RESNET-18 (FEATURE EXTRACTION)")
    print("="*50)
    
    train_dir = config.paths.DATASET_TRAIN_DIR
    test_dir = config.paths.DATASET_TEST_DIR
    
    train_feat = os.path.join(config.paths.FEATURES_DIR, "train_features.npy")
    train_lbl = os.path.join(config.paths.FEATURES_DIR, "train_labels.npy")
    test_feat = os.path.join(config.paths.FEATURES_DIR, "test_features.npy")
    test_lbl = os.path.join(config.paths.FEATURES_DIR, "test_labels.npy")
    
    if not os.path.exists(train_feat):
        print("\n--- XỬ LÝ TẬP TRAIN ---")
        extract_features(train_dir, train_feat, train_lbl, config.classes)
    else:
        print("[*] Bỏ qua tập train vì đã tồn tại file features.")
        
    if not os.path.exists(test_feat):
        print("\n--- XỬ LÝ TẬP TEST ---")
        extract_features(test_dir, test_feat, test_lbl, config.classes)
    else:
        print("[*] Bỏ qua tập test vì đã tồn tại file features.")

if __name__ == "__main__":
    main()
