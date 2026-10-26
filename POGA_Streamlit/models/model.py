import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = ["resnet18", "mobilenet_v3", "densenet121"]

def build_baseline_model(model_name: str = "resnet18", num_classes: int = 6, fine_tune: bool = False) -> nn.Module:
    """
    Xây dựng mô hình Deep Learning tiền huấn luyện trên ImageNet.
    
    Args:
        model_name: Tên mô hình ('resnet18', 'mobilenet_v3', 'densenet121')
        num_classes: Số lượng lớp đầu ra.
        fine_tune: Nếu False, đóng băng backbone và chỉ huấn luyện classifier head.
                   Nếu True, cho phép cập nhật toàn bộ mạng.
    """
    model_name = model_name.lower()
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_name}. Danh sách hỗ trợ: {SUPPORTED_MODELS}")
        
    return model
