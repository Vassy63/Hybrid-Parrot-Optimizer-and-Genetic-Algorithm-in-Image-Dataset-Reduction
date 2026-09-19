import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate_predictions(y_test, y_pred, classes):
    """
    Đánh giá kết quả dự đoán của mô hình.
    """
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, f1, report, cm

def evaluate_subset(solution):
    """
    Đánh giá độ nén của tập dữ liệu con (tỷ lệ 0 so với toàn bộ)
    """
    N = len(solution)
    M = np.sum(solution == 1)
    compression_ratio = 1.0 - (M / N)
    
    return M, compression_ratio
