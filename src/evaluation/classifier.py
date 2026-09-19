import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def train_and_predict(X_train, y_train, X_test, model_type='svm'):
    """
    Huấn luyện mô hình và dự đoán trên tập kiểm thử.
    Hỗ trợ: 'svm', 'knn', 'rf', 'mlp'
    """
    if model_type == 'svm':
        model = SVC(kernel='linear', max_iter=2000, random_state=42)
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    elif model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    predictions = model.predict(X_test)
    
    return predictions, train_time, model
