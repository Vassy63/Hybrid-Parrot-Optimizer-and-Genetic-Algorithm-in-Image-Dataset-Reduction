import os

class PathsConfig:
    def __init__(self):
        self.BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.DATASET_TRAIN_DIR = os.path.join(self.BASE_DIR, "seg_train", "seg_train")
        if not os.path.exists(self.DATASET_TRAIN_DIR):
            self.DATASET_TRAIN_DIR = os.path.join(self.BASE_DIR, "seg_train")
            
        self.DATASET_TEST_DIR = os.path.join(self.BASE_DIR, "seg_test", "seg_test")
        if not os.path.exists(self.DATASET_TEST_DIR):
            self.DATASET_TEST_DIR = os.path.join(self.BASE_DIR, "seg_test")
            
        self.FEATURES_DIR = os.path.join(self.BASE_DIR, "outputs", "features")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "outputs", "results")

class FitnessConfig:
    def __init__(self):
        self.global_bounds_samples = 100
        # Trọng số cho 4 thành phần fitness (Tổng = 1)
        self.alpha = 0.25  # Diversity
        self.beta = 0.25   # Coverage
        self.gamma = 0.25  # Balance
        self.delta = 0.25  # Compression

class Config:
    def __init__(self):
        self.paths = PathsConfig()
        self.fitness = FitnessConfig()
        # 6 lớp dữ liệu của Intel Image Dataset
        self.classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.num_classes = len(self.classes)

config = Config()
