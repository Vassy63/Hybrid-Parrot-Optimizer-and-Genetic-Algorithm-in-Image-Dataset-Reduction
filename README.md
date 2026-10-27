# PO-GA: Parrot Optimizer ket hop Genetic Algorithm cho Rut gon Tap du lieu Anh

**De tai Khoa luan Tot nghiep** - Ung dung thuat toan Parrot Optimizer ket hop Genetic Algorithm
trong rut gon tap du lieu anh cho bai toan phan loai canh tu nhien.

## 1. Nhom sinh vien thuc hien

| STT | Ho va ten | MSSV | Lop |
|---|---|---|---|
| 1 | Nguyen Quoc An | 2001230009 | 14DHTH03 |
| 2 | Luu Tri Kiet | 2001230349 | 14DHTH01 |
| 3 | Thai Anh Quoc | 2001230737 | 14DHTH02 |

## 2. Cau truc thu muc

    POGA-main-integration/
    +-- configs/                  # Cau hinh BaselineConfig, POGAConfig, ...
    +-- data/                     # Module tai du lieu va trich xuat dac trung
    +-- features/                 # Dac trung tien tinh san (train_features_train80.npy)
    +-- models/                   # Dinh nghia kien truc CNN
    +-- optimization/             # Bo loi PO, GA, hybrid PO-GA, ham fitness
    +-- training/                 # Huan luyen (BaselineTrainer)
    +-- utils/                    # Tien ich chung
    +-- scripts/                  # Tong hop ket qua 30 seed (merge_seed_results.py)
    +-- tests/                    # Bo kiem thu (test_integration.py, ...)
    +-- notebooks/                # Notebook Kaggle phan seed cho tung thanh vien
    |   +-- kaggle_30seeds_person1.ipynb
    |   +-- kaggle_30seeds_person2.ipynb
    |   +-- kaggle_30seeds_person3.ipynb
    |   +-- HUONG_DAN_30_SEEDS.md
    +-- POGA_Streamlit/           # Ung dung demo Streamlit
    |   +-- src/app.py            # Entry point Streamlit
    |   +-- src/poga_demo/        # Cac module dieu phoi demo
    |   +-- assets/results_30seeds/ # Du lieu ket qua 30 seed (chi doc)
    |   +-- requirements.txt      # Thu vien can thiet cho demo
    |   +-- setup_demo.bat        # Cai dat moi truong tren Windows
    |   +-- start_demo.bat        # Khoi chay Streamlit tren Windows
    +-- run_poga.py               # Entry point CLI thuc nghiem
    +-- train.py                  # Huan luyen don le
    +-- kaggle_poga_notebook.ipynb # Notebook chay toan bo 30 seed tren Kaggle
    +-- README.md                 # File nay
    +-- NGUON_MA_VA_LICH_SU.md    # Nguon ma va lich su tich hop
    +-- GIAI_THICH_TY_LE_GIU_LAI.md # Giai thich target_ratio va muc rut gon

## 3. Yeu cau he thong

- Python 3.10+ (khuyen nghi 3.11)
- CUDA GPU (khuyen nghi cho thuc nghiem; CPU dc ho tro nhung cham hon nhieu)
- Dataset: Intel Image Classification (Kaggle) -- khong luu trong Git

## 4. Chuan bi du lieu

Tai dataset Intel Image tu Kaggle:

    https://www.kaggle.com/datasets/puneet6060/intel-image-classification

Giai nen vao:

    POGA_Streamlit/datasets/intel/
    +-- seg_train/seg_train/
    |   +-- buildings/, forest/, glacier/, mountain/, sea/, street/
    +-- seg_test/seg_test/
        +-- buildings/, forest/, glacier/, mountain/, sea/, street/

Hoac dung script chuan bi tu dong:

    cd POGA_Streamlit
    python scripts/prepare_dataset.py

## 5. Chay demo Streamlit (Windows)

Buoc 1 -- Cai dat moi truong (chi can lam 1 lan):

    cd POGA_Streamlit
    setup_demo.bat

Buoc 2 -- Khoi chay Streamlit:

    cd POGA_Streamlit
    start_demo.bat

Hoac dung PowerShell:

    cd POGA_Streamlit
    .venv\Scripts\python.exe -m streamlit run src\app.py

Sau do mo trinh duyet tai: http://localhost:8501

**Luu y**:
- Slider "Rut gon toi da (%)" la muc rut gon toi da (hard constraint), mac dinh 70%.
- Ket qua demo it epoch chi minh hoa hoat dong, khong thay the ket qua nghien cuu.
- Cai dat CUDA neu muon chay nhanh hon: chay install_cuda.bat.

## 6. Chay thuc nghiem CLI (30 seed)

Xem huong dan day du trong: notebooks/HUONG_DAN_30_SEEDS.md

Tom tat cac lenh chinh:

    # Chay mot seed don le
    python run_poga.py --method c1 --model resnet18 --seeds 42

    # Chay voi target_ratio (rang buoc mem)
    python run_poga.py --method c1 --target_ratio 0.30 --seeds 42

    # Chay benchmark 30 seed (can GPU Kaggle hoac tu chay phan chia)
    # Xem notebooks/kaggle_30seeds_person{1,2,3}.ipynb

    # Tong hop ket qua sau khi chay xong
    python scripts/merge_seed_results.py --input outputs/ --output summary.csv

## 7. Chay tren Kaggle (30 seed)

Tham khao cac notebook trong thu muc notebooks/:
- kaggle_30seeds_person1.ipynb: seed 1-10
- kaggle_30seeds_person2.ipynb: seed 11-20
- kaggle_30seeds_person3.ipynb: seed 21-30
- kaggle_poga_notebook.ipynb: notebook tong hop

Dataset Kaggle Input can duoc dat dung duong dan. Xem HUONG_DAN_30_SEEDS.md.

## 8. Chay bo kiem thu

    # Kiem thu tich hop (khong can torch hay dataset)
    python -m unittest tests.test_integration -v

    # Kiem thu bo loi Streamlit
    python -m unittest discover -s POGA_Streamlit/tests -p "test_core.py" -v

    # Tat ca kiem thu (khi da cai torch)
    python -m unittest discover -s tests/ -v

## 9. Cac tham so chinh

### Bo loi thuc nghiem (run_poga.py)
| Tham so | Mac dinh | Mo ta |
|---|---|---|
| --method | c1 | Phuong phap lai: s1,s2,s3,s4,p1,c1,ga,po,random,stratified |
| --model | resnet18 | CNN: resnet18, mobilenet_v3, densenet121 |
| --seeds | 42 | Danh sach seed ngau nhien |
| --target_ratio | None | Ty le giu lai muc tieu mem trong F_Com |
| --epochs | 20 | So epoch huan luyen CNN |

### Demo Streamlit
| Tham so | Mac dinh | Mo ta |
|---|---|---|
| Phuong phap | S1 | Phuong phap rut gon |
| CNN | ResNet-18 | Mo hinh huan luyen |
| So anh train | 600 | So anh truoc rut gon |
| Rut gon toi da (%) | 70 | Hard constraint: khong rut gon hon muc nay |
| Epochs | 3 | So epoch demo (it hon thuc nghiem) |

## 10. Ket qua nghien cuu 30 seed

Ket qua tong hop 30 seed trong POGA_Streamlit/assets/results_30seeds/:
- experiment_results_30seeds.csv: toan bo ket qua tung lan chay
- summary_30seeds.csv: trung binh va do lech chuan
- benchmark_summary_full_b0.csv: so sanh voi B0 (toan bo du lieu)
- per_class_f1_full_b0.csv: F1 theo tung lop
- wilcoxon_C1_30seeds.csv: kiem dinh Wilcoxon cho phuong phap C1

**Luu y**: Bo ket qua nay duoc chay tren Kaggle (GPU T4 x2) voi cac notebook phan seed.
Ket qua demo (Streamlit, it epoch, CPU/GPU ngan sach) khong the so sanh truc tiep.

## 11. Tai lieu tham khao

- NGUON_MA_VA_LICH_SU.md: nguon ma va lich su tich hop bon nhanh
- GIAI_THICH_TY_LE_GIU_LAI.md: giai thich target_ratio va muc rut gon
- notebooks/HUONG_DAN_30_SEEDS.md: huong dan chay 30 seed tren Kaggle
- ARCHITECTURE.md: kien truc ky thuat cua he thong
- BAO_CAO_KET_QUA_THUC_NGHIEM_POGA.md: bao cao ket qua thuc nghiem

## 12. Nhan xet quan trong ve ket qua demo

Ket qua demo chay trong Streamlit voi it epoch (mac dinh 3 epoch) chi co y nghia minh hoa.
Do:
- So epoch it: mo hinh chua hoi tu
- So anh nho (600 anh mac dinh): khong dai dien cho toan bo dataset
- Thoi gian gioi han: toi da 15 phut
- Muc rut gon co gioi han cung 70%

Ket qua nghien cuu chinh thuc can tham khao tai POGA_Streamlit/assets/results_30seeds/.