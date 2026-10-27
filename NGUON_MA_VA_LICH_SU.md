# Nguon ma va Lich su tich hop (NGUON_MA_VA_LICH_SU.md)

> Tai lieu nay ghi nhan vai tro cua tung nhanh, phien ban tai thoi diem tich hop,
> nguon goc bo ket qua 30 seed va ranh gioi giua ket qua nghien cuu voi luot demo.

## 1. Bon nhanh nguon va vai tro

### backup-30seeds-baseline
- **SHA tai thoi diem tich hop**: 3275ade0c77fb0ed190118a34be2d5556585c1fe
- **Vai tro**: Nhanh luu tru an toan trang thai duong ong 30 seed da xac minh.
  - Chua notebook Kaggle (kaggle_poga_notebook.ipynb) voi 18 tep Python nhung inline.
  - Chua notebook hut zip ket qua va khoi kiem tra CPU.
  - Khong co commit --target_ratio; la diem tham chieu truoc khi them tinh nang moi.
- **Phan da ke thua vao main**: Cac notebook Kaggle va script phan seed chia cho thanh vien
  da duoc dua vao feature-target-ratio (commit 8bac385 -> 3275ade). Nhanh nay
  **khong can merge them**; tra cuu truc tiep tren nhanh nguon khi can.

### feature-target-ratio
- **SHA tai thoi diem tich hop**: d7ed8452db6cd7455b706a69d3baef072665a6ab
- **Vai tro**: Nhanh phat trien chinh cua duong ong thuc nghiem CLI.
  - Bo sung tham so --target_ratio (commit d7ed845) cho run_poga.py.
  - Chua toan bo bo loi: configs/, data/, optimization/, training/, utils/,
    run_poga.py, train.py.
  - Chua script tong hop scripts/merge_seed_results.py, huong dan 30 seed va
    tep dac trung tien tinh san features/train_features_train80.npy.
- **Phan da ke thua vao main**: Toan bo -- nhanh main duoc tao tu nhanh nay.

### jaxy
- **SHA tai thoi diem tich hop**: 2593b6a2d97878fe00b443b04e0a377093e4c3e6
- **Vai tro**: Nhanh phat trien som cua thanh vien Jaxy.
  - Chua notebook Kaggle phien ban dau (nhung 18 tep Python).
  - La tien than cua backup-30seeds-baseline (lich su commit giong nhau den 2593b6a).
- **Phan da ke thua vao main**: Thong qua backup-30seeds-baseline -> feature-target-ratio.
  Nhanh jaxy **khong can merge them**; tham chieu truc tiep khi can kiem tra lich su.

### woke2509
- **SHA tai thoi diem tich hop**: 0ef2e4dd7d144b52a1d88af99b81cc5a62be6152
- **Vai tro**: Nhanh phat trien ung dung Streamlit demo cua thanh vien woke2509.
  - Chua POGA_Streamlit/ -- giao dien nguoi dung hoan chinh.
  - Bo loi ben trong POGA_Streamlit/ (configs, data, optimization, training, utils)
    **dong nhat hoan toan** voi feature-target-ratio tai thoi diem tich hop
    (diff trong cho tat ca cac module loi).
  - Chua du lieu ket qua 30 seed CSV trong POGA_Streamlit/assets/results_30seeds/.
  - Chua ga_reduction.py va src/ (code GA doc lap som hon) -- khong dua vao main
    vi da duoc thay the hoan toan boi bo loi optimization/.
- **Phan da ke thua vao main**: Toan bo thu muc POGA_Streamlit/ duoc tich hop
  (xem Commit 2), bo loi khong chep them vi da dong nhat.

## 2. Bo ket qua 30 seed -- nguon goc va gioi han xac minh

### Xac nhan cua nhom
Bo ket qua 30 seed trong POGA_Streamlit/assets/results_30seeds/ (bao gom
experiment_results_30seeds.csv, summary_30seeds.csv, benchmark_summary_full_b0.csv,
per_class_f1_full_b0.csv, wilcoxon_C1_30seeds.csv) duoc chay bang cac notebook
phan seed cho tung thanh vien, thuoc dong ma backup-30seeds-baseline.

Notebook nguon duoc xac nhan la:
- notebooks/kaggle_30seeds_person1.ipynb (nguoi 1)
- notebooks/kaggle_30seeds_person2.ipynb (nguoi 2)
- notebooks/kaggle_30seeds_person3.ipynb (nguoi 3)
- kaggle_poga_notebook.ipynb -- notebook tong hop nhung inline

### Gioi han xac minh
- **Commit chinh xac** cua tung lan chay sinh ra cac so lieu CSV chua co log
  Kaggle dinh kem de chung minh. Khong tu khang dinh commit cu the.
- Bo ket qua nay **khong duoc sua doi** trong qua trinh tich hop.
- Cac lan chay demo (qua Streamlit) tao ket qua rieng biet, **khong ghi de**
  va **khong duoc cong gop** vao bo 30 seed.

## 3. Ranh gioi giua che do thuc nghiem va demo

| Diem phan biet | Thuc nghiem (CLI) | Demo (Streamlit) |
|---|---|---|
| Entry point | run_poga.py CLI | POGA_Streamlit/src/app.py |
| So seed | 30 (phan cho thanh vien) | 1 (do nguoi dung chon) |
| So anh | Toan bo tap train (~11.228) | 120-11.228 (slider) |
| target_ratio | Ty le giu lai muc tieu mem trong F_Com | Duoc tinh tu reduction_percent |
| Gioi han DR | Khong co gioi han cung | Toi da 70% |
| Bao toan lop | repair_solution loi | enforce_keep_floor (demo adapter) |
| Ket qua luu | outputs/ theo run_id | POGA_Streamlit/runs/ theo phien |
| Thoi gian | Hang gio (Kaggle GPU) | Toi da 15 phut (ngan sach demo) |
| Muc dich | Nghien cuu khoa hoc | Minh hoa hoat dong |

> **Luu y**: Ket qua demo it epoch chi dung de minh hoa hoat dong cua thuat toan.
> Khong thay the hoac so sanh truc tiep voi ket qua nghien cuu 30 seed.

## 4. Tep ket qua nghien cuu duoc bao toan

    POGA_Streamlit/assets/results_30seeds/
    +-- experiment_results_30seeds.csv   # Ket qua tung lan chay (Method x Model x Seed)
    +-- summary_30seeds.csv              # Tong hop trung binh +/- std
    +-- benchmark_summary_full_b0.csv    # Tom tat so sanh voi B0
    +-- per_class_f1_full_b0.csv         # F1 theo tung lop so voi B0
    +-- wilcoxon_C1_30seeds.csv          # Kiem dinh Wilcoxon cho C1

Cac tep nay **chi duoc doc** boi ung dung Streamlit (tab "Ket qua 30 seeds").
Khong co co che nao trong pipeline tich hop ghi de cac tep nay.

## 5. Cac tep khong dua vao main

- POGA_Streamlit/ga_reduction.py, POGA_Streamlit/src/ (ga_reduction doc lap):
  da duoc thay the hoan toan boi optimization/. Xem tren nhanh woke2509.
- outputs/results/run_ga_*/: ket qua chay GA tho tren nhanh woke2509.
  Xem truc tiep tren nhanh nguon.
- Dataset anh Intel Image: khong luu trong Git. Tai tu Kaggle
  (xem huong dan trong README.md va POGA_Streamlit/scripts/prepare_dataset.py).
- Trong so mo hinh (.pth), cache dac trung (.npy sinh ra khi chay),
  moi truong ao .venv: xem .gitignore.