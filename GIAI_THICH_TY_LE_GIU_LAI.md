# Giai thich: target_ratio, keep_ratio va muc rut gon toi da

Tai lieu nay giai thich su khac biet giua ba khai niem lien quan den ty le giu lai anh trong he thong PO-GA.

## 1. target_ratio -- ty le giu lai muc tieu mem trong F_Com (che do thuc nghiem)

**Vi tri**: Tham so --target_ratio trong run_poga.py; truong FitnessConfig.target_ratio.

**Y nghia**: Khi target_ratio duoc dat (vi du 0.30), thanh phan F_Com cua ham fitness duoc tinh theo
do tiem can muc tieu:

    F_Com = max(0, 1 - |ratio - target_ratio|)

Neu target_ratio = None (mac dinh), F_Com khyen khich nen du lieu toi da:

    F_Com = 1 - ratio   (ratio = so anh chon / tong so anh)

**Dac diem**:
- La rang buoc MEM (soft constraint): thuat toan duoc khuyen khich giu lai gan target_ratio anh,
  nhung khong bi ep buoc phai dung chinh xac so do.
- Khong co san luong nen (DR floor) cung: thuat toan co the chon it hon hoac nhieu hon ty le nay.
- Anh huong truc tiep vao fitness, khong thay doi logic khoi tao hay sua nghiem.
- Hanh vi thuc nghiem goc (--target_ratio khong duoc truyen vao) khong bi thay doi boi cac tinh nang demo.

**Vi du (CLI)**:

    python run_poga.py --target_ratio 0.30

Nay co nghia: F_Com se dat diem cao nhat khi thuat toan chon khoang 30% anh.
Ket qua thuc te co the la 28% hoac 34% tuy theo can bang voi F_Div, F_Cov, F_Bal.

## 2. reduction_percent -- muc rut gon toi da cua demo Streamlit

**Vi tri**: Slider "Rut gon toi da (%)" trong giao dien Streamlit; truong DemoSettings.reduction_percent.

**Y nghia**: Rang buoc CUNG (hard constraint) duoc ap dung SAU khi thuat toan chon tap con:
- Cho phep rut gon toi da reduction_percent phan tram anh.
- Thuat toan se chon ≥ (100 - reduction_percent)% anh, nhung co the giu nhieu hon.
- Duoc thuc thi boi ham enforce_keep_floor() trong poga_demo/constraints.py.

**Ke toan so anh toi thieu**:

    min_keep = ceil(n * (100 - reduction_percent) / 100)

Trong do n la so anh train truoc rut gon. Rang buoc nay dam bao:
- Khong bao gio giam xuong duoi min_keep anh bat ke thuat toan cho ket qua nao.
- Moi lop luon co it nhat mot anh dai dien (bao toan lop).

**Vi du (Streamlit)**:

    Slider: 70%   =>  min_keep = ceil(600 * 0.30) = 180 anh
    Thuat toan co the chon 180, 210, hoac 300 anh tuy ket qua toi uu.
    DR thuc te hien thi sau khi chon: "Rut gon thuc te: XX%"

**Khac voi target_ratio**: keep_ratio = 1 - reduction_percent/100 duoc truyen vao fitness.target_ratio
trong demo, nhung chi la gia tri khoi diem. Rang buoc cung la enforce_keep_floor, khong phai ham fitness.

## 3. keep_ratio -- ty le giu lai tuong ung voi reduction_percent

**Vi tri**: Thuoc tinh tinh trong DemoSettings.keep_ratio = 1 - reduction_percent / 100.

**Dac diem**:
- Duoc truyen vao poga.ga.init_ratio de khoi tao quan the (mac dinh chon ~keep_ratio anh).
- Duoc truyen vao poga.fitness.target_ratio nhu rang buoc mem trong F_Com cua demo.
- Khong dam bao so anh thuc te bang chinh xac keep_ratio * n.

## 4. Bang tom tat so sanh

| Khai niem | Vi tri | Kieu rang buoc | Ai thuc thi | Giu nguyen khi khong dung demo |
|---|---|---|---|---|
| target_ratio (CLI) | FitnessConfig | Mem (soft) | Ham fitness F_Com | Co |
| reduction_percent | DemoSettings | Cung (hard) sau toi uu | enforce_keep_floor | Khong ap dung |
| keep_ratio | DemoSettings.keep_ratio | Khoi tao + soft | GA init + fitness | Khong ap dung |

## 5. Hien thi ket qua trong demo

Sau khi thuat toan hoan tat, giao dien hien thi:
- **Anh da chon**: so anh thuc te trong tap con (>= min_keep).
- **Rut gon thuc te**: DR = 100 * (1 - selected / n) phan tram, tinh tu ket qua thuc.
- **Muc rut gon toi da**: gia tri da chon tren slider (gioi han tren cua DR).

Neu thuat toan giu nhieu anh hon muc toi da (vi du do cam bao toan lop), giao dien van hien thi
dung ty le thuc te va canh bao neu DR > reduction_percent.

## 6. Luu y quan trong

- Trong **che do thuc nghiem** (run_poga.py CLI), khong co gioi han DR cung.
  Ket qua 30 seed trong assets/results_30seeds/ la tu che do nay.
- Trong **demo Streamlit**, gioi han 70% la gioi han cung nham bao dam demo co the hoan tat
  trong 15 phut va tap con con du y nghia thong ke.
- Hai che do co the cho ket qua DR khac nhau vi cau hinh rang buoc khac nhau.
  Khong so sanh truc tiep ket qua DR giua demo va thuc nghiem.