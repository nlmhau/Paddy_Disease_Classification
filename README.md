# HỆ THỐNG PHÂN LOẠI BỆNH LÁ LÚA
## Rice Leaf Disease Classification System

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)

Dự án phân loại 10 loại bệnh lá lúa sử dụng Deep Learning (CNN, EfficientNet) với độ chính xác cao, kèm giao diện web tương tác.

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Dữ liệu](#dữ-liệu)
- [Sử dụng](#sử-dụng)
- [Các mô hình](#các-mô-hình)
- [Kết quả](#kết-quả)
- [Demo Web App](#demo-web-app)
- [Tác giả](#tác-giả)

---

## Giới thiệu

Dự án này xây dựng hệ thống phân loại bệnh lá lúa tự động sử dụng các mô hình Deep Learning tiên tiến. Hệ thống có khả năng nhận diện **10 loại bệnh** phổ biến trên cây lúa, hỗ trợ nông dân và chuyên gia nông nghiệp trong việc chẩn đoán sớm và xử lý kịp thời.

### Các loại bệnh được phân loại:
1. Bệnh bạc lá (vi khuẩn) - `bacterial_leaf_blight`
2. Bệnh sọc lá vi khuẩn - `bacterial_leaf_streak`
3. Bệnh thối bông vi khuẩn - `bacterial_panicle_blight`
4. Bệnh đạo ôn - `blast`
5. Bệnh đốm nâu - `brown_spot`
6. Bệnh chết tim - `dead_heart`
7. Bệnh sương mai - `downy_mildew`
8. Bệnh sâu hispa - `hispa`
9. Lá khỏe mạnh - `normal`
10. Bệnh vàng lùn (Tungro) - `tungro`

### Đặc điểm nổi bật:
- 2 mô hình Deep Learning (CNN tự thiết kế, EfficientNetB1)
- Pipeline xử lý dữ liệu chuẩn: Preprocessing → EDA → Model → Evaluation
- Data Augmentation nâng cao độ chính xác
- Web App tương tác với Streamlit
- Visualization chi tiết (confusion matrix, accuracy/loss curves, EDA plots)

---

## Cấu trúc thư mục

```
Heart_Disease_Prediction_ML/
│
├── README.md                           # Hướng dẫn cài đặt và chạy project
│
├── requirements.txt                    # Danh sách thư viện cần cài
│
├── data/                               # Thư mục chứa dữ liệu
│   ├── train.csv                       # File CSV chứa thông tin ảnh, nhãn, giống lúa, tuổi
│   └── train_images/                   # Thư mục chứa ảnh huấn luyện
│       ├── bacterial_leaf_blight/
│       ├── bacterial_leaf_streak/
│       ├── bacterial_panicle_blight/
│       ├── blast/
│       ├── brown_spot/
│       ├── dead_heart/
│       ├── downy_mildew/
│       ├── hispa/
│       ├── normal/
│       └── tungro/
│
├── models/                             # Thư mục chứa các model đã train
│   ├── monster_cnn_best.keras          # Model CNN tự thiết kế
│   └── efnet_b1_best.keras             # Model EfficientNetB1
│
├── src/                                # Mã nguồn Python
│   ├── preprocessing.py                # Xử lý dữ liệu (load, split, augmentation)
│   ├── eda.py                          # Phân tích dữ liệu khám phá (EDA)
│   ├── model_cnn.py                    # Huấn luyện mô hình CNN
│   ├── model_efficientnetb1.py         # Huấn luyện mô hình EfficientNetB1
│   ├── evaluation.py                   # Đánh giá mô hình
│   ├── CNN.ipynb                       # Notebook thử nghiệm CNN
│   └── efficienNetb1.ipynb             # Notebook thử nghiệm EfficientNet
│
└── web/                                # Giao diện web
    └── app.py                          # Web app Streamlit để demo dự đoán
```

---

## Cài đặt

### Yêu cầu hệ thống:
- Python 3.8+
- GPU (khuyến nghị) hoặc CPU
- RAM: Tối thiểu 8GB

### Các bước cài đặt:

1. **Clone repository:**
   ```bash
   git clone <https://github.com/nlmhau/Paddy_Disease_Classification>
   ```

2. **Tạo môi trường ảo (khuyến nghị):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Cài đặt thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

### Thư viện chính:
- `tensorflow` - Deep Learning framework
- `streamlit` - Web app framework
- `pandas`, `numpy` - Xử lý dữ liệu
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `scikit-learn` - Machine learning utilities
- `pillow` - Xử lý ảnh

---

##  Dữ liệu

### Cấu trúc dữ liệu:
- **Tổng số ảnh:** ~10,400 ảnh
- **Định dạng:** JPG
- **Kích thước:** 224x224 pixels (sau resize)
- **Phân bố:** 10 classes (không cân bằng - xử lý bằng class weights)

### File train.csv:
Chứa 4 cột:
- `image_id`: Tên file ảnh
- `label`: Nhãn bệnh (tiếng Anh)
- `variety`: Giống lúa
- `age`: Tuổi cây lúa (ngày)

### Download dữ liệu:
Nếu dữ liệu quá lớn, vui lòng tải từ link:
```
[Link Google Drive / Kaggle Dataset]
```
Sau đó giải nén vào thư mục `data/`

---

## Sử dụng

### 1. Phân tích dữ liệu (EDA):
```bash
python src/eda.py
```
Tạo các biểu đồ:
- Phân bố giống lúa
- Phân bố nhãn bệnh
- Thống kê tuổi lúa theo bệnh
- Hiển thị ảnh mẫu

### 2. Tiền xử lý dữ liệu:
```python
from src.preprocessing import *

# Load dữ liệu
paths = default_data_paths()
df = load_train_df(paths.train_csv)
df = add_image_path_column(df, paths.train_img_dir)

# Chia train/val (80/20)
train_df, val_df = split_train_val(df, test_size=0.2, random_state=42)

# Tạo data generators với augmentation
train_gen, val_gen = build_image_generators(train_df, val_df, batch_size=32)
```

### 3. Huấn luyện mô hình:

#### a) CNN tự thiết kế:
```bash
python src/model_cnn.py
```

#### b) EfficientNetB1:
```bash
python src/model_efficientnetb1.py
```

#### c) Sử dụng Notebook:
- Mở `src/CNN.ipynb` hoặc `src/efficienNetb1.ipynb`
- Chạy từng cell để train và đánh giá

### 4. Đánh giá mô hình:
```python
from src.evaluation import *
from tensorflow.keras.models import load_model

# Load model
model = load_model('models/monster_cnn_best.keras')

# Dự đoán và đánh giá
y_true, y_pred, class_names, y_pred_prob = predict_on_generator(model, val_gen)

# Hiển thị kết quả
print_classification_report(y_true, y_pred, class_names)
plot_confusion_matrix(y_true, y_pred, class_names)
```

### 5. Chạy Web App:
```bash
streamlit run web/app.py
```
Sau đó mở trình duyệt tại: `http://localhost:8501`

---

## Các mô hình

### 1. Monster CNN (Custom CNN)
- **Kiến trúc:** 6 Conv Blocks + BatchNorm + Dropout
- **Tham số:** ~2.5M parameters
- **Đặc điểm:** Thiết kế từ đầu, tối ưu cho bài toán
- **File:** `models/monster_cnn_best.keras`

### 2. EfficientNetB1
- **Kiến trúc:** Transfer Learning từ ImageNet
- **Tham số:** ~7M parameters
- **Đặc điểm:** Fine-tuning top layers, high accuracy
- **File:** `models/efnet_b1_best.keras`


### Kỹ thuật huấn luyện:
- **Optimizer:** Adam (lr=0.0001)
- **Loss function:** Categorical Crossentropy
- **Callbacks:** 
  - EarlyStopping (patience=10)
  - ReduceLROnPlateau (patience=5)
  - ModelCheckpoint (lưu best model)
- **Class weights:** Xử lý imbalanced data
- **Data Augmentation:**
  - Rotation (±20°)
  - Width/Height shift (±20%)
  - Zoom (±20%)
  - Horizontal flip
  - Brightness adjustment

---

## 📈 Kết quả

| Model | Accuracy (Val) | Precision | Recall | F1-Score |
|-------|---------------|-----------|--------|----------|
| Monster CNN | ~92% | 0.91 | 0.90 | 0.90 |
| EfficientNetB1 | ~95% | 0.94 | 0.93 | 0.93 |

*Lưu ý: Kết quả có thể khác nhau tùy thuộc vào seed và augmentation*

### Visualization:
- **Confusion Matrix:** Thể hiện phân loại đúng/sai cho từng class
- **Training curves:** Accuracy/Loss qua các epochs
- **Sample predictions:** Hiển thị ảnh với nhãn dự đoán

---

## Demo Web App

Web app cung cấp các tính năng:

### 1. Trang chủ:
- Giới thiệu dự án
- Thống kê tổng quan
- Hướng dẫn sử dụng

### 2. Phân tích dữ liệu (EDA):
- Biểu đồ phân bố dữ liệu
- Thống kê các loại bệnh
- Hình ảnh mẫu từng class

### 3. Dự đoán ảnh:
- Upload ảnh lá lúa
- Chọn model (CNN/EfficientNet)
- Xem kết quả dự đoán với xác suất
- Hiển thị top-2 predictions

### 4. So sánh mô hình:
- So sánh accuracy của 2 models
- Confusion matrix cho từng model
- Thống kê chi tiết


---


## Tham khảo

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Streamlit Documentation](https://docs.streamlit.io/)
- Rice Disease Dataset: [Kaggle/Research Paper]

---

