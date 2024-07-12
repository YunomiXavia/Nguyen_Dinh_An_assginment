# Giải thích chi tiết yêu cầu bài toán

## Mục tiêu bài toán
Mục tiêu xây dựng bài toán là xây dựng một mô hình Machine learning nhận dạng khuôn mặt có sử dụng HOG truy xuất đặc trưng của ảnh và SVM là thuật toán train để nhận dạng ảnh. Theo yêu cầu của bài tập thì phần xây dựng này sẽ gồm có 3 yu cầu:

1. **Chuẩn bị dữ liệu**
2. **Phân loại SVM**
3. **Phát hiện khuôn mặt**

## Phần 1: Chuẩn bị dữ liệu

### Bước 1.1: Input dữ liệu và trực quan hóa dữ liệu
#### Mục đích
Tải và hiển thị dữ liệu để hiểu rõ hơn về các mẫu dương (có khuôn mặt) và âm (không có khuôn mặt).

```python
import scipy.io as sio
import matplotlib.pyplot as plt

# Tải dữ liệu mẫu dương và âm
def load_data():
    pos_samples = sio.loadmat('datasets/possamples.mat')['possamples']
    neg_samples = sio.loadmat('datasets/negsamples.mat')['negsamples']
    return pos_samples, neg_samples

# Hiển thị ảnh mẫu dương và âm
def visualize_samples(pos_samples, neg_samples):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(pos_samples[0], cmap='gray')
    axes[0].set_title('Positive Sample')
    axes[1].imshow(neg_samples[0], cmap='gray')
    axes[1].set_title('Negative Sample')
    plt.show()

# Tải và trực quan hóa dữ liệu
pos_samples, neg_samples = load_data()
visualize_samples(pos_samples, neg_samples)
```
- `load_data`: Hàm này tải dữ liệu mẫu dương và âm từ các file .mat.
- `visualize_samples`: Hàm này hiển thị một ảnh mẫu dương và một ảnh mẫu âm bằng matplotlib.
- `pos_samples` và `neg_samples`: Các biến này lưu trữ các mẫu dương và âm sau khi được tải lên.

### Bước 1.2: Chuẩn hóa mean-variance
#### Mục đích
Chuẩn hóa dữ liệu để đưa các giá trị về cùng một phạm vi, giúp mô hình huấn luyện tốt hơn.

```python
import cv2
import numpy as np

# Chuẩn hóa ảnh
def normalize_images(images):
    normalized_images = []
    for img in images:
        img_normalized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        normalized_images.append(img_normalized)
    return np.array(normalized_images)

# Chuẩn hóa ảnh
pos_samples = normalize_images(pos_samples)
neg_samples = normalize_images(neg_samples)
```
- `normalize_images`: Hàm này chuẩn hóa ảnh bằng cách sử dụng OpenCV để đưa các giá trị pixel về khoảng từ 0 đến 255.
- `cv2.normalize`: Hàm này từ OpenCV thực hiện chuẩn hóa ảnh.

### Bước 1.3: Định dạng dữ liệu
#### Mục đích
Định dạng ảnh để phù hợp với đầu vào của SVM, bao gồm chuyển đổi sang ảnh xám, tăng cường dữ liệu và trích xuất đặc trưng HOG.

```python
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

target_size = (64, 64)

# Chuyển đổi ảnh sang ảnh xám
def to_grayscale(images):
    grayscale_images = []
    for img in images:
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        grayscale_images.append(img_gray)
    return np.array(grayscale_images)

# Tăng cường dữ liệu bằng cách lật ảnh
def augment_data(images):
    augmented_images = []
    for img in images:
        img_flipped = cv2.flip(img, 1)
        augmented_images.append(img)
        augmented_images.append(img_flipped)
    return np.array(augmented_images)

# Trích xuất đặc trưng HOG từ ảnh
def extract_hog_features(images):
    hog_features = []
    for img in images:
        img_resized = cv2.resize(img, target_size)
        features = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

# Tăng cường dữ liệu
pos_samples = augment_data(pos_samples)
neg_samples = augment_data(neg_samples)

# Chuyển đổi ảnh sang ảnh xám
pos_samples = to_grayscale(pos_samples)
neg_samples = to_grayscale(neg_samples)

# Trích xuất đặc trưng HOG
X_pos_hog = extract_hog_features(pos_samples)
X_neg_hog = extract_hog_features(neg_samples)

# Gộp các tập dữ liệu
X = np.vstack((X_pos_hog, X_neg_hog))

# Tạo nhãn: 1 cho các mẫu dương và 0 cho các mẫu âm
y = np.hstack((np.ones(X_pos_hog.shape[0]), np.zeros(X_neg_hog.shape[0])))
```
- `to_grayscale`: Hàm này chuyển đổi ảnh màu sang ảnh xám.
- `augment_data`: Hàm này tăng cường dữ liệu bằng cách lật ảnh.
- `extract_hog_features`: Hàm này trích xuất đặc trưng HOG từ các ảnh.
- `cv2.resize`: Hàm này từ OpenCV để thay đổi kích thước ảnh.
- `hog`: Hàm từ skimage để trích xuất đặc trưng HOG.

## Phần 2: Phân loại SVM

### Bước 2.1: Huấn luyện và kiểm tra SVM
#### Mục đích
Huấn luyện SVM và kiểm tra trên dữ liệu. Tìm giá trị C tốt nhất để tối ưu hóa mô hình.

```python
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Huấn luyện và đánh giá mô hình SVM
def train_model(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    C_values = np.logspace(-3, 3, 10)
    best_accuracy = 0
    best_C = 0
    best_clf = None

    for C in C_values:
        clf = svm.SVC(kernel='linear', C=C, probability=True)
        clf.fit(X_train, y_train)

        y_val_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Độ chính xác với C={C}: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C
            best_clf = clf

    print(f"Giá trị C tốt nhất: {best_C} với độ chính xác: {best_accuracy}")

    W = best_clf.coef_[0]
    b = best_clf.intercept_[0]

    return scaler, best_clf, W, b, X_train, y_train, X_val, y_val

# Huấn luyện mô hình
scaler, best_clf, W, b, X_train, y_train, X_val, y_val = train_model(X, y)
```
- `train_model`: Hàm này huấn luyện và đánh giá mô hình SVM. Tìm giá trị C tốt nhất để tối ưu hóa mô hình.
- `svm.SVC`: Hàm từ sklearn để tạo mô hình SVM.
- `train_test_split`: Hàm này chia dữ liệu thành các tập huấn luyện, kiểm tra và xác nhận.

### Bước 2.2: Tính toán lại confidence values
#### Mục đích
Tính toán lại giá trị confidence cho tập huấn luyện và tập validation sử dụng \(W\) và bias \(b\).

```python
# Tính toán giá trị confidence
def compute_confidence(X, W, b):
    return np.dot(X, W) + b

# Tính lại các giá trị confidence cho tập huấn luyện và tập validation
train_confidences = compute_confidence(X_train, W, b)
val_confidences = compute_confidence(X_val, W, b)

train_accuracy = accuracy_score(y_train, train_confidences > 0)
val_accuracy = accuracy_score(y_val, val_confidences > 0)

print(f"Độ chính xác trên tập huấn luyện: {train_accuracy}")
print(f"Độ chính xác trên tập validation: {val_accuracy}")
```
- `compute_confidence`: Hàm này tính toán giá trị confidence cho các mẫu dựa trên W và b.
- `accuracy_score`: Hàm từ sklearn để tính độ chính xác của mô hình.

### Bước 2.3: Trực quan hóa \(W\)
#### Mục đích
Hiển thị \(W\) như một hình ảnh để quan sát hình dạng của nó và hiểu tại sao nó trông giống như một khuôn mặt.

```python
import matplotlib.pyplot as plt

# Chuyển đổi W để trực quan hóa như một ảnh
if W.shape[0] == target_size[0] * target_size[1]:
    W_image = W.reshape(target_size)
    plt.imshow(W_image, cmap='gray')
    plt.title(f"Trực quan hóa W với C={best_clf.C}")
    plt.show()
else:
    print(f"Không thể chuyển đổi mảng kích thước {W.shape[0]} thành kích thước {target_size}")
```
- `plt.imshow`: Hàm từ matplotlib để hiển thị hình ảnh.
- `W.reshape`: Chuyển đổi W thành kích thước của ảnh để trực quan hóa.

### Bước 2.4: Lý giải và thử nghiệm
#### Mục đích
Lý giải tại sao giá trị \(W\) nhỏ cho \(C\) lại trông giống khuôn mặt hơn và tại sao không thể sử dụng ảnh trung bình như một hyper-plane.

```markdown
- Khi giá trị \(C\) nhỏ, SVM sẽ ít nhạy cảm hơn với các lỗi trên dữ liệu huấn luyện, điều này có thể dẫn đến việc tìm kiếm một hyper-plane tương tự với khuôn mặt trung bình.
- Tuy nhiên, việc sử dụng ảnh trung bình như một hyper-plane sẽ không đảm bảo tính phân biệt giữa các mẫu dương và âm.
```
- \(C\) nhỏ làm SVM ít nhạy cảm với lỗi, dẫn đến hyper-plane giống khuôn mặt trung bình.
- Ảnh trung bình không đảm bảo tính phân biệt giữa các mẫu.

## Phần 3: Phát hiện khuôn mặt
### Bước 3.1: Phát hiện khuôn mặt với kỹ thuật “scanning-window”
#### Mục đích
Áp dụng kỹ thuật phát hiện khuôn mặt “scanning-window” trên các ảnh kiểm tra. Sử dụng non-maxima suppression (NMS) để loại bỏ các phản hồi trùng lặp.

```python
import os
import cv2

def nms(box

es, scores, threshold):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep

# Phát hiện khuôn mặt trên các ảnh kiểm tra
def detect_faces(image_paths, scaler, best_clf, W, b, confthresh=0.5, confthreshnms=0.3, step_size=4):
    for img_path in image_paths:
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                window_sizes = range(64, min(img.shape[:2]) - 5, 10)
                detections = []

                for window_size in window_sizes:
                    for y in range(0, img.shape[0] - window_size, step_size):
                        for x in range(0, img.shape[1] - window_size, step_size):
                            window = img[y:y + window_size, x + window_size]
                            window_resized = cv2.resize(window, target_size)
                            window_hog = extract_hog_features([window_resized])[0]
                            window_hog = scaler.transform([window_hog])
                            score = best_clf.decision_function(window_hog)
                            if score > confthresh:
                                detections.append((x, y, score[0], window_size))

                if detections:
                    boxes = np.array([[x, y, x + ws, y + ws] for (x, y, score, ws) in detections])
                    scores = np.array([score for (x, y, score, ws) in detections])
                    keep_indices = nms(boxes, scores, confthreshnms)

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    for i in keep_indices:
                        (x1, y1, x2, y2) = boxes[i]
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        print(f"Phát hiện trong {img_path}: ({x1}, {y1}), ({x2}, {y2}), điểm số: {scores[i]}")

                    plt.imshow(img_rgb)
                    plt.title(f"Phát hiện trong {img_path}")
                    plt.show()
                else:
                    print(f"Không phát hiện thấy gì trong {img_path}.")
            else:
                print(f"Không thể mở file ảnh {img_path}.")
        else:
            print(f"File ảnh không tồn tại: {img_path}.")

# Đường dẫn đến các ảnh kiểm tra
image_paths = [
    'path/to/img1.jpg',
    'path/to/img2.jpg',
    'path/to/img3.jpg',
    'path/to/img4.jpg'
]

# Phát hiện khuôn mặt trên các ảnh kiểm tra
detect_faces(image_paths, scaler, best_clf, W, b)
```
- `nms`: Hàm này thực hiện non-maxima suppression để loại bỏ các phát hiện trùng lặp.
- `detect_faces`: Hàm này phát hiện khuôn mặt trên các ảnh kiểm tra sử dụng kỹ thuật scanning-window và NMS.

### Bước 3.2: Thử nghiệm với các ngưỡng khác nhau
#### Mục đích
Thử nghiệm phát hiện khuôn mặt với các giá trị ngưỡng khác nhau cho cả bước tiền chọn lọc và bước NMS.

```python
# Thử nghiệm phát hiện khuôn mặt với các ngưỡng khác nhau
for conf_thresh in [0.2, 0.5, 0.8]:
    for nms_thresh in [0.2, 0.5, 0.8]:
        print(f"Thử nghiệm với ngưỡng confidence {conf_thresh} và ngưỡng NMS {nms_thresh}")
        detect_faces(image_paths, scaler, best_clf, W, b, conf_thresh, nms_thresh)
```
- Thử nghiệm với các giá trị ngưỡng khác nhau để tìm ra các tham số tốt nhất cho việc phát hiện khuôn mặt.

## Kết luận
Qua các bước trên, em đã xây dựng và triển khai một hệ thống phát hiện khuôn mặt cơ bản sử dụng SVM và các đặc trưng HOG. Tuy nhiên mô hình này chưa thể phát hiện khuôn mặt trong các ảnh kiểm tra với độ chính xác cao :(( mặc dù để sử dụng phương pháp loại bỏ các phát hiện trùng lặp bằng kỹ thuật NMS.
