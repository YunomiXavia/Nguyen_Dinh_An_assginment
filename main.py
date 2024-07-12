import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_data
from augmentation import augment_data
from preprocessing import normalize_images, to_grayscale
from features import extract_hog_features
from model import train_model
from utils import compute_confidence, nms
from sklearn.metrics import accuracy_score

# Tải dữ liệu
pos_samples, neg_samples = load_data()

# Tăng cường dữ liệu
pos_samples = augment_data(pos_samples)
neg_samples = augment_data(neg_samples)

# Chuẩn hóa ảnh
pos_samples = normalize_images(pos_samples)
neg_samples = normalize_images(neg_samples)

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

# Huấn luyện mô hình
scaler, best_clf, W, b, X_train, y_train, X_val, y_val = train_model(X, y)

# Chuyển đổi W để trực quan hóa như một ảnh
target_size = (64, 64)
num_features = target_size[0] * target_size[1]
if W.shape[0] == num_features:
    W_image = W.reshape(target_size)
    plt.imshow(W_image, cmap='gray')
    plt.title(f"Trực quan hóa W với C={best_clf.C}")
    plt.show()
else:
    print(f"Không thể chuyển đổi mảng kích thước {W.shape[0]} thành kích thước {target_size}")

# Tính lại các giá trị confidence cho tập huấn luyện và tập validation
train_confidences = compute_confidence(X_train, W, b)
val_confidences = compute_confidence(X_val, W, b)

train_accuracy = accuracy_score(y_train, train_confidences > 0)
val_accuracy = accuracy_score(y_val, val_confidences > 0)

print(f"Độ chính xác trên tập huấn luyện: {train_accuracy} (độ chính xác càng lớn càng tốt)")
print(f"Độ chính xác trên tập validation: {val_accuracy} (độ chính xác càng lớn càng tốt)")

# Tải ảnh kiểm tra và thực hiện phát hiện
test_images = ['./images/img1.jpg', './images/img2.jpg', './images/img3.jpg', './images/img4.jpg']

confthresh = 1.0
confthreshnms = 0.3
window_sizes = [64, 80, 96]
step_size = 4

for img_path in test_images:
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.GaussianBlur(img, (5, 5), 0)
            detections = []

            for window_size in window_sizes:
                for y in range(0, img.shape[0] - window_size, step_size):
                    for x in range(0, img.shape[1] - window_size, step_size):
                        window = img[y:y + window_size, x:x + window_size]
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
