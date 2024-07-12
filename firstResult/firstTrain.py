import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os

# Load data from .mat files
pos_samples = sio.loadmat('./possamples.mat')['possamples']
neg_samples = sio.loadmat('./negsamples.mat')['negsamples']

# Define target size (100x100)
target_size = (100, 100)

# Resize all positive samples
X_pos_resized = np.array([cv2.resize(img, target_size).flatten() for img in pos_samples])

# Resize all negative samples
X_neg_resized = np.array([cv2.resize(img, target_size).flatten() for img in neg_samples])

# Stack the datasets
X = np.vstack((X_pos_resized, X_neg_resized))

# Create labels: 1 for positive samples and -1 for negative samples
y = np.hstack((np.ones(X_pos_resized.shape[0]), -1 * np.ones(X_neg_resized.shape[0])))

# Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM with default C value
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# Calculate vector W and bias b
W = clf.coef_
b = clf.intercept_


# Define Non-Maxima Suppression (NMS) function
def nms(boxes, scores, threshold):
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


# Load test images
test_images = ['./img1.jpg', './img2.jpg', './img3.jpg', './img4.jpg']

confthresh = 0.5
confthreshnms = 0.3
window_size = 100  # Same as the target size during training
step_size = 8

for img_path in test_images:
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Scanning window over the image
            detections = []

            for y in range(0, img.shape[0] - window_size, step_size):
                for x in range(0, img.shape[1] - window_size, step_size):
                    window = img[y:y + window_size, x:x + window_size]
                    window_flat = window.flatten()
                    window_flat = scaler.transform([window_flat])
                    score = clf.decision_function(window_flat)
                    if score > confthresh:
                        detections.append((x, y, score[0]))

            if detections:
                # Apply NMS
                boxes = np.array([[x, y, x + window_size, y + window_size] for (x, y, score) in detections])
                scores = np.array([score for (x, y, score) in detections])
                keep_indices = nms(boxes, scores, confthreshnms)

                # Visualize detections
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                for i in keep_indices:
                    (x1, y1, x2, y2) = boxes[i]
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

                plt.imshow(img_rgb)
                plt.title(f"Detections in {img_path}")
                plt.show()
            else:
                print(f"No detections found in {img_path}.")
        else:
            print(f"Không thể mở được tệp ảnh {img_path}.")
    else:
        print(f"Tệp ảnh không tồn tại: {img_path}.")
