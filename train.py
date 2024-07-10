import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import scipy.io as sio
import os

# Load data from .mat files
pos_samples = sio.loadmat('./possamples.mat')['possamples']
neg_samples = sio.loadmat('./negsamples.mat')['negsamples']

# Define target size (64x64)
target_size = (64, 64)


# Extract HOG features for positive samples
def extract_hog_features(images):
    hog_features = []
    for img in images:
        img_resized = cv2.resize(img, target_size)
        features, _ = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_features.append(features)
    return np.array(hog_features)


X_pos_hog = extract_hog_features(pos_samples)
X_neg_hog = extract_hog_features(neg_samples)

# Stack the datasets
X = np.vstack((X_pos_hog, X_neg_hog))

# Create labels: 1 for positive samples and 0 for negative samples
y = np.hstack((np.ones(X_pos_hog.shape[0]), np.zeros(X_neg_hog.shape[0])))

# Standardize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM with different C values
C_values = [0.01, 0.1, 1, 10, 100]
best_accuracy = 0
best_C = 0
best_clf = None

for C in C_values:
    clf = svm.SVC(kernel='linear', C=C, probability=True)
    clf.fit(X_train, y_train)

    # Calculate accuracy on validation set
    y_val_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Accuracy with C={C}: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_C = C
        best_clf = clf

print(f"Best C value: {best_C} with accuracy: {best_accuracy}")

# Calculate W and b for the best classifier
W = best_clf.coef_[0]
b = best_clf.intercept_[0]

# Reshape W to visualize as an image
W_image = W.reshape(target_size)
plt.imshow(W_image, cmap='gray')
plt.title(f"Visualization of W for C={best_C}")
plt.show()


# Re-compute confidence values for training and validation
def compute_confidence(X, W, b):
    return np.dot(X, W) + b


train_confidences = compute_confidence(X_train, W, b)
val_confidences = compute_confidence(X_val, W, b)

train_accuracy = accuracy_score(y_train, train_confidences > 0)
val_accuracy = accuracy_score(y_val, val_confidences > 0)

print(f"Training accuracy: {train_accuracy}")
print(f"Validation accuracy: {val_accuracy}")


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

confthresh = 1.0  # You can try different threshold values
confthreshnms = 0.3  # You can try different NMS threshold values
window_sizes = [64, 80, 96]  # Different window sizes
step_size = 4

for img_path in test_images:
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Apply Gaussian filter to smooth the image
            img = cv2.GaussianBlur(img, (5, 5), 0)

            # Scanning window over the image
            detections = []

            for window_size in window_sizes:
                for y in range(0, img.shape[0] - window_size, step_size):
                    for x in range(0, img.shape[1] - window_size, step_size):
                        window = img[y:y + window_size, x:x + window_size]
                        window_resized = cv2.resize(window, target_size)
                        window_hog, _ = hog(window_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                            visualize=True)
                        window_hog = scaler.transform([window_hog])
                        score = clf.decision_function(window_hog)
                        if score > confthresh:
                            detections.append((x, y, score[0], window_size))

            if detections:
                # Apply NMS
                boxes = np.array([[x, y, x + ws, y + ws] for (x, y, score, ws) in detections])
                scores = np.array([score for (x, y, score, ws) in detections])
                keep_indices = nms(boxes, scores, confthreshnms)

                # Visualize detections
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                for i in keep_indices:
                    (x1, y1, x2, y2) = boxes[i]
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                plt.imshow(img_rgb)
                plt.title(f"Detections in {img_path}")
                plt.show()
            else:
                print(f"No detections found in {img_path}.")
        else:
            print(f"Không thể mở được tệp ảnh {img_path}.")
    else:
        print(f"Tệp ảnh không tồn tại: {img_path}.")
