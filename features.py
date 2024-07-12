import cv2
import numpy as np
from skimage.feature import hog

target_size = (64, 64)

def extract_hog_features(images):
    hog_features = []
    for img in images:
        img_resized = cv2.resize(img, target_size)
        features = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        hog_features.append(features)
    return np.array(hog_features)
