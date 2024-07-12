import cv2
import numpy as np

def normalize_images(images):
    normalized_images = []
    for img in images:
        img_normalized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        normalized_images.append(img_normalized)
    return np.array(normalized_images)

def to_grayscale(images):
    grayscale_images = []
    for img in images:
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        grayscale_images.append(img_gray)
    return np.array(grayscale_images)
