import cv2
import numpy as np

def augment_data(images):
    augmented_images = []
    for img in images:
        img_flipped = cv2.flip(img, 1)
        augmented_images.append(img)
        augmented_images.append(img_flipped)
    return np.array(augmented_images)
