import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Function to decode masks
def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# Function to load test images
def load_test_images(data_dir, image_filenames):
    images = []

    for img_filename in image_filenames:
        img_path = os.path.join(data_dir, img_filename)

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    return np.array(images)

# Load bounding boxes data

bboxes = pd.read_csv('train_ship_segmentations_v2.csv')

# Load test data
test_data_dir = 'test'
test_image_filenames = [f for f in os.listdir(test_data_dir) if f.endswith('.jpg')]
x_test = load_test_images(test_data_dir, test_image_filenames)

# Load the trained U-Net model
model = tf.keras.models.load_model('semantic_segmentation_model.h5')

# Perform inference on test data
y_pred_test = model.predict(x_test, batch_size=4)

# Visualize the results
for i in range(len(test_image_filenames)):
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[i])
    plt.title('Original Image')

    # Predicted Mask
    plt.subplot(1, 2, 2)
    plt.imshow(y_pred_test[i, :, :, 0], cmap='gray')  # Use cmap='gray' for binary masks
    plt.title('Predicted Mask')

    plt.show()
