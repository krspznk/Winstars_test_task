import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


# Function to decode masks
def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


# Function to load images and masks
def load_images(data_dir, image_filenames, bboxes):
    images = []

    for img_filename in image_filenames:
        img_path = os.path.join(data_dir, img_filename)

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    x_train = np.array(images)

    # Load masks
    y_train = []
    for ImageId in image_filenames:
        img_masks = bboxes.loc[bboxes['ImageId'] == ImageId, 'EncodedPixels'].tolist()

        all_masks = np.zeros((768, 768))

        for mask in img_masks:
            if pd.notna(mask) and (len(mask) != 0):
                decoded_mask = rle_decode(mask)
                all_masks += decoded_mask
        y_train.append(all_masks)

    y_train = np.array(y_train)

    return x_train, y_train


# Define U-Net model
def unet(input_size=(768, 768, 3)):
    inputs = tf.keras.Input(input_size)

    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Middle
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Decoder
    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv3)
    concat1 = tf.keras.layers.concatenate([conv2, up1], axis=-1)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(concat1)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    concat2 = tf.keras.layers.concatenate([conv1, up2], axis=-1)
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat2)
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

    output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


# Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load bounding boxes data
bboxes = pd.read_csv('train_ship_segmentations_v2.csv')

# Load data
data_dir = 'train'
image_filenames = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
x_train, y_train = load_images(data_dir, image_filenames, bboxes)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Compile the model
model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Configure callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_weights.h5', save_best_only=True, save_weights_only=True)

# Train the model with data augmentation
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=4),
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, model_checkpoint]
)

# Save the trained model
model.save('semantic_segmentation_model.h5')
