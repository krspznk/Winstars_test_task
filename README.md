
Semantic Segmentation Model
This repository contains my solution for a semantic segmentation task using a U-Net architecture and TensorFlow's tf.keras. The goal was to create a model capable of segmenting objects in images, specifically focusing on ship segmentation in this case.

Project Structure
The project is organized as follows:

train_model.py: Python script for model training.
inference.py: Python script for model inference on test data.
data_analysis.ipynb: Jupyter notebook containing exploratory data analysis.
README.md: This file, providing an overview of the project.
Model Overview
The U-Net architecture was chosen for this semantic segmentation task. The model consists of an encoder-decoder structure with skip connections, allowing it to capture both local and global features.

Data
The training data comes from a dataset containing images of ships and corresponding segmentation masks. The dataset was preprocessed to load images and masks, and a custom data augmentation strategy was employed to enhance model generalization.

Challenges Faced
Due to resource limitations on my personal machine, I had to reduce the dataset size significantly. This reduction may impact the model's performance on real-world, large-scale data. The primary aim was to showcase my ability to use machine learning tools and techniques rather than achieving state-of-the-art accuracy.

Model Performance
The model's performance may not be optimal due to the reduced dataset size. Additionally, the training duration was limited by hardware constraints. As a result, the primary focus is on demonstrating proficiency in machine learning workflows and tools rather than achieving the highest accuracy.
