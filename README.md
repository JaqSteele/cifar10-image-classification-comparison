# CIFAR-10 Image Classification (CNN vs ResNet50)

## Overview
This project compares a basic Convolutional Neural Network (CNN) with a ResNet50-based model on the CIFAR-10 dataset.

The goal is to demonstrate the performance difference between a simple deep learning model and a more advanced architecture.

## Dataset
- CIFAR-10 (10 classes of images)
- 60,000 images (50k train, 10k test)

## Models Compared
- CNN (baseline model)
- ResNet-50-based model

## Results
The models produced the following results on CIFAR-10:

| Model | Accuracy | Precision | Recall | F1-score |
|------|----------|-----------|--------|----------|
| CNN | 0.6912 | 0.6966 | 0.6912 | 0.6926 |
| ResNet50 | 0.8804 | 0.8828 | 0.8804 | 0.8796 |

The results showed a clear performance improvement when using the ResNet50-based model compared with the baseline CNN.

## Outputs
- Trained models (.keras files)
- Performance metrics
- Learning curves
- Comparison charts

## Tech Used
- Python
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib

## Purpose
This project demonstrates:
- Model comparison
- Deep learning fundamentals
- Evaluation metrics (accuracy, precision, recall, F1-score)