# CIFAR-10 Image Classification: CNN vs ResNet50
This project compares the performance of a baseline Convolutional Neural Network (CNN) and a deeper ResNet50 architecture on the CIFAR-10 dataset.

## Overview

The aim was to evaluate how model complexity affects classification performance on a standard multi-class computer vision benchmark, using accuracy, precision, recall, and F1-score as evaluation metrics. 

* CIFAR-10 (10 classes of images)
* 60,000 images (50k train, 10k test)

## Models Compared

* CNN (baseline model)
* ResNet-50-based model

## Results

The models produced the following results on CIFAR-10:

|Model|Accuracy|Precision|Recall|F1-score|
|-|-|-|-|-|
|CNN|0.6912|0.6966|0.6912|0.6926|
|ResNet50|0.8804|0.8828|0.8804|0.8796|

The ResNet50-based model showed a clear performance improvement over the baseline CNN across all evaluation metrics.

ResNet50 significantly outperformed the baseline CNN, demonstrating the effectiveness of deeper architectures for image classification tasks.

## Key Results

- CNN achieved ~69% accuracy
- ResNet50 achieved ~88% accuracy
- ResNet50 significantly outperformed the baseline model

ResNet50 demonstrates the benefit of deeper architectures for image classification tasks compared to baseline CNN models.

## Outputs

The repository includes:

\- evaluation metrics (`model\_metrics.csv`)

\- learning curve plots

\- model comparison plots



Note: trained model weight files are not included in the repository due to file size constraints.



## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib

## Purpose

This project demonstrates:

* Model comparison
* Deep learning fundamentals
* Evaluation metrics (accuracy, precision, recall, F1-score)

## Note
This project was developed as part of my MSc studies and demonstrates deep learning model comparison on image classification tasks.