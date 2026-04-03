# CIFAR-10 Image Classification (CNN vs ResNet50)

## Overview

This project compares a baseline Convolutional Neural Network (CNN) with a ResNet50-based model on the CIFAR-10 image classification dataset.

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

The results showed a clear performance improvement when using the ResNet50-based model compared with the baseline CNN.

## Outputs

The repository includes:

\- evaluation metrics (`model\_metrics.csv`)

\- learning curve plots

\- model comparison plots



Note: trained model weight files are not included in the repository due to file size constraints. Tech Used

* Python
* TensorFlow / Keras
* NumPy / Pandas
* Matplotlib

## Purpose

This project demonstrates:

* Model comparison
* Deep learning fundamentals
* Evaluation metrics (accuracy, precision, recall, F1-score)

