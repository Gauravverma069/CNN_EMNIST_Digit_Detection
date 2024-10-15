# EMNIST Digit and Character Recognition using CNN

This repository contains a project for digit and character recognition using the EMNIST dataset with a Convolutional Neural Network (CNN). The EMNIST dataset is an extension of the popular MNIST dataset and includes handwritten characters in addition to digits.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Results](#results)
5. [Dependencies](#dependencies)

## Introduction

This project aims to classify handwritten digits and characters using the EMNIST dataset, which extends the traditional MNIST dataset to include letters from the English alphabet. A CNN model is trained on this dataset to recognize both digits (0-9) and characters (A-Z).

## Dataset

The EMNIST dataset consists of several splits:
- **EMNIST ByClass**: 814,255 characters, 62 classes
- **EMNIST ByMerge**: 814,255 characters, 47 classes
- **EMNIST Balanced**: 131,600 characters, 47 classes
- **EMNIST Letters**: 145,600 characters, 26 classes
- **EMNIST Digits**: 280,000 characters, 10 classes
- **EMNIST MNIST**: 70,000 characters, 10 classes (same as MNIST)

This project uses the `EMNIST Balanced` split, which contains 47 classes, including digits (0-9) and lowercase and uppercase letters. The dataset is available in the form of images (28x28 pixels) with corresponding labels.

## Model Architecture

The CNN model used for this project consists of the following layers:
1. **Input Layer**: Accepts a 28x28 grayscale image.
2. **Convolutional Layers**: Multiple convolutional layers with ReLU activation functions for feature extraction.
3. **Pooling Layers**: Max-pooling layers to reduce dimensionality.
4. **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
5. **Fully Connected Layers**: Dense layers for classification.
6. **Output Layer**: Softmax activation to predict the probability distribution over 47 classes.

## Dependencies

Make sure to install the following dependencies to run the project:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV (for image preprocessing)
- Scikit-learn

## Results

The model achieves the following accuracy on the EMNIST Balanced dataset:

Training Accuracy: ~100%
Validation Accuracy: ~99.5%

Install these dependencies using `pip`:

```bash
pip install tensorflow keras numpy matplotlib opencv-python scikit-learn

