
# K-Nearest Neighbors

This repository contains a K-Nearest Neighbors (KNN) implementation built from scratch using only the Numpy library, focusing on minimal dependencies and an in-depth explanation of the algorithm. The code calculates distances, identifies neighbors, and predicts labels for a given query point.

## Table of Contents

1. [Introduction](#introduction)
2. [Running Example: 2D Dataset](#running-example)
3. [Components & Detailed Calculations](#components-and-detailed-calculations)
    - [Distance Calculation](#distance-calculation)
    - [Neighbor Selection](#neighbor-selection)
    - [Prediction](#prediction)

## Introduction

The K-Nearest Neighbors (KNN) algorithm is a simple, instance-based learning method used for classification and regression tasks. Given a query point, KNN finds the **K** closest points in the training data and predicts the output based on these neighbors' labels. This implementation emphasizes efficient calculations and a clear understanding of each step.

## Running Example: 2D Dataset

For demonstration, assume a dataset of 2D points, each labeled as one of two classes (e.g., A or B). Given a query point, KNN identifies the closest points and predicts its label based on the most common label among them.

### Example Details

1. **Dataset**: A set of points in a 2D space with known labels.
2. **Query Point**: The point we want to classify based on the nearest neighbors.
3. **K**: The number of neighbors to consider for prediction.

## Components and Detailed Calculations

### Distance Calculation

The **distance calculation** function computes the Euclidean distance between the query point and each point in the training dataset.

- **Euclidean Distance Formula**:
  $d(x_i, x_j) = \sqrt{\sum_{k=1}^{d} (x_{i,k} - x_{j,k})^2}$
  where  $x_i$ and $x_j$ are data points, and $d$ is the number of dimensions.

Example:
```python
# Calculates the Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
```

This function enables KNN to efficiently calculate distances between the query point and all points in the dataset.

### Neighbor Selection

The **neighbor selection** step identifies the **K** points in the dataset with the smallest distances to the query point.

- **Selecting Neighbors**: Sort the distances in ascending order and select the K smallest values.

This step sorts points based on distance, then retrieves the K closest points.

### Prediction

The **prediction** function uses the selected neighbors' labels to determine the output for the query point.

- **Classification**: For classification tasks, the predicted label is the most common label among the K neighbors. This step applies majority voting to classify the query point, balancing simplicity with effectiveness.

    $y_{pred} = \text{argmax}(\sum_{i=1}^{K} \mathbb{1}(y_i = c))$ where $y_i$ is the label of the $i$-th neighbor, and $c$ is a class label.

- **Regression**: For regression tasks, the predicted value is the average of the K neighbors' values. This approach leverages the neighbors' values to estimate the query point's output.

    $y_{pred} = \frac{1}{K} \sum_{i=1}^{K} y_i$ where $y_i$ is the value of the $i$-th neighbor.