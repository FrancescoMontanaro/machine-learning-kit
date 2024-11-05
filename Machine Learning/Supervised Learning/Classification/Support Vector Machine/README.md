# Support Vector Machine (SVM)

This repository contains a Support Vector Machine (SVM) classifier built from scratch in Python. The SVM model is implemented with a hinge loss function and a regularization parameter, making it suitable for binary classification tasks.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
    - [Objective Function](#objective-function)
    - [Hinge Loss](#hinge-loss)
    - [Regularization](#regularization)
3. [SVM Algorithm Example](#svm-algorithm-example)
4. [Detailed Components & Calculations](#detailed-components-and-calculations)
    - [Fit Method](#fit-method)
    - [Predict Method](#predict-method)
    - [Visualization](#visualization)

## Introduction

Support Vector Machines (SVMs) are supervised learning models for binary classification that find a hyperplane separating the two classes with maximum margin. This implementation uses a linear SVM with stochastic gradient descent optimization.

## Mathematical Foundations

### Objective Function

The objective of the SVM is to find a hyperplane $ w \cdot x - b = 0 $ that maximizes the margin between the two classes. For a given sample $ x_i $ with label $ y_i \in \{-1, 1\} $, the decision boundary is found by minimizing:

$$
\text{Objective} = \frac{1}{2} ||w||^2 + \lambda \sum_{i=1}^n \max(0, 1 - y_i (w \cdot x_i - b))
$$

where:
- $ w $ is the weight vector defining the hyperplane,
- $ b $ is the bias term,
- $ \lambda $ is the regularization parameter controlling overfitting.

### Hinge Loss

The hinge loss function $ \max(0, 1 - y_i (w \cdot x_i - b)) $ penalizes misclassified points or points within the margin. This loss encourages the model to correctly classify points with a margin of at least 1.

### Regularization

The term $ \frac{1}{2} ||w||^2 $ acts as regularization, preventing the model from fitting the data too closely (overfitting). The regularization parameter $ \lambda $ controls the balance between maximizing the margin and minimizing classification errors.

## SVM Algorithm Example

1. **Input Data**: A dataset $ X $ with shape $ (n, d) $, where $ n $ is the number of samples and $ d $ is the feature dimension.
2. **Labels**: Binary labels $ y \in \{-1, 1\} $.
3. **Hyperparameters**: Learning rate $ \text{lr} $, regularization parameter $ \lambda $, and number of epochs.

## Detailed Components and Calculations

### Fit Method

The `fit` method trains the SVM model using stochastic gradient descent:

1. **Weight Initialization**: Initializes $ w $ randomly and $ b $ as zero.
2. **Condition Check**: For each sample $ x_i $, checks whether the margin condition $ y_i (w \cdot x_i - b) \geq 1 $ is satisfied.
3. **Weight Update**:
   - If the condition is met, only the regularization term updates the weights.
   - Otherwise, both the regularization and hinge loss terms contribute to weight and bias updates.

### Predict Method

The `predict` method classifies new data points based on the sign of $ w \cdot x - b $:

1. **Linear Combination**: Computes $ w \cdot x - b $.
2. **Sign Function**: Assigns class $ 1 $ if positive, and class $ -1 $ if negative.

### Visualization

The `visualize` method plots the data points and the SVM decision boundary:

1. **Decision Boundary**: Plots the main decision hyperplane $ w \cdot x - b = 0 $.
2. **Margin Hyperplanes**: Plots the margin boundaries $ w \cdot x - b = \pm 1 $.
3. **Data Points**: Visualizes the data points with labels, providing an intuitive view of classification regions.