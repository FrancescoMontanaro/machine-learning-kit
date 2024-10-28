
# Linear Regression

This repository contains an implementation of linear regression built from scratch, focusing on minimal dependencies and providing a clear explanation of each computational step involved.

## Table of Contents

1. [Introduction](#introduction)
2. [Running Example: Simple Dataset](#running-example)
3. [Components & Detailed Calculations](#components-and-detailed-calculations)
    - [Hypothesis Function](#hypothesis-function)
    - [Cost Function](#cost-function)
    - [Gradient Descent](#gradient-descent)

## Introduction

Linear regression is a fundamental supervised learning algorithm for predicting a continuous output variable based on one or more input features. This implementation of linear regression is based on **ordinary least squares** and employs **gradient descent** for optimization.

## Running Example: Simple Dataset

For demonstration, assume a small dataset of points, where each input feature has a corresponding output. We aim to fit a line that minimizes the mean squared error between predicted and actual outputs.

### Example Details

1. **Dataset**: A set of data points with one or more input features and a continuous output variable.
2. **Objective**: Minimize the difference between predicted values and actual values by finding optimal model parameters.

## Components and Detailed Calculations

### Hypothesis Function

The hypothesis function computes the predicted output for a given input by applying a linear transformation using learned weights.

- **Hypothesis Function**:
  $h(x) = w^T x + b$
  where $w$ is the weight vector, $x$ is the input feature vector, and $b$ is the bias term.

Example:
```python
# Hypothesis function to predict output based on input and model parameters
def predict(self, X):
    return np.dot(x, self.weights) + self.bias
```

This function calculates the predicted output by taking the dot product of the input features and weights, then adding the bias.

### Cost Function

The cost function measures the error in predictions, defined here as the **Mean Squared Error** (MSE), which we aim to minimize.

- **Mean Squared Error (MSE)**:
  $J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2$
  where $m$ is the number of training examples, $h(x^{(i)})$ is the predicted output, and $y^{(i)}$ is the actual output.

### Gradient Descent

The **gradient descent** algorithm iteratively updates the weights and bias to minimize the cost function, adjusting each parameter in the direction of the negative gradient.

- **Weight Update Rule**:
  $w := w - \alpha \frac{\partial J}{\partial w}$ = $w - \alpha \frac{2}{n} X^T (Xw - y)$
- **Bias Update Rule**:
  $b := b - \alpha \frac{\partial J}{\partial b}$ = $b - \alpha \frac{2}{n} \sum_{i=1}^{n} (h(x^{(i)}) - y^{(i)})$

where $\alpha$ is the learning rate, $X$ is the input matrix, $y$ is the output vector, and $n$ is the number of training examples.

Example:
```python
# Performs gradient descent to update weights and bias based on the learning rate
def fit(self, x: np.ndarray, y: np.ndarray) -> None:
    # Extracting the number of samples and features
    n_samples, n_features = x.shape

    # Iterating over the specified number of epochs
    for _ in range(self.epochs):
        # Computing the predictions
        y_pred = self.predict(x)

        # Computing the Ridge Regression Term
        ridge_w = (2 * self.lambda_reg / n_samples) * self.weights
        ridge_b = (2 * self.lambda_reg / n_samples) * self.bias

        # Computing the gradients with regularization term
        dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Updating the weights and the bias
        self.weights -= self.alpha * dw
        self.bias -= self.alpha * db
```

In each iteration, this function calculates the gradients for weights and bias, updating them according to the learning rate and the gradients.