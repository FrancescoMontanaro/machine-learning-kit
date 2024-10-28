
# Logistic Regression

This repository contains an implementation of logistic regression built from scratch in Python, focusing on minimal dependencies and a clear, step-by-step explanation of each component and calculation involved.

## Table of Contents

1. [Introduction](#introduction)
2. [Running Example: Binary Classification Dataset](#running-example)
3. [Components & Detailed Calculations](#components-and-detailed-calculations)
    - [Hypothesis Function](#hypothesis-function)
    - [Cost Function](#cost-function)
    - [Gradient Descent](#gradient-descent)

## Introduction

Logistic regression is a supervised learning algorithm used primarily for binary classification tasks. It models the probability of an observation belonging to a specific class and uses a sigmoid function to map inputs to a probability value between 0 and 1. This implementation utilizes **gradient descent** to find optimal model parameters.

## Running Example: Binary Classification Dataset

For demonstration, assume a binary classification dataset, where each data point is labeled as either 0 or 1. We aim to learn a decision boundary that separates the classes by maximizing the accuracy of predictions.

### Example Details

1. **Dataset**: A set of labeled points in a feature space with binary labels.
2. **Objective**: Classify points by predicting their class probability, based on feature inputs.

## Components and Detailed Calculations

### Hypothesis Function

The hypothesis function uses the **sigmoid function** to convert a linear combination of inputs into a probability.

- **Sigmoid Function**:
  $h(x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$
  where $w$ is the weight vector, $x$ is the input feature vector, and $b$ is the bias term.

Example:
```python
# Hypothesis function to predict probability using the sigmoid activation
def predict(self, x: np.ndarray) -> np.ndarray:    
    # Computing the linear predictions
    linear_predictions = np.dot(x, self.weights) + self.bias

    # Computing the sigmoid predictions
    y_pred = 1 / (1 + np.exp(-linear_predictions))

    # Converting the predictions to binary values
    class_predictions = [0 if y <= 0.5 else 1 for y in y_pred]

    # Returning the predictions
    return np.array(class_predictions)
```

This function calculates the predicted probability by applying the sigmoid function to the linear combination of features.

### Cost Function

The cost function measures the error in predictions, defined here as the **Binary Cross-Entropy Loss**. This function penalizes incorrect classifications more heavily as predictions approach certainty.

- **Binary Cross-Entropy Loss**:
  $J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h(x^{(i)})) + (1 - y^{(i)}) \log(1 - h(x^{(i)})) \right]$
  where $m$ is the number of training examples, $h(x^{(i)})$ is the predicted probability, and $y^{(i)}$ is the actual label.

Example:
```python
# Computes the binary cross-entropy loss for given predictions and actual values
def compute_cost(self, X, y):
    predictions = self.predict_proba(X)
    return -(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)).mean()
```

This function calculates the average cross-entropy between predicted probabilities and actual labels.

### Gradient Descent

The **gradient descent** algorithm iteratively updates weights and bias to minimize the cost function by adjusting each parameter in the direction of the negative gradient.

- **Weight Update Rule**:
  $w := w - \alpha \frac{\partial J}{\partial w}$ = $w - \alpha \frac{2}{n} X^T (Xw - y)$
- **Bias Update Rule**:
  $b := b - \alpha \frac{\partial J}{\partial b}$ = $b - \alpha \frac{2}{n} \sum_{i=1}^{n} (h(x^{(i)}) - y^{(i)})$

where $\alpha$ is the learning rate.

Example:
```python
# Performs gradient descent to update weights and bias based on the learning rate
def fit(self, x: np.ndarray, y: np.ndarray) -> None:    
    # Extracting the number of samples and features
    n_samples, n_features = x.shape

    # Iterating over the specified number of epochs
    for _ in range(self.epochs):
        # Computing the linear predictions
        linear_predictions = np.dot(x, self.weights) + self.bias

        # Computing the sigmoid predictions
        y_pred = self._sigmoid(linear_predictions)

        # Computing the gradients with regularization term
        dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        # Updating the weights and the bias
        self.weights -= self.alpha * dw
        self.bias -= self.alpha * db
```

Each iteration calculates gradients for the weights and bias, updating them based on the learning rate to minimize the loss.