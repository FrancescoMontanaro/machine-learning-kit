
# Perceptron

This repository includes a Perceptron model implemented from scratch using Python with minimal dependencies. The code contains a simple, step-by-step approach to training a binary classifier using the perceptron learning algorithm.

## Table of Contents

1. [Introduction](#introduction)
2. [Perceptron Algorithm Example](#perceptron-algorithm-example)
3. [Detailed Components & Calculations](#detailed-components-and-calculations)
    - [Weight Initialization](#weight-initialization)
    - [Activation Function](#activation-function)
    - [Model Training](#model-training)
    - [Prediction](#prediction)

## Introduction

The perceptron is a foundational binary classifier, useful for linearly separable data. It updates its weights based on a simple rule: adjusting weights for each misclassified data point until the model correctly classifies the training data or reaches a maximum iteration count. Here, we use a simple example to explain each computational step.

## Perceptron Algorithm Example

The following example illustrates the core steps of the perceptron learning algorithm, from initialization to training and prediction.

### Example Details

1. **Input Data**: A dataset containing two features per sample, represented as a matrix $X$ with shape $(n, d)$, where $n$ is the number of samples and $d$ is the number of features.
2. **Labels**: Binary labels for each sample, represented as a vector $y \in \{-1, 1\}^n$.
3. **Learning Rate**: The learning rate $\eta$, set to control weight update steps.
4. **Epochs**: The number of training iterations (set to 100 by default).

## Detailed Components and Calculations

### Weight Initialization

The perceptron begins by initializing weights and a bias term to zero. The weight vector $\mathbf{w}$ has the same dimension as the feature space, and the bias term $b$ is a scalar.

- **Weight Vector $\mathbf{w}$**: Initialized as a vector of zeros with shape $(d,)$.
- **Bias $b$**: Initialized to zero.

### Activation Function

The perceptron uses a simple step function as its activation function. The output $f(x)$ for a given input $x$ is determined by:

$$
f(x) = \text{sign}(\mathbf{w} \cdot x + b)
$$

This function returns $1$ for positive inputs and $-1$ for negative ones.

### Model Training

During training, the model iterates through the dataset and updates weights based on misclassified points. For each sample $x_i$ with label $y_i$, the perceptron performs the following steps:

1. **Prediction**: Compute the output $\hat{y}_i = \text{sign}(\mathbf{w} \cdot x_i + b)$.
2. **Weight Update**: If $\hat{y}_i \neq y_i$, update the weights and bias:
   $$
   \mathbf{w} = \mathbf{w} + \eta \cdot y_i \cdot x_i
   $$
   $$
   b = b + \eta \cdot y_i
   $$

The model continues these updates for each sample in each epoch until all points are correctly classified or the maximum number of epochs is reached.

### Prediction

Once trained, the perceptron predicts new samples using the learned weights and bias:

$$
f(x) = \text{sign}(\mathbf{w} \cdot x + b)
$$

This function returns binary classifications ($-1$ or $1$), depending on the sign of the result.
