
# Machine Learning from Scratch

This repository contains implementations of various machine learning algorithms developed from scratch for educational purposes. Each algorithm is implemented in Python and organized according to its category (Supervised, Unsupervised, or Reinforcement Learning). The code follows a clear structure, with each folder containing separate modules for the core algorithm and training routines.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Algorithms](#algorithms)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
- [Usage](#usage)

---

## Overview

This repository serves as a comprehensive collection of foundational machine learning algorithms, with each algorithm implemented from scratch. Each model is accompanied by its training pipeline, allowing to test the algorithms on data.

## Requirements

Ensure you have Python 3.11 or later. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/FrancescoMontanaro/Machine-learning-kit.git
   cd repo-name
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Folder Structure

The project structure is organized into categories and subcategories for each algorithm, as shown below:

```plaintext
├── Reinforcement Learning
│   └── Multi Armed Bandit
├── Supervised Learning
│   ├── Classification
│   │   ├── Logistic Regression
│   │   ├── Naive Bayes
│   │   └── Support Vector Machine
│   ├── Regression
│   │   └── Linear Regression
│   ├── General Algorithms
│   │   ├── K Nearest Neighbours
│   │   ├── Perceptron
│   │   └── Random Forests
│   └── Deep Learning
│       ├── Neural Networks
│       └── Transformer
├── Unsupervised Learning
│   ├── Clustering
│   │   └── K Means Clustering
│   └── Dimensionality Reduction
│       └── Principal Component Analysis
└── requirements.txt
```

## Algorithms

### Reinforcement Learning

1. **Multi-Armed Bandit** - Implementations of algorithms for the Multi-Armed Bandit problem, such as:
   - **Upper Confidence Bound (UCB1)**: `ucb1.py`
   - **Thompson Sampling**: `thompson_sampling.py`
   - Environment and utilities in `environment.py` and `utils.py`.

### Supervised Learning

1. **K-Nearest Neighbours (KNN)**: An implementation of the K-Nearest Neighbors algorithm (`knn.py`).
2. **Linear Regression**: Linear regression from scratch with closed-form solutions (`linear_regression.py`).
3. **Logistic Regression**: Binary classification using logistic regression (`logistic_regression.py`).
4. **Naive Bayes**: A Naive Bayes classifier for text or other categorical data (`naive_bayes.py`).
5. **Perceptron**: A single-layer perceptron implementation for binary classification (`perceptron.py`).
6. **Random Forest**: Ensemble method using multiple decision trees (`random_forest.py`).
7. **Support Vector Machine (SVM)**: A linear SVM classifier using custom optimization (`svm.py`).
8. **Neural Networks**: Feedforward neural network with different layers, activation functions, loss functions and  the implementation of the backpropagation algorithm (`neural_network.py`).
9. **Transformer**: A custom transformer implementation for language modeling, including:
   - **Attention Mechanism** (`attention_mechanism.py`)
   - **Feed-Forward Network** (`feed_forward.py`)
   - **Regularization** (`regularization.py`)
   - **Data Handling and Utilities** (`data_loader.py`, `utils.py`)

### Unsupervised Learning

1. **K-Means Clustering**: Basic implementation of the K-Means algorithm for clustering tasks (`k_means.py`).
2. **Principal Component Analysis (PCA)**: Dimensionality reduction technique for data visualization and preprocessing (`pca.py`).

## Usage

Each algorithm comes with a `train.py` script that can be used to train the model with sample data. For example, to train the Linear Regression model:

```bash
cd Supervised\ Learning/Regression/Linear\ Regression
python train.py
```

For algorithms that involve reinforcement learning, such as the Multi-Armed Bandit, the `train.py` script simulates the environment and evaluates the agent’s performance.