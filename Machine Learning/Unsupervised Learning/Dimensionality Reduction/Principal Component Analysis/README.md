
# Principal Component Analysis (PCA)

This repository contains an implementation of Principal Component Analysis (PCA) built from scratch in Python with minimal dependencies. PCA is a dimensionality reduction technique that transforms a high-dimensional dataset into a lower-dimensional one by identifying principal components.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
    - [Covariance Matrix](#covariance-matrix)
    - [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
    - [Data Projection](#data-projection)
3. [PCA Algorithm Example](#pca-algorithm-example)
4. [Detailed Components & Calculations](#detailed-components-and-calculations)
    - [Fit Method](#fit-method)
    - [Transform Method](#transform-method)

## Introduction

Principal Component Analysis (PCA) identifies directions (principal components) in the data along which variance is maximized. It reduces dimensionality by projecting data onto a subspace of fewer dimensions, preserving the most important information while minimizing information loss.

## Mathematical Foundations

### Covariance Matrix

The covariance matrix represents the relationships between features in the data. For a dataset $X$ with $n$ samples and $d$ features, we center each feature by subtracting the mean and then calculate the covariance matrix $C$:

$$
C = \frac{1}{n-1} X^T X
$$

The covariance matrix provides insights into feature variances and correlations, which PCA uses to identify directions of maximum variance.

### Eigenvalues and Eigenvectors

PCA identifies principal components by decomposing the covariance matrix into eigenvalues and eigenvectors. Given the covariance matrix $C$, we find the eigenvalues $\lambda$ and eigenvectors $v$:

$$
C v = \lambda v
$$

- **Eigenvalues**: Represent the variance explained by each principal component.
- **Eigenvectors**: Represent the directions of principal components.

Eigenvectors corresponding to the largest eigenvalues define the directions of maximum variance in the data.

### Data Projection

The data is projected onto the principal components by selecting the top $k$ eigenvectors (corresponding to the $k$ largest eigenvalues). Given a matrix $W$ of the top $k$ eigenvectors, the projection of data $X$ onto the new basis is:

$$
X_{\text{proj}} = X W
$$

This projection reduces the dimensionality while retaining most of the dataset's variance.

## PCA Algorithm Example

An example of applying PCA involves:

1. **Input Data**: A dataset $X$ with shape $(n, d)$ where $n$ is the number of samples and $d$ is the number of features.
2. **Number of Components**: $k$, the desired number of principal components.

## Detailed Components and Calculations

### Fit Method

The `fit` method computes the principal components:

1. **Mean Centering**: Calculate the mean of each feature and center the dataset.
2. **Covariance Matrix**: Compute the covariance matrix of the centered data.
3. **Eigen Decomposition**: Find eigenvalues and eigenvectors of the covariance matrix.
4. **Principal Components**: Sort eigenvectors by eigenvalues, selecting the top $k$.

### Transform Method

The `transform` method projects data onto the computed principal components:

1. **Data Centering**: Subtract the mean from the data to center it.
2. **Projection**: Multiply centered data by the matrix of principal components to reduce dimensionality.

