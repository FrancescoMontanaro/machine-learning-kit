
# K-means Clustering

This repository contains an implementation of K-means clustering, a popular unsupervised learning algorithm for partitioning data into $k$ clusters. The implementation is built from scratch in Python with minimal dependencies.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
    - [Cluster Assignment](#cluster-assignment)
    - [Centroid Update](#centroid-update)
    - [Convergence Criteria](#convergence-criteria)
3. [K-means Algorithm Example](#k-means-algorithm-example)
4. [Detailed Components & Calculations](#detailed-components-and-calculations)
    - [Initialization](#initialization)
    - [Cluster Assignment Step](#cluster-assignment-step)
    - [Centroid Update Step](#centroid-update-step)
    - [Convergence Check](#convergence-check)

## Introduction

K-means clustering groups $n$ data points into $k$ clusters by iteratively minimizing the variance within each cluster. The algorithm alternates between assigning data points to the nearest cluster centroid and recalculating centroids until convergence.

## Mathematical Foundations

### Cluster Assignment

The algorithm begins by assigning each data point $x_i$ to the nearest centroid $\mu_j$, where $j$ is the index of the centroid. The Euclidean distance is commonly used for measuring proximity:

$$
d(x_i, \mu_j) = ||x_i - \mu_j||
$$

### Centroid Update

After all data points are assigned, centroids are updated to the mean of points within each cluster. For cluster $C_j$, the centroid $\mu_j$ is updated as:

$$
\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
$$

### Convergence Criteria

The algorithm stops when either:
- **Centroids stabilize**: The change in centroids is below a threshold.
- **Maximum iterations**: The algorithm reaches the predefined number of iterations.

## K-means Algorithm Example

An example with $k = 3$ clusters involves:

1. **Input Data**: A dataset $X$ with shape $(n, d)$, where $n$ is the number of samples and $d$ is the feature dimension.
2. **Number of Clusters**: $k = 3$.
3. **Maximum Iterations**: A maximum of 100 iterations.

## Detailed Components and Calculations

### Initialization

In the initialization step, $k$ centroids are selected randomly from the dataset or by another method.

### Cluster Assignment Step

Each data point is assigned to the nearest centroid based on Euclidean distance.

### Centroid Update Step

For each cluster, centroids are recalculated as the mean of the points assigned to that cluster.

### Convergence Check

The algorithm terminates once the centroids do not change significantly or after reaching the maximum number of iterations.
