# Naive Bayes

This repository contains a Naive Bayes classifier built from scratch in Python, using Gaussian probability density functions for continuous features. Naive Bayes is a probabilistic classifier based on Bayes' theorem, assuming independence between features.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
    - [Class Prior](#class-prior)
    - [Likelihood](#likelihood)
    - [Posterior Probability](#posterior-probability)
3. [Naive Bayes Algorithm Example](#naive-bayes-algorithm-example)
4. [Detailed Components & Calculations](#detailed-components-and-calculations)
    - [Fit Method](#fit-method)
    - [Predict Method](#predict-method)

## Introduction

Naive Bayes classifiers are a family of probabilistic classifiers that apply Bayes' theorem with strong independence assumptions. This implementation utilizes Gaussian distributions to estimate the likelihood of each feature given the class, making it suitable for continuous data.

## Mathematical Foundations

### Class Prior

The prior probability represents the relative frequency of each class in the dataset. For each class $c $, the prior $P(y = c)$ is calculated as:

$$
P(y = c) = \frac{\text{number of samples in class } c}{\text{total number of samples}}
$$

### Likelihood

Assuming features follow a Gaussian distribution, the likelihood $ P(x_i | y = c) $ for a feature $ x_i $ given class $ c $ is:

$$
P(x_i | y = c) = \frac{1}{\sqrt{2 \pi \sigma_c^2}} \exp \left(- \frac{(x_i - \mu_c)^2}{2 \sigma_c^2} \right)
$$

where:
- $\mu_c $ is the mean of the feature values for class $ c $,
- $\sigma_c^2 $ is the variance of the feature values for class $ c $.

### Posterior Probability

Using Bayes' theorem, the posterior probability for a class $ c $ given input $ x $ is calculated as:

$$
P(y = c | x) \propto P(y = c) \prod_{i=1}^d P(x_i | y = c)
$$

The model predicts the class with the highest posterior probability.

## Naive Bayes Algorithm Example

1. **Input Data**: A dataset $ X $ with shape $ (n, d) $, where $ n $ is the number of samples and $ d $ is the number of features.
2. **Classes**: The model supports multiple classes, assuming Gaussian distributions for each feature per class.

## Detailed Components and Calculations

### Fit Method

The `fit` method calculates and stores parameters for each class:

1. **Mean**: Calculates the mean $ \mu_c $ of each feature for each class.
2. **Variance**: Calculates the variance $ \sigma_c^2 $ of each feature for each class.
3. **Prior**: Calculates the prior $ P(y = c) $ for each class based on class frequencies.

### Predict Method

The `predict` method calculates posterior probabilities for each class and selects the class with the highest probability:

1. **Compute Log-Prior**: Calculates the log of the prior $ \\log(P(y = c)) $ for numerical stability.
2. **Log-Likelihood Calculation**: Calculates the sum of the log of the likelihoods for each feature.
3. **Combine Posterior Components**: Adds the log-prior and log-likelihoods to compute the posterior for each class.
4. **Prediction**: Returns the class with the maximum posterior probability.