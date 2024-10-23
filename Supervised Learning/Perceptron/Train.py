import numpy as np
from sklearn import datasets
from perceptron import Perceptron
from sklearn.model_selection import train_test_split

# Loading the dataset
x, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2) # type: ignore

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# Creating the perceptron
perceptron = Perceptron(epochs=100, lr=0.01)

# Training the perceptron
perceptron.fit(x_train, y_train)

# Predicting the labels
y_pred = perceptron.predict(x_test)

# Computing the accuracy
accuracy = np.sum(y_test == y_pred) / len(y_test)

# Printing the accuracy
print("Accuracy:", accuracy)