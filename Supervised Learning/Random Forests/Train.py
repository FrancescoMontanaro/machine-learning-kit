import numpy as np
from sklearn import datasets
from random_forest import RandomForest
from sklearn.model_selection import train_test_split

# Loading the dataset
data = datasets.load_breast_cancer()

# Extracting the features and labels
x, y = data.data, data.target # type: ignore

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1234)

# Creating the decision tree
random_forest = RandomForest(n_trees=30, max_depth=150)

# Training the decision tree
random_forest.fit(x_train, y_train)

# Predicting the labels
y_pred = random_forest.predict(x_test)

# Computing the accuracy
accuracy = np.sum(y_test == y_pred) / len(y_test)

# Printing the accuracy
print("Accuracy:", accuracy)