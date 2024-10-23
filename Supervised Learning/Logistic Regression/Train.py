import numpy as np
from sklearn import datasets
from logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split

# Loading the dataset
data = datasets.load_breast_cancer()

# Extracting the features and the labels
x, y = data.data, data.target # type: ignore

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1234)

# Creating the logistic regression model
logistic_regression = LogisticRegression(epochs=1000)

# Fitting the model
logistic_regression.fit(x_train, y_train)

# Predicting the labels
y_pred = logistic_regression.predict(x_test)

# Computing the accuracy of the predictions
accuracy = np.sum(y_pred == y_test) / len(y_test)

# Printing the accuracy
print("Accuracy:", accuracy)