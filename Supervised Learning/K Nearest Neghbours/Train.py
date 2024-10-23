import numpy as np
from knn import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Loading the dataset
data = datasets.load_breast_cancer()

# Extracting the features and the labels
x, y = data.data, data.target # type: ignore

# Splitting the dataset into the training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# Creating the model
model = KNN(k=5)

# Fitting the model
model.fit(x_train, y_train)

# Predicting the labels
y_pred = model.predict(x_test)

# Computing the accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)

# Printing the accuracy
print("Accuracy:", accuracy)