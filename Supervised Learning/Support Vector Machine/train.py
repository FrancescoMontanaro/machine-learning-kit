import numpy as np
from svm import SVM
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Loading the dataset
x, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40) # type: ignore

# Converting the labels to {-1, 1}
y = np.where(y == 0, -1, 1)

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

# Instantiating the SVM model
svm = SVM()

# Fitting the SVM model
svm.fit(x_train, y_train)

# Predicting the labels
y_pred = svm.predict(x_test)

# Computing the accuracy
accuracy = np.sum(y_test == y_pred) / len(y_test)

# Printing the accuracy
print("Accuracy:", accuracy)

# Plotting the decision boundary
svm.visualize(x, y)