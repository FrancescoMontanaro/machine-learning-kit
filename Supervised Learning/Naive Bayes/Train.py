import numpy as np
from sklearn import datasets
from naive_bayes import NaiveBayes
from sklearn.model_selection import train_test_split

# Loading the dataset
x, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

# Splitting the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

# Creating the Naive Bayes classifier
naive_bayes = NaiveBayes()

# Fitting the model to the data
naive_bayes.fit(x_train, y_train)

# Predicting the labels
y_pred = naive_bayes.predict(x_test)

# Evaluating the model
accuracy = np.sum(y_test == y_pred) / len(y_test)

# Printing the accuracy
print("Accuracy:", accuracy)
