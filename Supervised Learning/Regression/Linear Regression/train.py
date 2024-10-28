import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split

# Loading the dataset
x, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=1234) # type: ignore

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1234)

# Creating the linear regression model
model = LinearRegression(epochs=1000)

# Fitting the model
model.fit(x_train, y_train)

# Predicting the labels
y_pred = model.predict(x_test)

# Computing the Mean Squared Error (MSE) of the predictions
mse = np.mean((y_pred - y_test) ** 2)

# Printing the MSE
print("Mean Squared Error:", mse)

# Plotting the predictions
plt.scatter(x_test, y_test, s=5)
plt.plot(x_test, y_pred, color="red")
plt.show()