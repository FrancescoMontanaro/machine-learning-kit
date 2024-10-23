import numpy as np

class Perceptron:

    """ Magic methods """
    
    def __init__(self, lr: float = 1e-03, epochs: int = 100) -> None:
        """
        Class constructor
        :param lr: Learning rate
        :param epochs: Number of epochs
        """

        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    

    """ Public methods """

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data
        :param x: Features of the dataset
        :param y: Labels of the dataset
        """

        # Extracting the number of samples and features
        n_samples, n_features = x.shape

        # Initializing the weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Iterating over the number of epochs
        for _ in range(self.epochs):
            # Computing the linear combination of the weights and features
            linear_model = np.dot(x, self.weights) + self.bias

            # Computing the predicted labels
            y_pred = self._activation(linear_model)

            # Updating the weights and bias
            self.weights += self.lr * np.dot(x.T, y - y_pred)
            self.bias += self.lr * np.sum(y - y_pred)


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the dataset
        :param x: Features of the dataset
        :return: Predicted labels
        """

        # Checking if the model has been trained
        if self.weights is None or self.bias is None:
            raise Exception("Train the model first!")

        # Computing the linear combination of the weights and features
        linear_model = np.dot(x, self.weights) + self.bias

        # Computing the predicted labels
        y_pred = self._activation(linear_model)

        return y_pred


    """ Protected methods """

    def _activation(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the activation function
        :param x: Input of the activation function
        :return: Output of the activation function
        """

        # Computing the activation function (Step function)
        return np.where(x >= 0, 1, 0)