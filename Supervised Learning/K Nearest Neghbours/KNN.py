import numpy as np
from collections import Counter

class KNN:

    """ Magic methods """

    def __init__(self, k: int = 3) -> None:
        """
        Class constructor
        :param k: Number of neighbours to consider
        """
        
        self.k = k


    """ Public methods """

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model
        :param x: Features of the dataset
        :param y: Labels of the dataset
        """
        
        # Setting the training data
        self.x_train = x
        self.y_train = y

    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the dataset
        :param x: Features of the dataset
        :return: Predicted labels
        """

        # Predicting the labels for each sample
        y_pred = [self._predictSample(sample) for sample in x]

        return np.array(y_pred)


    """ Protected methods """
    
    def _predictSample(self, sample: np.ndarray) -> int:
        """
        Predict the label of a sample
        :param sample: Sample to predict
        :return: Predicted label
        """

        ### Compute the distances ###

        # Compute the Euclidean distances between the sample and the training data
        distance = [self._euclideanDistance(sample, x) for x in self.x_train]

        ### Get the k nearest neighbours ###

        # Get the indices of the k nearest neighbours
        k_indices = np.argsort(distance)[:self.k]

        ### Determine the majority vote ###

        # Get the labels of the k nearest neighbours
        k_neighbours = [self.y_train[i] for i in k_indices]

        # Count the number of labels
        counter = Counter(k_neighbours)

        # Get the most common label
        return int(counter.most_common(1)[0][0])


    @staticmethod
    def _euclideanDistance(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute the euclidean distance between points
        :param x1: First sample
        :param x2: Second sample
        :return: Euclidean distance
        """

        # Compute the euclidean distance
        return np.sqrt(np.sum((x1 - x2) ** 2))