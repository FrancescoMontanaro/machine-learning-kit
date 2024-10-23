import numpy as np
from collections import Counter
from decision_tree import DecisionTree

class RandomForest:

    """ Magic methods """

    def __init__(self, n_trees: int = 20, max_depth: int = 100, min_samples_split: int = 2, n_features: int | None = None) -> None:
        """
        Class constructor
        :param n_trees: Number of trees
        :param max_depth: Maximum depth of the tree
        :param min_samples_split: Minimum number of samples required to split an internal node
        :param n_features: Number of features to consider when looking for the best split
        """

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []


    """ Public methods """

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model
        :param x: Features of the dataset
        :param y: Labels of the dataset
        """

        # Initializing the trees list
        self.trees = []

        # Iterating over the number of trees
        for _ in range(self.n_trees):
            # Print status
            print(f"Training tree {_ + 1}/{self.n_trees}", end="\r")

            # Creating the decision tree
            tree = DecisionTree(
                max_depth = self.max_depth, 
                min_samples_split = self.min_samples_split, 
                n_features = self.n_features
            )

            # Creating the bootstrap samples
            x_sample, y_sample = self._bootstrapSamples(x, y)

            # Training the decision tree
            tree.fit(x_sample, y_sample)

            # Adding the tree to the trees list
            self.trees.append(tree)


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the dataset
        :param x: Features of the dataset
        :return: Predicted labels
        """

        # Check if the trees list is empty
        if len(self.trees) == 0:
            raise Exception("Fit the model first!")

        # Predicting the labels
        y_pred = np.array([tree.predict(x) for tree in self.trees])

        # Computing the majority vote
        y_pred = np.swapaxes(y_pred, 0, 1)
        y_pred = [self._mostCommonLabel(y) for y in y_pred]

        return np.array(y_pred)


    """ Protected methods """

    def _bootstrapSamples(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create bootstrap samples of the dataset
        :param x: Features of the dataset
        :param y: Labels of the dataset
        :return: Bootstrap samples
        """
        
        # Extracting the number of samples of the dataset
        n_samples = x.shape[0]

        # Creating the bootstrap indices
        indices = np.random.choice(n_samples, n_samples, replace=True)

        return x[indices], y[indices]
    

    def _mostCommonLabel(self, y: np.ndarray) -> int:
        """
        Find the most common label
        :param y: Labels of the dataset
        :return: Most common label
        """
        
        # Counting the number of occurrences of each label
        collection = Counter(y)

        # Computing the most common label
        most_common = collection.most_common(1)[0][0]

        return most_common