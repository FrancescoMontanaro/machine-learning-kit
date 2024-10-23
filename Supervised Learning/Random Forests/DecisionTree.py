import numpy as np
from Node import Node
from collections import Counter

class DecisionTree:

    """ Magic methods """

    def __init__(self, min_samples_split : int = 2, max_depth: int = 100, n_features: int | None = None) -> None:
        """
        Class constructor
        :param min_samples_split: Minimum number of samples required to split an internal node
        :param max_depth: Maximum depth of the tree
        :param n_features: Number of features to consider when looking for the best split
        """

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None


    """ Public methods """

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model
        :param x: Features of the dataset
        :param y: Labels of the dataset
        """
        
        # Extracting the number of features of the dataset
        self.n_features = x.shape[1] if not self.n_features else min(self.n_features, x.shape[1])

        # Growing the tree
        self.root = self._growTree(x, y)


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the labels of the dataset
        :param x: Features of the dataset
        :return: Predicted labels
        """

        # Check if the root is a valid node
        if self.root is None:
            raise Exception("Fit the model first!")

        # Predicting the labels
        y_pred = [self._traverseTree(sample, self.root) for sample in x]

        return np.array(y_pred)


    """ Protected methods """

    def _growTree(self, x: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Grow the tree
        :param x: Features of the dataset
        :param y: Labels of the dataset
        :param depth: Depth of the tree
        :return: Root node of the tree
        """
        
        # Extracting the number of samples and the number of features
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        ### Checking the stopping criteria ###

        # If the stopping criteria is met, return a leaf node
        if n_labels == 1 or depth >= self.max_depth or n_samples < self.min_samples_split:
            # Compute the leaf value
            leaf_value = self._mostCommonLabel(y)

            # Return a leaf node
            return Node(value=leaf_value)

        ### Finding the best split ###

        # Selecting the features randomly
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False) 

        # Finding the best split
        best_feature, best_threshold = self._bestSplit(x, y, feature_idxs) # type: ignore

        ### Creating the sub-trees ###

        # Splitting the dataset using the best split
        left_idxs, right_idxs = self._split(x[:, best_feature], best_threshold)

        # Creating the left and right sub-trees
        left = self._growTree(x[left_idxs, :], y[left_idxs], depth + 1)
        right = self._growTree(x[right_idxs, :], y[right_idxs], depth + 1)

        # Return the root node
        return Node(best_feature, best_threshold, left, right)


    def _mostCommonLabel(self, y: np.ndarray) -> int:
        """
        Find the most common label in the dataset
        :param y: Labels of the dataset
        :return: Most common label in the dataset
        """

        # Counting the number of samples for each label
        counter = Counter(y)

        # Finding the most common label
        most_common = counter.most_common(1)[0][0]

        return most_common
    

    def _bestSplit(self, x: np.ndarray, y: np.ndarray, feature_idxs: np.ndarray) -> tuple:
        """
        Find the best split for the dataset
        :param x: Features of the dataset
        :param y: Labels of the dataset
        :param feature_idxs: Features to consider when looking for the best split
        :return: Best split for the dataset
        """
        
        # Initializing the best gain, split index and split threshold
        best_gain = -1
        split_idx, split_threshold = None, None

        # Trying all the possible splits
        for feature_idx in feature_idxs:
            # Selecting the feature column
            feature_col = x[:, feature_idx]

            # Finding all the possible thresholds
            thresholds = np.unique(feature_col)

            # Trying all the possible thresholds
            for threshold in thresholds:
                # Computing the information gain
                gain = self._informationGain(y, feature_col, threshold)

                # Checking if the current information gain is the best one
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold


    def _informationGain(self, y: np.ndarray, feature_col: np.ndarray, threshold: float) -> float:
        """
        Compute the information gain
        :param y: Labels of the dataset
        :param feature_col: Feature column of the dataset
        :param threshold: Threshold of the split
        :return: Information gain
        """

        ### Computing the parent entropy ###

        parent_entropy = self._entropy(y)

        ### Creating the children ###

        # Splitting the dataset
        left_idxs, right_idxs = self._split(feature_col, threshold)

        # Checking if the split is valid
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        ### Computing the weighted average entropy of the children ###

        # Computing the number of samples and the number of samples in each child
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)

        # Computing the entropy of each child
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

        # Computing the weighted average entropy of the children
        child_entropy = float(np.average([e_left, e_right], weights = [n_left / n, n_right / n]))

        ### Computing the information gain ###

        information_gain = parent_entropy - child_entropy

        return information_gain


    def _entropy(self, y: np.ndarray) -> float:
        """
        Compute the entropy of the dataset
        :param y: Labels of the dataset
        :return: Entropy of the dataset
        """

        # Computing the number of occurrences of each label
        hist = np.bincount(y)

        # Computing the probability of each label
        ps = hist / len(y)

        # Computing the entropy
        return - np.sum([p * np.log2(p) for p in ps if p > 0])
    

    def _split(self, x_column: np.ndarray, split_threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the dataset
        :param x_column: Feature column of the dataset
        :param split_threshold: Threshold of the split
        :return: Dataset splitted
        """

        # Creating the left and right datasets
        left_idxs = np.argwhere(x_column < split_threshold).flatten()
        right_idxs = np.argwhere(x_column >= split_threshold).flatten()

        return left_idxs, right_idxs
    

    def _traverseTree(self, x: np.ndarray, node: Node | None) -> float:
        """
        Traverse the tree
        :param x: Features of the dataset
        :param node: Current node
        :return: Predicted label
        """

        # Checking if the node is valid
        if node is None:
            raise Exception("Error: the node is not valid")

        # Checking if the node is a leaf node
        if node.isLeafNode():
            # Consistency check
            if node.value is None:
                raise Exception("Error: the node is a leaf Node but it does not have any value")
            
            # Returning the node value
            return node.value

        # Checking if the sample goes to the left or right sub-tree
        if x[node.feature] <= node.threshold:
            return self._traverseTree(x, node.left)
        
        return self._traverseTree(x, node.right)