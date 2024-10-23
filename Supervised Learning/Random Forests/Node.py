from typing import Optional

class Node:

    """ Magic methods """

    def __init__(self, feature: int | None = None, threshold: float | None = None, left: Optional['Node'] = None, right: Optional['Node'] = None, *, value = None) -> None:
        """
        Class constructor
        :param feature: Feature of the node
        :param threshold: Threshold of the node
        :param left: Left child of the node
        :param right: Right child of the node
        :param value: Value of the node
        """

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    
    """ Public methods """
    
    def isLeafNode(self) -> bool:
        """
        Check if the node is a leaf node
        :return: True if the node is a leaf node, False otherwise
        """
        
        return self.value is not None