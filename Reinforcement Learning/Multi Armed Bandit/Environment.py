import numpy as np

class Environment:
    
    """ Magic methods """

    def __init__(self, n_arms: int, probabilities: np.ndarray | None = None) -> None:
        """
        Class constructor
        :param n_arms: number of arms
        :param probabilities: the probabilities of the arms
        """
        
        self.n_arms = n_arms

        # Generating the random probabilities of the arms
        self.probabilities = np.random.rand(n_arms) if probabilities is None else probabilities

    
    """ Public methods """

    def round(self, pulled_arm: int) -> float:
        """
        Returns the reward from a probability distribution
        :param pulled_arm: the pulled arm
        :return: the reward
        """

        # Sampling the reward from a Bernoulli distribution
        return np.random.binomial(1, self.probabilities[pulled_arm])