import numpy as np

class Learner:

    """ Magic methods """

    def __init__(self, n_arms: int) -> None:
        """
        Class constructor
        :param n_arms: number of arms
        """
        
        self.t = 0
        self.n_arms = n_arms
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])


    """ Public methods """

    def updateObservations(self, pulled_arm: int, reward: float) -> None:
        """ 
        Updates the observations for the pulled arm
        :param pulled_arm: the pulled arm
        :param reward: the reward obtained
        """
        
        # Updating the rewards for the pulled arm
        self.rewards_per_arm[pulled_arm].append(reward)

        # Update the collected rewards
        self.collected_rewards = np.append(self.collected_rewards, reward)