import Utils
import numpy as np
from UCB1 import UCB1
import matplotlib.pyplot as plt
from Environment import Environment
from ThompsonSampling import ThompsonSampling

# Global variables and experiment parameters
T = 1000
n_arms = 4
n_experiments = 1000
ts_rewards_per_experiment = []
ucb1_rewards_per_experiment = []

# Print status
print("Starting Simulation...")
Utils.printProgressBar(0, n_experiments)

# Initializing the environment
environment = Environment(n_arms=n_arms)

# Sampling the optimal arm
opt = environment.probabilities.max()

# Iterating over the experiments
for e in range(n_experiments):
    # Initializing the learners
    ts_learner = ThompsonSampling(n_arms=n_arms)
    ucb1 = UCB1(n_arms=n_arms)

    # Iterating over the time
    for t in range(T):
        ### Thompson Sampling ###
        pulled_arm = ts_learner.pullArm()
        reward = environment.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        ### UCB1 ###
        pulled_arm = ucb1.pullArm()
        reward = environment.round(pulled_arm)
        ucb1.update(pulled_arm, reward)
    
    # Updating the rewards per experiment for each learner
    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb1_rewards_per_experiment.append(ucb1.collected_rewards)

    # Print status
    Utils.printProgressBar(e+1, n_experiments)


# Plotting the results
plt.figure(0)
plt.xlabel("time")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiment, axis=0)), 'b')
plt.legend(['TS', 'UCB1'])
plt.show()