
# Multi-Armed Bandit

This repository contains implementations of classic multi-armed bandit algorithms built from scratch using only the Numpy library. Each algorithm is described with an in-depth explanation of its logic, step-by-step computations, and matrix manipulations when applicable.

## Table of Contents

1. [Introduction](#introduction)
2. [Running Example: Reward Distributions](#running-example)
3. [Components & Detailed Calculations](#components-and-detailed-calculations)
    - [Environment](#environment)
    - [Learner](#learner)
    - [UCB1](#ucb1)
    - [Thompson Sampling](#thompson-sampling)

## Introduction

The multi-armed bandit problem is a classic reinforcement learning problem where an agent must choose between multiple options, or "arms," each with an unknown reward distribution. The agent aims to maximize its cumulative reward by balancing exploration (trying each arm) and exploitation (selecting arms that previously gave high rewards).

In this implementation, we detail the **UCB1** (Upper Confidence Bound) and **Thompson Sampling** algorithms. The following sections cover the setup, reward distributions, and component logic used for decision-making.

## Running Example: Reward Distributions

For demonstration, assume each arm of the bandit has a different reward distribution:
- **Arm 1**: Rewards follow a Bernoulli distribution with $p=0.2$.
- **Arm 2**: Rewards follow a Bernoulli distribution with $p=0.5$.
- **Arm 3**: Rewards follow a Bernoulli distribution with $p=0.8$.

Through repeated trials, the algorithms converge on the arm with the highest reward probability.

### Example Details

1. **Arms**: Three arms, each with a different probability of success.
2. **Reward Distribution**: Bernoulli distributions with varying success probabilities.
3. **Objective**: Maximize cumulative rewards over a fixed number of trials.

## Components and Detailed Calculations

### Environment

The `Environment` class (from `environment.py`) sets up the reward distributions for each arm and provides an interface for the agent to interact with the bandit. 

- **Reward Function**: For each arm, the environment samples from its Bernoulli distribution to return a reward of 0 or 1.

Example:
```python
# Simulates pulling an arm and returns a reward based on its probability distribution
def get_reward(self, arm: int) -> int:
    pass
```

The environment handles random reward generation, mimicking real-world uncertainties in action outcomes.

### Learner

The `Learner` class (from `learner.py`) provides the base structure for any bandit algorithm, storing essential attributes like the number of arms, the total reward accumulated, and the record of each arm's selections and rewards.

Attributes include:
- `n_arms`: Total number of arms available.
- `rewards_per_arm`: List of rewards earned from each arm.
- `collected_rewards`: Total rewards accumulated.

This base class supports initialization and basic tracking of arm performance, facilitating exploration-exploitation strategies in derived classes.

### UCB1

The **UCB1** algorithm is implemented in `ucb1.py` and balances exploration and exploitation using a confidence bound. Each time an arm is pulled, UCB1 calculates an upper confidence bound to estimate potential rewards based on the arm's observed mean reward and selection count.

#### Step 1: Calculating UCB Values

For each arm $i$, UCB1 calculates:
- **Mean Reward**: $\hat{\mu}_i = \frac{\text{total rewards from arm } i}{\text{times arm } i \text{ was selected}}$
- **Confidence Bound**: $U_i = \sqrt{\frac{2 \ln(n)}{n_i}}$

where $n$ is the total number of trials, and $n_i$ is the number of times arm $i$ was chosen. The upper confidence bound for each arm is given by:
$\text{UCB}_i = \hat{\mu}_i + U_i$
The division by $n_i$ in the confidence bound term ensures that arms with fewer trials have higher uncertainty and are more likely to be explored. This balance between exploration and exploitation is crucial for maximizing cumulative rewards.

#### Step 2: Selecting an Arm

The arm with the highest UCB value is chosen for the next trial, balancing the reward mean and uncertainty.

### Thompson Sampling

The **Thompson Sampling** algorithm (from `thompson_sampling.py`) selects arms probabilistically, updating its belief about each arm's reward probability using a beta distribution. This method naturally balances exploration and exploitation by assigning a higher probability to arms that have previously yielded better rewards.

#### Step 1: Sampling from Beta Distributions

For each arm $i$, Thompson Sampling maintains a beta distribution, initially set to $Beta(1, 1)$. After each trial, the parameters of the distribution are updated based on the reward received.

The parameters are:
- $\alpha_i$: Success count for arm $i$.
- $\beta_i$: Failure count for arm $i$.

Each arm's probability of being the best arm is estimated by sampling from its beta distribution.

#### Step 2: Choosing an Arm

The arm with the highest sampled probability is chosen for the next trial.

#### Example Update

After each pull, the following updates occur:
1. **Reward 1**: Increment $\alpha_i$ for the chosen arm.
2. **Reward 0**: Increment $\beta_i$ for the chosen arm.

By maintaining dynamic distributions, Thompson Sampling achieves a balance between known successful arms and unexplored arms with uncertain potential.