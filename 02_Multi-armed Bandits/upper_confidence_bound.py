# k-Armed Bandit Problem - UCB vs Epsilon-Greedy

# This code compares the performance of two strategies, Upper Confidence Bound (UCB)
# and epsilon-greedy, in a k-armed bandit problem. The k-armed bandit is a classical
# reinforcement learning problem where an agent needs to choose between k different
# actions, each associated with an unknown reward. The goal is to learn the optimal
# action to maximize cumulative reward.

import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
np.random.seed(310) # Set a seed for reproducibility
k = 10 # number of arms
num_steps = 2000
reps = 1 # number of repeats to get an average

# Reality Model
reward_mean = np.random.normal(0, 1, k) # mean = 0, var = 1, size = k
reward_variance = np.random.rand(k)

reward_mean_change = np.zeros([1, k])
reward_variance_change = np.ones([1, k])*0.1

reward = np.zeros([k, num_steps])

reward[:,0] = np.random.normal(reward_mean[:], reward_variance[:])
for n in range(1, num_steps):
    reward[:,n] = reward[:, n - 1] + np.random.normal(reward_mean_change[:], reward_variance_change[:])

# UCB

# Initialization
Q = np.zeros([k, reps]) # the estimated reward for each action
N = np.zeros([k, reps]) # stores the number of times each action has been taken
reward_UCB = np.zeros([num_steps,reps])
c = 5 # confidence level
e = 0.000001 # very small number to prevent division by zero

for rep in range(reps):
    if rep%20==0:
        print(f"UCB {int(rep/reps*100)} %")
    for n in range(num_steps):
        ucb = c*(np.log(n+1)/(N[:, rep]+e)**0.5)
        action = np.argmax(Q[:, rep] + ucb)
        # print(action)
        reward_observed = reward[action, n]
        reward_UCB[n, rep] = reward_observed
        N[action, rep] += 1
        Q[action, rep] = Q[action, rep] + 1/N[action, rep] * (reward_observed - Q[action, rep])



# Epsilon-Greedy
epsilon = 0.1

# Initialization
Q = np.zeros([k, reps]) # the estimated reward for each action
N = np.zeros([k, reps]) # stores the number of times each action has been taken
reward_epsilon_greedy = np.zeros([num_steps,reps])

for rep in range(reps):
    if rep%20==0:
        print(f"Epsilon-Greedy 0.1 {int(rep/reps*100)} %")
    for n in range(num_steps):
        eps = np.random.rand()
        if eps < epsilon:
            action = np.random.randint(0,10)
        else:
            action = np.argmax(Q[:, rep])
        reward_observed = reward[action, n]
        reward_epsilon_greedy[n, rep] = reward_observed
        N[action, rep] += 1
        Q[action, rep] = Q[action, rep] + 1/N[action, rep] * (reward_observed - Q[action, rep])





# Plots
plt.plot(np.average(reward_UCB,axis=1),'x', label='UCB')
plt.plot(np.average(reward_epsilon_greedy,axis=1),'o', label=r'$\epsilon$-Greedy $\epsilon$ = 0.1')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.title(r'UCB vs $\epsilon$-Greedy Strategy in k-Armed Bandit')
for arm in range(k):
    plt.plot(reward[arm,:], alpha=0.5, linewidth=1)
plt.show()

