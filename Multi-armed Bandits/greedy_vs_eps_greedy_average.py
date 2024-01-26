# k-armed bandit Problem

import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
np.random.seed(31) # Set a seed for reproducibility
k = 10 # number of arms
num_steps = 1000
reps = 400

# Reality Model
reward_mean = np.random.normal(0, 1, k)
reward_variance = np.random.rand(k)
print(reward_mean)
print(reward_variance)

# Greedy

# Initialization
Q = np.zeros([k, reps]) # the estimated reward for each action
N = np.zeros([k, reps]) # stores the number of times each action has been taken
reward_greedy = np.zeros([num_steps,reps])

for rep in range(reps):
    if rep%20==0:
        print(f"Greedy {int(rep/reps*100)} %")
    for n in range(num_steps):
        action = np.argmax(Q[:, rep])
        reward_observed = np.random.normal(reward_mean[action], reward_variance[action])
        reward_greedy[n, rep] = reward_observed
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
        reward_observed = np.random.normal(reward_mean[action], reward_variance[action])
        reward_epsilon_greedy[n, rep] = reward_observed
        N[action, rep] += 1
        Q[action, rep] = Q[action, rep] + 1/N[action, rep] * (reward_observed - Q[action, rep])



# Epsilon-Greedy
epsilon = 0.01

# Initialization
Q = np.zeros([k, reps]) # the estimated reward for each action
N = np.zeros([k, reps]) # stores the number of times each action has been taken
reward_epsilon_greedy2 = np.zeros([num_steps,reps])

for rep in range(reps):
    if rep%20==0:
        print(f"Epsilon-Greedy 0.01 {int(rep/reps*100)} %")
    for n in range(num_steps):
        eps = np.random.rand()
        if eps < epsilon:
            action = np.random.randint(0,10)
        else:
            action = np.argmax(Q[:, rep])
        reward_observed = np.random.normal(reward_mean[action], reward_variance[action])
        reward_epsilon_greedy2[n, rep] = reward_observed
        N[action, rep] += 1
        Q[action, rep] = Q[action, rep] + 1/N[action, rep] * (reward_observed - Q[action, rep])



# Plots
plt.plot(np.average(reward_greedy,axis=1), label='Greedy')
plt.plot(np.average(reward_epsilon_greedy,axis=1), label='Epsilon Greedy - 0.1' )
plt.plot(np.average(reward_epsilon_greedy2,axis=1), label='Epsilon Greedy - 0.01' )
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.title(r'Greedy vs $\epsilon$-Greedy Strategy in k-Armed Bandit')
plt.show()

