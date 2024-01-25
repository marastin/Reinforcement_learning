# k-armed bandit Problem

import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
np.random.seed(3180) # Set a seed for reproducibility
k = 10 # number of arms
num_steps = 1000

# Reality Model
reward_mean = np.random.normal(0, 2, k)
reward_variance = 0.5 + np.random.rand(10)/30
print(reward_mean)
print(reward_variance)
# Initialization
Q = np.zeros(k) # the estimated reward for each action
N = np.zeros(k) # stores the number of times each action has been taken



# Greedy
reward_greedy = np.zeros(num_steps)
for n in range(num_steps):
    action = np.argmax(Q)
    reward_observed = np.random.normal(reward_mean[action], reward_variance[action])
    reward_greedy[n] = reward_observed
    N[action] += 1
    Q[action] = Q[action] + 1/N[action] * (reward_observed - Q[action])

print("Greedy")
print(N)
print(np.average(reward_greedy))
print("")



# Epsilon-Greedy: epsilon = 0.1
epsilon = 0.1

# Initialization
Q = np.zeros(k) # the estimated reward for each action
N = np.zeros(k) # stores the number of times each action has been taken

reward_epsilon_greedy = np.zeros(num_steps)
for n in range(num_steps):
    eps = np.random.rand()
    if eps < epsilon:
        action = np.random.randint(0,10)
    else:
        action = np.argmax(Q)
    reward_observed = np.random.normal(reward_mean[action], reward_variance[action])
    reward_epsilon_greedy[n] = reward_observed
    N[action] += 1
    Q[action] = Q[action] + 1/N[action] * (reward_observed - Q[action])

print("Epsilon_greedy")
print(N)
print(np.average(reward_epsilon_greedy))
print("")


# Plots
plt.plot(reward_greedy, label='Greedy')
plt.plot(reward_epsilon_greedy, label='Epsilon Greedy' )
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.title('Greedy vs Epsilon Greedy Strategy in k-Armed Bandit')
plt.show()

