# Non-stationary k-arm-bandits problem with epsilon greedy strategy

import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
np.random.seed(31) # Set a seed for reproducibility
k = 10 # number of arms
num_steps = 10000
reps = 400

# Reality Model
reward_mean = np.random.normal(0, 1, k) # mean = 0, var = 1, size = k
reward_variance = np.random.rand(k)

reward_mean_change = np.zeros([1, k])
reward_variance_change = np.ones([1, k])*0.01

reward = np.zeros([k, num_steps])

reward[:,0] = np.random.normal(reward_mean[:], reward_variance[:])
for n in range(1, num_steps):
    reward[:,n] = reward[:, n - 1] + np.random.normal(reward_mean_change[:], reward_variance_change[:])



# Epsilon-Greedy
epsilon = 0.1
alpha = 0.1

# Initialization
Q = np.zeros([k, reps]) # the estimated reward for each action
N = np.zeros([k, reps]) # stores the number of times each action has been taken
reward_epsilon_greedy = np.zeros([num_steps,reps])

for rep in range(reps):
    if (rep+1)%20==0:
        print(f"Epsilon-Greedy 0.1 {int((rep+1)/reps*100)} %")
    for n in range(num_steps):
        eps = np.random.rand()
        if eps < epsilon:
            action = np.random.randint(0,10)
        else:
            action = np.argmax(Q[:, rep])
        reward_observed = reward[action, n]
        reward_epsilon_greedy[n, rep] = reward_observed
        N[action, rep] += 1
        Q[action, rep] = Q[action, rep] + alpha * (reward_observed - Q[action, rep])



# Plots
# plt.plot(np.average(reward_greedy,axis=1), label='Greedy')
plt.plot(np.average(reward_epsilon_greedy,axis=1), label='Epsilon Greedy - 0.1' )
# plt.plot(np.average(reward_epsilon_greedy2,axis=1), label='Epsilon Greedy - 0.01' )
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.title(r"$\epsilon$-Greedy Strategy in k-Armed Bandit for $\epsilon$ = 0.1 and $\alpha$ = 0.1")

for arm in range(k):
    plt.plot(reward[arm,:], alpha=0.5, linewidth=1)

plt.show()

