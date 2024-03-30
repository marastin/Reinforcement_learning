# Gradient Bandit Algorithm with Changing Action Values

# This code implements a gradient bandit algorithm to solve a k-armed bandit problem
# where the action values change over time. The algorithm uses a softmax policy to
# select actions based on estimated action values. The action values are updated using
# a gradient update rule, and the rewards are generated from a changing distribution.


import numpy as np
import matplotlib.pyplot as plt

# Math:
def softmax(logits):
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits)
    return probabilities

def choose_action(probabilities):
    action = np.random.choice(probabilities.size, p=probabilities)
    return action

# Problem parameters
np.random.seed(31) # Set a seed for reproducibility
k = 10 # number of arms
num_steps = 1000
reps = 1
action_value_is_changing = True

# Reality Model
reward_mean = np.random.normal(4, 1, k) # mean = 4, var = 1, size = k
reward_variance = np.random.rand(k)

reward_mean_change = np.zeros([1, k])
reward_variance_change = np.ones([1, k])*0.05

reward = np.zeros([k, num_steps])

reward[:,0] = np.random.normal(reward_mean[:], reward_variance[:])
for n in range(1, num_steps):
    reward[:,n] = reward[:, n - 1] + np.random.normal(reward_mean_change[:], reward_variance_change[:]) * action_value_is_changing



# Gradient Bandit Algorithms
alpha = 0.1

# Initialization
# Q = np.zeros([k, reps]) # the estimated reward for each action
H = np.zeros([k, num_steps+1]) 
N = np.zeros([k, reps]) # stores the number of times each action has been taken
reward_gradient = np.zeros([num_steps,reps])

for rep in range(reps):
    if (rep+1)%20==0:
        print(f"Progress: {int((rep+1)/reps*100)} %")
    
    PI = softmax(np.zeros([k]))
    reward_baseline = 0
    for n in range(num_steps):
        action = choose_action(PI)
        reward_observed = reward[action, n]
        reward_gradient[n, rep] = reward_observed

        H[:, n+1] = H[:, n] - alpha * (reward_observed - reward_baseline)*PI #update for not selected action
        H[action, n+1] = H[action, n] + alpha * (reward_observed - reward_baseline)*(1 - PI[action]) #update for selected action
        
        reward_baseline = reward_baseline + 1/(n+1)*(reward_observed - reward_baseline)
        
        PI = softmax(H[:, n+1]) # Calc probabilities for the next step
        N[action, rep] += 1



# Plots

plt.plot(np.average(reward_gradient,axis=1), 'o', label=r'gradient bandit - $\alpha$ = 0.1' )
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.legend()
plt.title(r"Gradient Bandit Algorithms $\alpha$ = 0.1")

for arm in range(k):
    plt.plot(reward[arm,:], alpha=0.5, linewidth=1)

plt.show()

