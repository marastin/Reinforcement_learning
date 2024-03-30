import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

np.random.seed(5)

# Problem Parameters
n_blocks = 1000

# Define the start point, and terminal states
terminal_state_right = n_blocks - 1
terminal_state_left = 0
terminals = [terminal_state_left, terminal_state_right]
start_point = (n_blocks + 1) // 2

# Define reward for each transition
rewards = defaultdict(float) # default value 0.0
rewards[terminal_state_left] = -1 
rewards[terminal_state_right] = 1

# Define Actions
actions = [act for act in range(-100, 101) if act != 0]
n_actions = len(actions)

# Initialize W
n_groups = n_blocks//100
W = np.zeros((n_groups))

# Define the value of step size
alpha = 2e-5

# Define the number of episodes
num_episodes = 100000

# Initializtion
memory = 10**6
St = np.empty(memory, dtype=int)

# Main Loop
for episode in range(num_episodes):
    state = start_point
    t = 0
    G = 0
    while True:
        St[t] = state
        action = np.random.choice(actions)
        new_state = max(0, min(state + action, n_blocks - 1))
        G += rewards[new_state]
        state = new_state
        t += 1
        if state in terminals:
            T = t
            break

    for t in range(T, -1, -1):
        W[St[t]//100] += alpha * (G - W[St[t]//100])
    
    if (episode + 1) % 100 == 0:
        print(episode + 1)
print(W)

x = np.arange(n_blocks)
idx = x // 100
y = W[idx]
plt.plot(x,y)
plt.show()
