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

# Initialize function approximation
order = 10
feature_funcs = []
for i in range(0, order + 1):
    feature_funcs.append(lambda s, i=i: pow(s, i))

W = np.zeros(order + 1)

# Define the value of step size
alpha = 1e-4

# Define the number of episodes
num_episodes = 5000

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

    for t in range(T-1, -1, -1):
        s = St[t] / float(n_blocks)
        features = np.asarray([func(s) for func in feature_funcs])
        derivative = features
        Vhat = np.dot(W, features)
        delta = alpha * (G - Vhat)
        W += delta * derivative
    
    if (episode + 1) % 100 == 0:
        print(episode + 1, T)
print(W)

x = np.arange(n_blocks)
s = x / float(n_blocks)
features = np.asarray([func(s) for func in feature_funcs])
Vhat = np.dot(W, features)
plt.plot(x, Vhat)
plt.show()
