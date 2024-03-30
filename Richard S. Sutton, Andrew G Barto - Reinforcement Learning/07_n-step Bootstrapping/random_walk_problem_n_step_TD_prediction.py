import numpy as np
from tabulate import tabulate
from collections import defaultdict

np.random.seed(5)

# Problem Parameters
n_blocks = 19 # an odd number

# Define the states
states = np.zeros((n_blocks))

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
RIGHT = 1
LEFT = -1
actions = [LEFT, RIGHT]
n_actions = len(actions)

# Define the number of episodes
num_episodes = 10
time_elapsed_in_episode = np.zeros((num_episodes), dtype=int)

# Initialize state values
V = np.zeros((n_blocks))

# Define the discount factor
gamma = 1.0

# Define the value of step size
alpha = 0.5

# Define the number of step in bootstrapping
n = 4

# Initializtion
# All store and access operations (for St and Rt) can take their index mod n + 1
St = np.empty((n+1), dtype=int)
Rt = np.empty((n+1), dtype=float)


# Main Loop
for episode in range(num_episodes):
    T = np.inf
    t = 0
    
    state = start_point

    while True:
        
        St[t % (n+1)] = state
        
        if t < T:
            # Take an action according to policy (here equal probability to go right or left)
            action = np.random.choice(actions)
            
            new_state = state + action
            new_state = max(0, min(new_state, n_blocks - 1))
            
            if new_state in terminals:
                T = t + 1
            
            reward = rewards[new_state]
            Rt[(t+1) % (n+1)] = reward

        tau = t - n + 1 # τ is the time whose state’s estimate is being updated

        if tau >= 0:
            G = sum([pow(gamma, i - tau - 1) * Rt[i % (n+1)] for i in range(tau + 1, min(tau + n, T) + 1)]) 
            
            if tau + n < T:
                G += pow(gamma, n) * V[St[(tau + n) % (n+1)]]

            V[St[tau % (n+1)]] += alpha * (G - V[St[tau % (n+1)]])
        
        t += 1
        
        if tau >= T - 1:
            break

        state = new_state
    time_elapsed_in_episode[episode] = t

formatted_V = [[f"block {b}", f"{value:.3f}"] for b, value in enumerate(V)]
print(tabulate(formatted_V, tablefmt='fancy_grid'))
