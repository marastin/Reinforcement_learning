"""
This Python script implements the constant TD method to estimate the state values for a random walk problem 
with equal transition probabilities for moving left and right.

Problem Description:
There are 7 states: L, A, B, C, D, E, R
L and R are terminal states.
The reward of state R is 1, and the reward of state L is 0.
All other transitions have a reward of 0.
At each step, there is an equal probability to go to the right or left.

Method:
TD method is used to estimate the state values. 
The TD method updates the state values using a fixed learning rate (alpha) for each step.
"""


import numpy as np

# Define the states
states = ['L', 'A', 'B', 'C', 'D', 'E', 'R']

# Define Terminal states
terminal_states = ['L', 'R']

# Define rewards
rewards = {'R': 1, 'L': 0}

# Define Actions
RIGHT = 1
LEFT = -1
actions = [LEFT, RIGHT]

# Define the transition
transition_probs = {
    'L': {},
    'A': {RIGHT: 'L', LEFT: 'B'},
    'B': {RIGHT: 'A', LEFT: 'C'},
    'C': {RIGHT: 'B', LEFT: 'D'},
    'D': {RIGHT: 'C', LEFT: 'E'},
    'E': {RIGHT: 'D', LEFT: 'R'},
    'R': {}
}

# Define the number of episodes
num_episodes = 1000

# Initialize state values
V = {state: 0 for state in states}

# Define the discount factor
gamma = 1.0

# Define the value of step size
alpha = 0.01

# Perform Temporal Difference updates
for _ in range(num_episodes):

    # Start from a nonterminal random state
    state = np.random.choice([state for state in states if state not in terminal_states])

    # Generate an episode, calculate returns and update values
    while True:

        if state in terminal_states:
            break
        
        action = np.random.choice(actions)
        
        next_state = transition_probs[state][action]
        reward = rewards.get(next_state, 0)

        V[state] += alpha * (reward + gamma * V[next_state] - V[state])

        state = next_state


# Calculate the exact value for each state from Mathematics
exact_values = {state: i / 6 for i, state in enumerate(states)}

# Print the state values along with errors
print("State Values and Errors:")
for state, value in V.items():
    exact_value = exact_values[state]
    error = value - exact_value
    print(f"{state}: {value:.3f} | error = {error:6.3f}")