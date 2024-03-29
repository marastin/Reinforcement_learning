"""
This code implements the Double-Q-Learning algorithm to solve a cliff walking problem.
Double-Q-learning is an off-policy temporal-difference reinforcement learning algorithm that uses two state-action values.

The grid world consists of a specified number of rows and columns, with a start point, cliffs and a terminal state.

The agent can move in four directions (N, S, E, W).

Upon going to the cliff zone the agent will move to the start point with large negative reward.

The goal is to learn the optimal policy that maximizes cumulative rewards while navigating from the start point to the terminal state.

"""

import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

np.random.seed(5)

# Problem Parameters
n_rows = 4
n_cols = 12

# Define the states
states = np.zeros((n_rows, n_cols))

# Define the start point, cliffs, and terminal state
terminal_state = np.array((3, 11))
start_point = np.array((3, 0))
cliffs = [np.array((n_rows-1, col)) for col in range(1, n_cols-1)]

# Define reward for each transition
rewards = -1
rewards_cliff = -100

# Define Actions
actions = {
    0 : np.array((-1, 0)), # Move Up (N)
    1 : np.array((1, 0)), # Move Down (S)
    2 : np.array((0, 1)), # Move Right (E)
    3 : np.array((0, -1)) # Move Left (W)
    }
n_actions = len(actions)

# Define the transition
transitions = np.empty((n_rows, n_cols), dtype=object)
for row in range(n_rows):
    for col in range(n_cols):
        transition = []
        if row > 0:
            transition.append(0) # Move Up (N)
        if row < n_rows - 1:
            transition.append(1) # Move Down (S)
        if col > 0:
            transition.append(3) # Move Left (W)
        if col < n_cols - 1:
            transition.append(2) # Move Right (E)
        transitions[row, col] = transition


# Define the number of episodes
num_episodes = 500
time_elapsed_in_episode = np.zeros((num_episodes, 1), dtype=int)

# Record reward in each episode
rewards_in_episodes = np.zeros((num_episodes, 1), dtype=int)

# Initialize state-action values Q1 and Q2
Q1 = np.zeros((n_rows, n_cols, n_actions))
Q1[*terminal_state][:] = 0
Q2 = np.zeros((n_rows, n_cols, n_actions))
Q2[*terminal_state][:] = 0


# Define the discount factor
gamma = 1.0

# Define the value of step size
alpha = 0.5

# Define the epsilon for epsilon-greedy policy
epsilon = 0.1

# Perform Q-learning Upadate
for episode in range(num_episodes):

    t = 0
    
    # Start from the start point
    state = start_point

    # Generate an episode, calculate returns and update values
    while True:
        
        # Check being at terminal state
        if all(state == terminal_state):
            break
        
        Q = Q1[*state] + Q2[*state]

        # Choose an action with epsilon-reedy policy
        if np.random.rand() > epsilon:
            # Select the greedy action
            selected_action = transitions[*state][np.argmax(Q[transitions[*state]])]

        else:
            # Select a random action from valid transition
            selected_action = np.random.choice(list(transitions[*state]))

        # Calulate next_state
        next_state = state + actions[selected_action]

        # Addjust the state so that the agent remains inside the gridworld
        next_state = np.array((min(n_rows - 1, max(0, next_state[0])), min(n_cols -1, max(0, next_state[1]))), dtype=int)
        
        # Check if the state is in the cliffs so that the agent moves to the start point
        if any(np.array_equal(next_state, cliff) for cliff in cliffs):
            next_state = start_point
            reward = rewards_cliff # Considering constant reward -100
        else:
            reward = rewards # Considering constant reward -1

        if np.random.rand() >= 0.5:
            # Calculating next_action_max_Q1 in the next_state by maximising the Q1
            next_action_max_Q1 = transitions[*next_state][np.argmax(Q1[*next_state][transitions[*next_state]])]

            # Update action-state value Q1
            Q1[*state][selected_action] += alpha * (reward + gamma * Q1[*next_state][next_action_max_Q1] - Q1[*state][selected_action])
        else:
            # Calculating next_action_max_Q2 in the next_state by maximising the Q2
            next_action_max_Q2 = transitions[*next_state][np.argmax(Q2[*next_state][transitions[*next_state]])]

            # Update action-state value Q2
            Q2[*state][selected_action] += alpha * (reward + gamma * Q2[*next_state][next_action_max_Q2] - Q2[*state][selected_action])

        # Update state and action for next loop
        state = next_state
        rewards_in_episodes[episode] += reward

        t += 1
    
    # Record the number of step
    time_elapsed_in_episode[episode] = t
    

# Draw a table to visualize the greedy action based on the argmax of Q
# Q can be the sum or average of Q1 and Q2
Q = Q1 + Q2
greedy_actions = np.empty((n_rows, n_cols), dtype=object)
for row in range(n_rows):
    for col in range(n_cols):
        state = (row, col)
        if all(state == terminal_state):
            greedy_actions[row, col] = '■'
        elif any(np.array_equal(state, cliff) for cliff in cliffs):
            greedy_actions[row, col] = '*'
        else:
            greedy_action = transitions[*state][np.argmax(Q[*state][transitions[*state]])]
            if greedy_action == 0:
                greedy_actions[row, col] = 'N'
            elif greedy_action == 1:
                greedy_actions[row, col] = 'S'
            elif greedy_action == 2:
                greedy_actions[row, col] = 'E'
            elif greedy_action == 3:
                greedy_actions[row, col] = 'W'

print(tabulate(greedy_actions, tablefmt='fancy_grid'))

# Create a greedy episode
state = start_point
selected_action = transitions[*state][np.argmax(Q[*state][transitions[*state]])]
greedy_actions_steps = np.empty((n_rows, n_cols), dtype=object)

t = 0
greedy_actions_steps[*state] = t
while True:

    # Check being at terminal state
    if all(state == terminal_state):
        break
    
    # Calulate next_state
    next_state = state + actions[selected_action]

    # Addjust the state so that the agent remains inside the gridworld
    next_state = np.array((min(n_rows - 1, max(0, next_state[0])), min(n_cols -1, max(0, next_state[1]))), dtype=int)
    
    # Check if the state is in the cliffs so that the agent moves to the start point
    if any(np.array_equal(next_state, cliff) for cliff in cliffs):
        next_state = start_point
        reward = rewards_cliff # Considering constant reward -100
    else:
        reward = rewards # Considering constant reward -1

    # Select the greedy action
    next_action = transitions[*next_state][np.argmax(Q[*next_state][transitions[*next_state]])]

    # Update state and action for next loop
    state = next_state
    selected_action = next_action

    t += 1
    greedy_actions_steps[*state] = t
    
    # To prevent infinite loop
    if t > 100:
        break

# Show the path according to greedy action
print(tabulate(greedy_actions_steps, tablefmt='fancy_grid'))

# plot the path length per episode
plt.plot(rewards_in_episodes)
plt.grid(True)
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
