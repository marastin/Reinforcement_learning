import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

np.random.seed(5)

# Problem Parameters
n_rows = 10
n_cols = 20

# Define the states
states = np.zeros((n_rows, n_cols))

START_POINT = np.array((0, 0), dtype=int)
TERMINAL = np.array((n_rows - 2, n_cols - 1), dtype=int)

grid = np.zeros((n_rows, n_cols))
grid[0:7, 3] = 1
grid[3:, 6] = 1
grid[5, 8:] = 1
grid[5, 7] = 0
grid[*TERMINAL] = 3
grid[*START_POINT] = 0

# Define reward for each transition
rewards = -1

# Define Actions
actions = {
    0 : np.array((-1, 0)), # Move Up (U)
    1 : np.array((1, 0)), # Move Down (D)
    2 : np.array((0, 1)), # Move Right (R)
    3 : np.array((0, -1)) # Move Left (L)
    }
n_actions = len(actions)

def is_state_valid(state) -> bool:
    if (state[0] < 0) or (state[0] >= n_rows):
        return False
    if (state[1] < 0) or (state[1] >= n_cols):
        return False
    if grid[state] == 1: # Wall
        return False
    return True

def get_valid_actions(state) -> list:
    v_actions = []
    row, col = state
    for idx in actions:
        s = (row + actions[idx][0], col + actions[idx][1])
        if is_state_valid(s):
            v_actions.append(idx)
    return v_actions

valid_actions = np.empty((n_rows, n_cols), dtype=object)
for row in range(n_rows):
    for col in range(n_cols):
        valid_actions[row, col] = get_valid_actions((row, col))

# Define the number of episodes
num_episodes = 500
time_elapsed_in_episode = np.zeros((num_episodes, 1), dtype=int)

# Record reward in each episode
rewards_in_episodes = np.zeros((num_episodes, 1), dtype=int)

# Initialize state-action values
Q = np.zeros((n_rows, n_cols, n_actions))

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
    state = START_POINT

    # Generate an episode, calculate returns and update values
    while True:
        
        # Check being at terminal state
        if all(state == TERMINAL):
            break

        # Choose an action with epsilon-reedy policy
        if np.random.rand() > epsilon:
            # Select the greedy action
            selected_action = valid_actions[*state][np.argmax(Q[*state][valid_actions[*state]])]

        else:
            # Select a random action from valid transition
            selected_action = np.random.choice(list(valid_actions[*state]))

        # Calulate next_state
        next_state = state + actions[selected_action]

        # Addjust the state so that the agent remains inside the gridworld
        # next_state = np.array((min(n_rows - 1, max(0, next_state[0])), min(n_cols -1, max(0, next_state[1]))), dtype=int)
        
        # Considering constant reward -1
        reward = rewards 
          
        # Calculating next_action_max_Q in the next_state by maximising the Q
        next_action_max_Q = valid_actions[*next_state][np.argmax(Q[*next_state][valid_actions[*next_state]])]

        # Update action-state value
        Q[*state][selected_action] += alpha * (reward + gamma * Q[*next_state][next_action_max_Q] - Q[*state][selected_action])

        # Update state and action for next loop
        state = next_state
        rewards_in_episodes[episode] += reward

        t += 1
    
    # Record the number of step
    time_elapsed_in_episode[episode] = t
    print(f"Episode: {episode}: {t}")

# Draw a table to visualize the greedy action based on the argmax of Q
greedy_actions = np.empty((n_rows, n_cols), dtype=object)
for row in range(n_rows):
    for col in range(n_cols):
        state = (row, col)
        if all(state == TERMINAL):
            greedy_actions[*state] = '■'
        elif grid[*state] == 1:
            greedy_actions[*state] = '#'
        else:
            greedy_action = valid_actions[*state][np.argmax(Q[*state][valid_actions[*state]])]
            if greedy_action == 0:
                greedy_actions[*state] = 'U'
            elif greedy_action == 1:
                greedy_actions[*state] = 'D'
            elif greedy_action == 2:
                greedy_actions[*state] = 'R'
            elif greedy_action == 3:
                greedy_actions[*state] = 'L'

print(tabulate(greedy_actions, tablefmt='fancy_grid'))

# Create a greedy episode
state = START_POINT
selected_action = valid_actions[*state][np.argmax(Q[*state][valid_actions[*state]])]
greedy_actions_scenario = np.empty((n_rows, n_cols), dtype=object)

t = 0
greedy_actions_scenario[*state] = t
while True:

    # Check being at terminal state
    if all(state == TERMINAL):
        break
    
    # Calulate next_state
    next_state = state + actions[selected_action]

    # Addjust the state so that the agent remains inside the gridworld
    next_state = np.array((min(n_rows - 1, max(0, next_state[0])), min(n_cols -1, max(0, next_state[1]))), dtype=int)
    
    reward = rewards # Considering constant reward -1

    # Select the greedy action
    next_action = valid_actions[*next_state][np.argmax(Q[*next_state][valid_actions[*next_state]])]

    # Update state and action for next loop
    state = next_state
    selected_action = next_action

    t += 1
    greedy_actions_scenario[*state] = t
    
    # To prevent infinite loop
    if t > 100:
        break

for row in range(n_rows):
    for col in range(n_cols):
        state = (row, col)
        if all(state == TERMINAL):
            greedy_actions_scenario[*state] = '■'
        elif grid[*state] == 1:
            greedy_actions_scenario[*state] = '#'

# Show the path according to greedy action
print(tabulate(greedy_actions_scenario, tablefmt='fancy_grid'))


# plot the path length per episode
# plt.plot(rewards_in_episodes)
# plt.grid(True)
# plt.xlabel('Episode')
# plt.ylabel('Total reward')
# plt.show()
