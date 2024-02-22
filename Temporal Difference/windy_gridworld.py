import numpy as np

# Problem Parameters
n_rows = 7
n_cols = 10

# Define the states
states = np.zeros((n_rows, n_cols))

# Define the start point and terminal state
terminal_state = np.array((4, 7))
start_point = np.array((3, 0))


# Define reward for each transition
rewards = -1

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
            transition.append(0) # Move Down (N)
        if row < n_rows - 1:
            transition.append(1) # Move Up (S)
        if col > 0:
            transition.append(3) # Move Right (W)
        if col < n_cols - 1:
            transition.append(2) # Move Left (E)
        transitions[row, col] = transition

# Define transition duo to the wind
wind = np.empty((n_rows, n_cols, 2))
for row in range(n_rows):
    for col in range(n_cols):
        if col in (3, 4, 5, 8):
            wind[row, col, :] = np.array((-1, 0))
        elif col in (6, 7):
            wind[row, col, :] = np.array((-2, 0))
        else:
            wind[row, col, :] = np.array((0, 0))

# Define the number of episodes
num_episodes = 10000
time_elapsed_in_episode = np.zeros((num_episodes, 1), dtype=int)

# Initialize state-action values
Q = np.zeros((n_rows, n_cols, n_actions))

# Define the discount factor
gamma = 1.0

# Define the value of step size
alpha = 0.5

# Define the epsilon for epsilon-greedy policy
epsilon = 0.1

# Perform Temporal Difference updates
for episode in range(num_episodes):
    t = 0

    # Start from the start point
    state = start_point

    # Choose an action with epsilon-reedy policy
    if np.random.rand() > epsilon:
        # Select the greedy action
        selected_action = transitions[*state][np.argmax(Q[*state][transitions[*state]])]

    else:
        # Select a random action from valid transition
        selected_action = np.random.choice(list(transitions[*state]))

    
    # Generate an episode, calculate returns and update values
    while True:
        
        # Check being at terminal state
        if all(state == terminal_state):
            break
        
        next_state = state + actions[selected_action] + wind[*state] # TODO
        
        # Addjust the state so that the agent remains inside the gridworld
        next_state = np.array((min(n_rows - 1, max(0, next_state[0])), min(n_cols -1, max(0, next_state[1]))), dtype=int)

        # Considering constant reward -1
        reward = rewards

        if np.random.rand() > epsilon:
            # Select the greedy action
            next_action = transitions[*state][np.argmax(Q[*state][transitions[*state]])]

        else:
            # Select a random action from valid transition
            next_action = np.random.choice(list(transitions[*state]))

                
        # Update action-state value
        Q[*state][selected_action] += alpha * (reward + gamma * Q[*next_state][next_action] - Q[*state][selected_action]) # TODO

        # Update state and action for next loop
        state = next_state
        selected_action = next_action

        t += 1
    if (episode + 1) % 100 == 0:
        print(f"episode {episode + 1}: T = {t}")