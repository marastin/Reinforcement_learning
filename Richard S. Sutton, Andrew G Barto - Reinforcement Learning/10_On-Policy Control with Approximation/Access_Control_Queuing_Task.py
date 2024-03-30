from ValueFunction import ValueFunctionTiling
import numpy as np
from tabulate import tabulate

np.random.seed(5)

server_states = list(range(11)) # 0:10
priorities = [1, 2, 4, 8]
actions = [0, 1]

alpha = 0.01
beta = 0.01


def greedy_action(q: ValueFunctionTiling, server_state, priority):
    values = []
    for action in actions:
        values.append(q.value([], (server_state, priority, action)))
    max_value = max(values)
    indices = [index for index, value in enumerate(values) if value == max_value]
    action =  actions[np.random.choice(indices)]
    return action

def eps_greedy_action(q: ValueFunctionTiling, server_state, priority, eps=0.1):
    if np.random.binomial(1, eps) == 1:
        return np.random.choice(actions)
    else:
        return greedy_action(q, server_state, priority)

def take_action(state, action):
    
    next_priority = np.random.choice(priorities)
    next_server_state = state[0]
    for i in range(state[0]):
        next_server_state -= np.random.binomial(1, 0.06)
    
    if action == 0:
        reward = 0
    elif action == 1:
        reward = state[1]
        next_server_state += 1
    
    return next_server_state, next_priority, reward
    
        
n_episodes = 200000 # It may set to 2000000

q = ValueFunctionTiling(8, alpha=alpha)
r_av = 0

state = (np.random.choice(server_states), np.random.choice(priorities))
action = eps_greedy_action(q, *state)

for episode in range(n_episodes):
    next_server_state, next_priority, reward = take_action(state, action)
    next_action = eps_greedy_action(q, next_server_state, next_priority)
    delta = reward - r_av + q.value([], (next_server_state, next_priority, next_action)) - q.value([], (*state, action))
    r_av += beta*delta
    q.update(delta, [], (*state, action))
    state = (next_server_state, next_priority)
    action = next_action
    if (episode + 1) % 1000 == 0:
        print(f"Episode: {episode+1} - Average reward: {r_av:.3f}")
    

print(f"Average reward: {r_av}")

greedy_actions_table = np.empty((len(server_states), len(priorities)), dtype=int)
for s in range(len(server_states)):
    for p in range(len(priorities)):
        greedy_actions_table[s,p] = greedy_action(q,s,p)

print(tabulate(greedy_actions_table.T, tablefmt='fancy_grid'))



