import numpy as np
import random
import matplotlib.pyplot as plt

# number of race track rows and columns
margin = 5
rows, cols = 32 + margin, margin + 17 + margin # 37, 27

Track = np.zeros((rows, cols))
Track[0, 8:15] = 2  # start line
start_line = list(range(8,15))
Track[1:4, 8:15] = 1
Track[3:11, 7:15] = 1
Track[10:19, 6:15] = 1
Track[18:26, 5:15] = 1
Track[25, 5:16] = 1
Track[26:28, 5:21] = 1
Track[28, 6:21] = 1
Track[29:31, 7:21] = 1
Track[31, 8:21] = 1
Track[26:32, 21:27] = 3  # finish line

# plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
# plt.imshow(Track, cmap='tab10', aspect='equal')
# plt.show()


actions = [
    (0, 0),
    (0, 1),
    (0, -1),
    (1, 0),
    (1, 1),
    (1, -1),
    (-1, 0),
    (-1, 1),
    (-1, -1),
]

n_actions = len(actions)
n_vel = 5 # vel = 0,1,2,3,4

valid_actions = []
for v1 in range(n_vel):
    for v2 in range(n_vel):
        tmp = []
        for idx, a in enumerate(actions):
            if (0 <= a[0] + v1 < 5) and (0 <= a[1] + v2 < 5) and (a[0] + a[1] + v1 + v2 > 0):
                tmp.append(idx)
        valid_actions.append(tmp)

Q = np.random.rand([rows, cols, n_vel, n_vel, n_actions]) # action value 4-D matrix
C = np.zeros([rows, cols, n_vel, n_vel, n_actions]) # cumulative weights for each state-action pair
policy = np.ones((rows, cols, n_vel, n_vel), dtype=int) # policy matrix

for r in range(rows):
    for c in range(cols):
        for h in range(n_vel):
            for v in range(n_vel):
                policy[r, c, h, v] = np.argmax(Q[r, c, h, v, :])


S = np.empty(10**6, dtype=tuple) # memory for state
B = np.empty(10**6, dtype=int)   # memory for selected actions by behaviour policy
P = np.empty(10**6, dtype=float) # memory for selected actions probabilities

def generate_episode(epsilon):

    t = 0
    S[0] = (0, random.choice(start_line), 0, 0)
    
    while True:
        s = S[t]
        possible_actions = valid_actions[s[3]*n_vel + s[4]]
        n_possible_actions = len(possible_actions)

        a_policy = policy[*s] # action based on the target policy
        a_policy_is_valid = a_policy in possible_actions
        if np.random.rand() > epsilon:
            if a_policy_is_valid:
                behavior = a_policy # behaviour policy is the same as target policy
                prob = 1 - epsilon + epsilon / n_possible_actions
            else:
                behavior = random.choice(possible_actions)
                prob = 1 / n_possible_actions
                
        else:
            behavior = random.choice(possible_actions)
            prob = (epsilon if a_policy_is_valid else 1) / n_possible_actions
        
        B[t] = behavior
        P[t] = prob
        act_b = actions(behavior)
        



            















'''
n_episodes = 100
epsilon = 0.1

for episode in range(n_episodes):
    t = 0
    state[t] = (0, random.choice(start_line), 0, 0) # dir_1, dir_2, v_dir_1, v_dir_2

    while True:
        s = state[t]
        possible_act_1 = possible_actions_1[s[2]]
        possible_act_2 = possible_actions_2[s[3]]

        if random.random() < epsilon:
            act_1 = random.choice(possible_act_1)
            act_2 = random.choice(possible_act_2)
            














ε = 0.1
gamma = 1
actions = [
    (0, 0),
    (0, 1),
    (0, -1),
    (1, 0),
    (1, 1),
    (1, -1),
    (-1, 0),
    (-1, 1),
    (-1, -1),
]
act_len = len(actions)

# 5 velocity values 0 to 4, represented by 1 to 5
vel_len = 5

# initialise Q, C and π
# state is 4-tuple: (row, col, velocity_horizontal, velocity_vertical)
Q = np.random.rand(rows, cols, vel_len, vel_len, act_len) * 400 - 500
C = np.zeros((rows, cols, vel_len, vel_len, act_len))
π = np.ones((rows, cols, vel_len, vel_len), dtype=int)
for r in range(rows):
    for c in range(cols):
        for h in range(vel_len):
            for v in range(vel_len):
                π[r, c, h, v] = np.argmax(Q[r, c, h, v, :])



# set up the 1st race track map, origin (1,1) is at the bottom left,
# boundaries are marked with 1
track = np.zeros((rows, cols), dtype=int)
track[31:32, 0:3] = 1
track[30:31, 0:2] = 1
track[29:30, 0:2] = 1
track[28:29, 0:1] = 1
track[0:18, 0:1] = 1
track[0:10, 1:2] = 1
track[0:3, 2:3] = 1
track[0:26, 9:16] = 1
track[25:26, 9:10] = 0
start_cols = list(range(3, 9))  # start line columns
fin_cells = set([
    (26, cols),
    (27, cols),
    (28, cols),
    (29, cols),
    (30, cols),
    (31, cols),
])  # finish cells

# plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
# plt.imshow(track, cmap='tab10', aspect='equal')
# plt.show()

valid_acts = [
    [a[0] for a in enumerate(actions) if (h + a[1][0] in range(5)) and (v + a[1][1] in range(5)) and not((h + a[1][0]) == 0 and (v + a[1][1]) == 0)]
    for h in range(vel_len) for v in range(vel_len)
]
print(valid_acts)
print(len(valid_acts))

# pre-allocated state, action, probability trajectory array
S = np.empty(10**6, dtype=tuple)
A = np.empty(10**6, dtype=int)
B = np.empty(10**6, dtype=float)
'''