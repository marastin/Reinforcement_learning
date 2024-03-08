import numpy as np
import random
import matplotlib.pyplot as plt

Reward = -1
Gamma = 1
n_episodes = 100000
noise_prob = 0.1

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

Track_visited = -Track.copy()

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

Q = np.random.rand(rows, cols, n_vel, n_vel, n_actions) * 400 - 500 # action value 4-D matrix
C = np.zeros((rows, cols, n_vel, n_vel, n_actions)) # cumulative weights for each state-action pair
policy = np.ones((rows, cols, n_vel, n_vel), dtype=int) # policy matrix (target)
R = np.zeros((n_episodes, 1))

for r in range(rows):
    for c in range(cols):
        for h in range(n_vel):
            for v in range(n_vel):
                policy[r, c, h, v] = np.argmax(Q[r, c, h, v, :])


S = np.empty(10**6, dtype=tuple) # memory for state
A = np.empty(10**6, dtype=int)   # memory for selected actions by behaviour policy
P = np.empty(10**6, dtype=float) # memory for selected actions probabilities

def generate_episode(epsilon=0.1, noise=True):
    t = 0
    S[0] = (0, random.choice(start_line), 0, 0)
    
    while True:
        s = S[t]
        possible_actions = valid_actions[s[2]*n_vel + s[3]]
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
        
        if noise and random.random() < noise_prob:
            behavior = 0
            prob = noise_prob


        A[t] = behavior
        P[t] = prob
        act_b = actions[behavior]
        
        vel = (s[2] + act_b[0], s[3] + act_b[1])
        next_s = (s[0] + vel[0], s[1] + vel[1], vel[0], vel[1])

        # Check being in finish area
        if Track[next_s[0], next_s[1]] == 3:
            return t+1
        
        # check hit the boundries
        elif Track[next_s[0], next_s[1]] == 0:
            s = (0, random.choice(start_line), 0, 0)
        else:
            s = next_s

        t += 1
        S[t] = s

def update_policy(T):
    G = 0
    W = 1
    
    for t in range(T-1, -1, -1):
        s = S[t]
        a = A[t]
        sa = (*s, a)

        G = Reward + Gamma*G
        C[sa] += W
        Q[sa] += W * (G - Q[sa]) / C[sa]
        
        possible_actions = valid_actions[s[2]*n_vel + s[3]]
        policy[s] = possible_actions[np.argmax(Q[s][possible_actions])]

        Track_visited[s[0], s[1]] = 1
        if policy[s] != a:
            return
        
        W /= P[t]


def make_sample(n=1):
    for i in range(n):
        T = generate_episode(epsilon=0, noise=0)
        for t in range(T):
            print(f"S: {S[t]}", end=' ')
            print(f" -> A: {actions[A[t]]}", end="\n")
        print(f"Reward = {-T}")


def main():
    for i in range(n_episodes):
        T = generate_episode(epsilon=0.1)
        update_policy(T)    
        R[i] = Reward*T
        if (i+1) % 100 == 0:
            print(f"episode {i+1} / {n_episodes}: {R[i]}")

    make_sample(2)
    
    plt.plot(R)
    plt.title("Reward")
    plt.show()

    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.imshow(Track_visited, aspect='equal')
    plt.colorbar()
    plt.title("Track Visited show as 1")
    plt.show()




if __name__ == "__main__":
    main()