import numpy as np
np.random.seed(31)
States = [0, 1, 2, 3, 4, 5, 6]

V = np.zeros(np.size(States))
S = np.empty(1000, dtype=int)

gamma = 1
alpha = 0.04

for i in range(100):
    T = 0
    t = 0
    S[t] = 3
    reward = 0

    while True:
        s = S[t]
        new_s = s + np.random.choice([-1, 1])
        if new_s == 6:
            reward = 1 # Terminate at right
            T = t + 1
            break
        if new_s == 0:
            reward = 0 # Terminate at left
            T = t + 1
            break
        t += 1
        S[t] = new_s

    G = reward

    for t in range(T-1, -1, -1):
        s = S[t]
        G = G + 0 # reward is 0 for other transition
        V[s] += alpha*(G - V[s])

for i in range(len(V)):
    print(f"{V[i]:.3f} - {i/6:.3f}")
