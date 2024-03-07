import numpy as np
from TileCodingSoftware import IHT, tiles
import matplotlib.pyplot as plt

np.random.seed(5)

actions = [-1, 0, 1]

X_MIN = -1.2
X_MAX = 0.5
V_MIN = -0.07
V_MAX = 0.07

epsilon = 0

n_episodes = 500

def get_next_state(x, v, action):
    next_v = v + 0.001 * action - 0.0025 * np.cos(3 * x)
    next_v = min(max(V_MIN, next_v), V_MAX)
    next_x = x + next_v
    next_x = min(max(X_MIN, next_x), X_MAX)
    reward = -1.0
    if next_x == X_MIN:
        next_v = 0.0
    return next_x, next_v, reward

numOfTilings = 8
maxSize = 4096
iht=IHT(maxSize)
W = np.zeros(maxSize)
x_scale = numOfTilings / (X_MAX - X_MIN)
v_scale = numOfTilings / (V_MAX - V_MIN)

alpha = 1 / numOfTilings

def get_eps_greedy_action(x,v):
    if np.random.binomial(1, epsilon) == 1:
        return np.random.choice(actions)
    values = []
    for a in actions:
        active_tiles = tiles(iht, numOfTilings, [x*x_scale, v*v_scale], [a])
        values.append(np.sum(W[active_tiles]))
    max_value = max(values)
    indices = [index for index, value in enumerate(values) if value == max_value]
    action =  actions[np.random.choice(indices)]
    return action

memory = 10**6
xt = np.zeros(memory, dtype=float)
vt = np.zeros(memory, dtype=float)
at = np.zeros(memory, dtype=int)
rt = np.zeros(memory, dtype=float)

# tiles(iht,8,[8*x/(0.5+1.2),8*xdot/(0.07+0.07)],A)

T_episodes = np.zeros(n_episodes, dtype=int)

for episode in range(n_episodes):
    x = np.random.uniform(-0.6, -0.4)
    v = 0.0
    action = get_eps_greedy_action(x, v)

    
    t = 0
    
    xt[t] = x
    vt[t] = v
    at[t] = action
    rt[t] = 0

    # the length of this episode
    T = float('inf')

    while True:
        
        t += 1
        if t < T:
            next_x, next_v, reward = get_next_state(x, v, action)
            next_action = get_eps_greedy_action(next_x, next_v)
            xt[t] = next_x
            vt[t] = next_v
            at[t] = action
            rt[t] = reward

            if next_x == X_MAX:
                T = t

        tau = t - 1
        if tau >= 0:
            returns = 0.0
            # calculate corresponding rewards
            for t in range(tau + 1, min(T, tau + 1) + 1):
                returns += rt[t]
            # add estimated state action value to the return
            if tau + 1 <= T:
                active_tiles = tiles(iht, numOfTilings, [xt[tau + 1]*x_scale, vt[tau + 1]*v_scale], [at[tau + 1]])
                returns += np.sum(W[active_tiles])
            # update the state value function
            if xt[tau] != X_MAX:
                active_tiles = tiles(iht, numOfTilings, [xt[tau]*x_scale, vt[tau]*v_scale], [at[tau]]) 
                value = np.sum(W[active_tiles])
                delta = alpha * (returns - value)
                for tile in active_tiles:
                    W[tile] += delta

        if tau == T - 1:
            break
        x = next_x
        v = next_v
        action = next_action
    
    print(episode, T)
    T_episodes[episode] = T

print(f"The number of indices used in iht: {iht.count()}")

plt.plot(T_episodes)
plt.show()