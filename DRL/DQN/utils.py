import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym # https://pypi.org/project/gymnasium/
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


def evaluation(env, q_network, episode_max_length=1000, seed=None):
    """Evaluate the agent's performance under greedy policy"""
    if seed == None: seed = time.time_ns()
    state, info = env.reset(seed=seed)
    total_reward = 0
    
    for _ in range(episode_max_length):
        with T.no_grad():
            q_values = q_network(T.tensor(state, dtype=T.float32).to(q_network.device))
            action = q_values.argmax().item()
        
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    return total_reward