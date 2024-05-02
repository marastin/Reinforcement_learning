import os
import random
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym # https://pypi.org/project/gymnasium/
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import PolicyNetwork

def evaluate(env, policy:PolicyNetwork, n_episode):
    score_per_episode = []
    for episode in range(n_episode):
        state, info = env.reset()
        score = 0

        while True:
            # action = policy.select_action(state=state)
            action = policy.select_greedy_action(state=state)

            next_state, reward, terminated, truncated, info = env.step(action)

            state = next_state
            score += reward

            if terminated or truncated:
                score_per_episode.append(score)
                break
        
    return np.average(score_per_episode)





if __name__ == "__main__":

    # Hyperparameters
    n_episodes = 2000
    hidden_layers = (512, 128)
    gamma = 0.99
    evaluation_goal = 475 # must be <= 500
    lr = 0.0005

    evaluation_score_history = []
    evaluation_score_history_mean = []
    
    env = gym.make('CartPole-v1')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNetwork(n_states, n_actions, hidden_dims=hidden_layers)
    optimizer = optim.RMSprop(policy.parameters(), lr = lr)

    # Main training loop
    for episode in range(n_episodes):

        state, info = env.reset()
        rewards = []
        log_pas = []
        score = 0

        while True:
            # action = policy.select_action(state=state)
            action, is_exploratory, log_pa, entropy = policy.full_pass(state=state)

            next_state, reward, terminated, truncated, info = env.step(action)
            
            rewards.append(reward)
            log_pas.append(log_pa)

            state = next_state
            score += reward

            if terminated or truncated:
                break
        

        evaluation_score = evaluate(env=env, policy=policy, n_episode=20)
        evaluation_score_history.append(evaluation_score)
        print(f"episod: {episode:4.0f} | evaluation score: {evaluation_score:4.0f}")

        # Learning
        Tau = len(rewards)

        discounts = np.logspace(0, Tau, num=Tau, base=gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:Tau-t] * rewards[t:]) for t in range(Tau)])
        
        # TODO Check bellow
        discounts = T.FloatTensor(discounts).unsqueeze(1)
        returns = T.FloatTensor(returns).unsqueeze(1)
        log_pas = T.cat(log_pas)

        # TODO Check shapes
        policy_loss = -(discounts * returns * log_pas).mean() 

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        evaluation_score_history_mean.append(np.mean(evaluation_score_history[-10:]))
        if  evaluation_score_history_mean[-1] > evaluation_goal:
            print(f"The goal achieved at episode {episode}")
            # If You want to save a checkpoint (uncomment below)
            # checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
            # checkpoint_file='q_network_ddqn'
            # q_network_online.save_checkpoint(checkpoint_dir=checkpoint_dir, checkpoint_file=checkpoint_file)
            break


plt.plot(evaluation_score_history)
plt.plot(evaluation_score_history_mean, 'r-')
plt.show()
