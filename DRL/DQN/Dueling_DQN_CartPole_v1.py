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

from buffer import ReplayBuffer
from networks import DuelingQNetwork
from utils import evaluation


if __name__ == "__main__":

    # Hyperparameters
    n_episodes = 200
    batch_size = 64
    hidden_layers = (512, 128)
    buffer_capacity = 50000
    target_network_update_rate = 10
    evaluation_goal = 475 # must be <= 500
    gamma = 1
    epsilon_init = 1
    epsilon_decay_steps = 5000
    epsilon_min = 0.1
    evaluation_rate = 2
    episode_max_length = 1000
    
    evaluation_results = []

    
    # Initialize environment
    env = gym.make('CartPole-v1')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize Q-network and optimizer
    q_network_online = DuelingQNetwork(n_states, n_actions, hidden_dims=hidden_layers)
    q_network_target = DuelingQNetwork(n_states, n_actions, hidden_dims=hidden_layers)
    optimizer = optim.RMSprop(q_network_online.parameters(), lr = 0.0005)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    t = 0 # the number of q_network_online updates
    epsilon = epsilon_init

    # Main training loop
    for episode in range(n_episodes):
        state, info = env.reset()
        total_reward = 0
        
        for _ in range(episode_max_length):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with T.no_grad():
                    q_values = q_network_online(T.tensor(state, dtype=T.float32).to(q_network_online.device))
                    action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Store transition in replay buffer
            buffer.push(state, action, reward, next_state, terminated, truncated)
            state = next_state

            # Update Q-network (only if there is enough of data in the buffer)
            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                b_state, b_action, b_reward, b_next_state, b_terminated, b_truncated = zip(*batch)
                
                # Convert to tensors and move to device
                b_state = T.tensor(b_state, dtype=T.float32).to(q_network_online.device)
                b_action = T.tensor(b_action, dtype=T.int64).to(q_network_online.device).unsqueeze(1)  # must be int64 in order to be compatible with gather function
                b_reward = T.tensor(b_reward, dtype=T.float32).to(q_network_online.device).unsqueeze(1)
                b_next_state = T.tensor(b_next_state, dtype=T.float32).to(q_network_online.device)
                b_terminated = T.tensor(b_terminated, dtype=T.float32).to(q_network_online.device).unsqueeze(1)
                b_truncated = T.tensor(b_truncated, dtype=T.float32).to(q_network_online.device).unsqueeze(1)


                '''
                q_network(b_states) returns a tensor of shape (batch_size, n_actions) and b_action is a tensor
                of shape (batch_size,) containing action indices. b_action.unsqueeze(1) would change the shape
                to (batch_size, 1), aligning it with the shape of the Q-values tensor along dimension 1.
                Then, .gather(1, b_actions.unsqueeze(1)) gathers Q-values corresponding to the specified 
                actions (dimension 1). Note that dimension 0 is related to batch.
                '''
                # Compute Q-values
                b_q_value = q_network_online(b_state).gather(1, b_action)

                # Compute target values
                with T.no_grad():
                    b_next_q_value = q_network_target(b_next_state)
                    b_max_next_q_value = T.max(b_next_q_value, dim=1, keepdim=True)[0]
                    b_target = b_reward + (1 - b_terminated) * gamma * b_max_next_q_value # TD Method
                
                # Compute loss and update network
                loss = F.mse_loss(b_q_value, b_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t += 1

                if t % target_network_update_rate == 0:
                    q_network_target.load_state_dict(q_network_online.state_dict())

                    # The line above is equivalent to:
                    # for target, online in zip(q_network_target.parameters(), q_network_online.parameters()):
                    #     target.data.copy_(online.data)
            
            if terminated or truncated:
                print(f"Ep {episode:4.0f} | TR: {total_reward:4.0f} | ε: {epsilon:.4f} | t: {t}")
                break
        
        
        # Epsilon reduces linearly until epsilon_min
        epsilon = np.maximum((epsilon_init - epsilon_min) * (1 - t / epsilon_decay_steps) + epsilon_min, epsilon_min)


        # Perform evaluation periodically
        if episode % evaluation_rate == 0:
            evaluation_result = evaluation(env, q_network_online)
            print(f"------------ Evaluation: Episode {episode}: Total Reward: {evaluation_result}")
            evaluation_results.append(evaluation_result)
        
        if np.mean(evaluation_results[-10:]) > evaluation_goal:
            print(f"Evaluation goal achieved at episode {episode}")
            # If You want to save a checkpoint (uncomment below)
            # checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
            # checkpoint_file='q_network_ddqn'
            # q_network_online.save_checkpoint(checkpoint_dir=checkpoint_dir, checkpoint_file=checkpoint_file)
            break
        
    

    # q_network_tmp = QNetwork(n_states, n_actions)
    # q_network_tmp.load_checkpoint(checkpoint_dir, checkpoint_file)

    
    # Plot evaluation results
    x = np.arange(1, len(evaluation_results)*evaluation_rate + 1, evaluation_rate)
    plt.plot(x, evaluation_results)
    plt.title("Evaluation Results - Dueling DQN Method")
    plt.xlabel("number of episode")
    plt.ylabel("score")
    plt.show()