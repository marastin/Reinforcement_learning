import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyNetwork(nn.Module):
    """Policy network"""

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(32,32),
                 activation_fc = F.relu,):
        """Initialize Q-network"""
        super().__init__()

        self.activation_fc = activation_fc
        
        # Define layers
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Check if GPU is available
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        """Forward pass through the network"""
        x = state
        
        # Convert input to tensor if necessary
        if not isinstance(x, T.Tensor):
            x = T.tensor(x, device=self.device, dtype=T.float32)
            x = x.unsqueeze(0)
        
        # Forward pass
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        logits = self.output_layer(x)
        
        return logits
    
    def select_action(self, state):
        logits = self.forward(state)
        dist = T.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()
    
    def select_greedy_action(self, state):
        logits = self.forward(state)
        return np.argmax(logits.detach().numpy())
    
    def full_pass(self, state):
        logits = self.forward(state)
        dist = T.distributions.Categorical(logits=logits)
        action = dist.sample()
        
        log_pa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        is_exploratory = action != np.argmax(logits.detach().numpy())
        
        return action.item(), is_exploratory.item(), log_pa, entropy
       
