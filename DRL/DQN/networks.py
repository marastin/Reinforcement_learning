import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    """Q-value network"""

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
        x = self.output_layer(x)
        
        return x
    
    def save_checkpoint(self, checkpoint_dir=None, checkpoint_file='q_network_dqn'):
        """Save model checkpoint to file"""
        if checkpoint_dir is None:
            checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
        T.save(self.state_dict(), os.path.join(checkpoint_dir, checkpoint_file))

    def load_checkpoint(self, checkpoint_dir, checkpoint_file):
        """Load model checkpoint from file"""
        self.load_state_dict(T.load(os.path.join(checkpoint_dir, checkpoint_file)))



class DuelingQNetwork(nn.Module):
    """Dueling Q-value network"""

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
        # self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.value_output = nn.Linear(hidden_dims[-1], 1)
        self.advantage_output = nn.Linear(hidden_dims[-1], output_dim)
        
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
        a = self.advantage_output(x)
        v = self.value_output(x)
        v = v.expand_as(a)
        q = v + a - a.mean().expand_as(a)
        
        return q
        
    def save_checkpoint(self, checkpoint_dir=None, checkpoint_file='q_network_dqn'):
        """Save model checkpoint to file"""
        if checkpoint_dir is None:
            checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
        T.save(self.state_dict(), os.path.join(checkpoint_dir, checkpoint_file))

    def load_checkpoint(self, checkpoint_dir, checkpoint_file):
        """Load model checkpoint from file"""
        self.load_state_dict(T.load(os.path.join(checkpoint_dir, checkpoint_file)))