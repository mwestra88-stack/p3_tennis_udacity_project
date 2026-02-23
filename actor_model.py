import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers: number and sizes of hidden layers, provided as an array
        """
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        # Add the first layer, input to a hidden layer        
        self.fc1 = nn.Linear(state_size, hidden_layers[0])

        # Second layer takes state features
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        
        self.output = nn.Linear(hidden_layers[1], action_size)

        # Add weights
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.output.weight.data.uniform_(-3e-3, 3e-3) # Following DDPG paper

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Forward through each layer in `hidden_layers`, with ReLU activation and tanh output function
        x=state

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
            
        return torch.tanh(x)

    def hidden_init(self, layer):
        # Following the DDPG paper The other layers were initialized from uniform distributions [−1/√f , 1/√f] where f isthe fan-in of the layer
        f_in = layer.weight.data.size()[1]
        bound = 1. / np.sqrt(f_in)
        return (-bound, bound)
