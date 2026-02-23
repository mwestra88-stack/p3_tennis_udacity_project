import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CriticNetwork(nn.Module):
    """Critic (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action size (int): dimension of each action
            seed (int): Random seed
            hidden_layers: number and sizes of hidden layers, provided as an array
        """
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.output_size = 1
        
        # First layer processes state only
        self.fc1 = nn.Linear(state_size, hidden_layers[0])

        # Second layer takes state features + action
        self.fc2 = nn.Linear(hidden_layers[0] + action_size, hidden_layers[1])
        
        self.output = nn.Linear(hidden_layers[1], self.output_size)

        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.output.weight.data.uniform_(-3e-3, 3e-3) # Following DDPG paper
       

    def forward(self, state, action):
        # Forward through each layer in `hidden_layers`, with ReLU activation
        xs=state
        xs = F.relu(self.fc1(xs))

        # Concatenate action
        x = torch.cat((xs, action), dim=1)

        x = F.relu(self.fc2(x))
        x = self.output(x)
            
        return x

    def hidden_init(self, layer):
        # Following the DDPG paper The other layers were initialized from uniform distributions [−1/√f , 1/√f] where f isthe fan-in of the layer
        f_in = layer.weight.data.size()[1]
        bound = 1. / np.sqrt(f_in)
        return (-bound, bound)