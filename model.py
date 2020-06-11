import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, num_agents, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            num_agents (int) number of agents in the exercise
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = seed
        self.fc1 = nn.Linear((state_size * num_agents), fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialise network parameters"""
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return  torch.tanh(self.fc3(x))
    
    def hidden_init(self, layer):
        """Set bounds for the initial values of layer weights"""
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, num_agents, seed, fcs1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            num_agents (int) number of agents in the exercise
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = seed
        self.fcs1 = nn.Linear((state_size * num_agents), fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+ (action_size * num_agents), fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
       
        #self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.d1 = nn.Dropout(0.1)
        self.d2 = nn.Dropout(0.1)
        self.reset_parameters()
    def reset_parameters(self):
        """Initialise network parameters"""
        self.fcs1.weight.data.uniform_(*self.hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.d1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.d2(self.fc2(x)))
        return self.fc3(x)
    
    def hidden_init(self, layer):
        """Set bounds for the initial values of layer weights"""
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)