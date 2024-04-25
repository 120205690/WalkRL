import sys
sys.path.append("../src")
from replay_buffer import *
from config import *
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as ptd

#Q
class Critic(nn.Module):
    def __init__(self, beta, name, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1):
        super().__init__()
        self.net_name=name
        self.hidden_0=hidden_0
        self.hidden_1=hidden_1
        
        self.dense_0=nn.LazyLinear(self.hidden_0) #in_dim = env.observation_space.shape[0]
        nn.ReLU()
        self.dense_1=nn.Linear(self.hidden_0, self.hidden_1)
        nn.ReLU()
        self.q_value=nn.Linear(self.hidden_1, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_action_value = self.dense_0(torch.cat(state, action))
        assert state_action_value.size()[0]==self.hidden_0
        state_action_value = self.dense_1(state_action_value)

        q_value = self.q_value(state_action_value)
        assert q_value.size()[0]==1
        return q_value

#State Value
class CriticValue(nn.Module):
    def __init__(self, beta, name, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1):
        super().__init__()
        self.net_name=name
        self.hidden_0=hidden_0
        self.hidden_1=hidden_1
        
        self.dense_0=nn.LazyLinear(self.hidden_0) #in_dim = env.observation_space.shape[0]
        nn.ReLU()
        self.dense_1=nn.Linear(self.hidden_0, self.hidden_1)
        nn.ReLU()
        self.q_value=nn.Linear(self.hidden_1, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.dense_0(state)
        assert value.size()[0]==self.hidden_0
        value = self.dense_1(value)
        
        value = self.q_value(value)
        assert value.size()[0]==1
        return value
    
class Actor(nn.Module):
    def __init__(self, beta, name, upper_bound, actions_dim, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1, epsilon=EPSILON, log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        
        super().__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.actions_dim = actions_dim
        self.net_name = name
        self.upper_bound = upper_bound
        self.epsilon = epsilon
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.dense_0=nn.LazyLinear(self.hidden_0) #in_dim = env.observation_space.shape[0]
        nn.ReLU()
        self.dense_1=nn.Linear(self.hidden_0, self.hidden_1)
        nn.ReLU()
        self.mean=nn.Linear(self.hidden_1, self.actions_dim)
        self.std=nn.Linear(self.hidden_1, self.actions_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        policy = self.dense_0(state)
        policy = self.dense_1(policy)

        mean = self.mean(policy)
        log_std = self.log_std(policy)

        log_std = torch.clip_by_value(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std
    
    def get_action_log_probs(self, state, reparameterization_trick=True):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)

        normal_distr = ptd.Normal(mean, std)
        
        # Reparameterization trick
        z = torch.randn(shape=mean.size())

        if reparameterization_trick:
            actions = mean + std * z
        else:
            actions = normal_distr.sample()

        action = torch.tanh(actions) * self.upper_bound.to(self.device)
        log_probs = normal_distr.log_prob(actions) - torch.log(1 - torch.pow(action,2) + self.epsilon)
        log_probs = torch.sum(log_probs, axis=1, keepdims=True)

        return action, log_probs
    