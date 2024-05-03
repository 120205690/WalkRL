"""Basic Dense
=======================================

Visualization of basic dense model
"""  # noqa: D205

import matplotlib.pyplot as plt
import torch
from torch import nn
from config import *


class Critic(nn.Module):
    def __init__(self, beta, indim, name, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1, chkpt_dir='tmp/sac'):
        super().__init__()
        self.net_name=name
        self.hidden_0=hidden_0
        self.hidden_1=hidden_1
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        indim=61
        self.dense_0=nn.Linear(indim, self.hidden_0) #in_dim = env.observation_space.shape[0]
        nn.ReLU()
        self.dense_1=nn.Linear(self.hidden_0, self.hidden_1)
        nn.ReLU()
        self.q_value=nn.Linear(self.hidden_1, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        # print(state.size(), action.size())
        # print()
        state_action_value = self.dense_0(torch.cat((state, action), dim=1))
        # assert state_action_value.size()[0]==self.hidden_0
        state_action_value = self.dense_1(state_action_value)

        q_value = self.q_value(state_action_value)
        # assert q_value.size()[0]==1
        return q_value


from torchview import draw_graph

model = Critic(beta=CRITIC_LR, indim=376, name='critic_0')
batch_size = 2
# device='meta' -> no memory is consumed for visualization
model_graph = draw_graph(model, input_data= input_size=(batch_size, 128), device='meta')
model_graph.visual_graph