import sys
# sys.path.append("../src")
import torch
import torch.nn as nn
import numpy as np
import random
import time
from config import *
from replay_buffer import *
from networks import *
import torch.distributions as ptd
import torch.nn.functional as F


class Agent:
    def __init__(self, env, path_save=PATH_SAVE, path_load=PATH_LOAD, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, tau=TAU, reward_scale=REWARD_SCALE):
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(env)
        self.actions_dim = env.action_space.shape[0]
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.path_save = path_save
        self.path_load = path_load
        indim = env.observation_space.shape[0]

        self.actor = Actor(indim=indim, actions_dim=self.actions_dim, name='actor', upper_bound=env.action_space.high, beta=self.actor_lr)
        self.critic_0 = Critic(indim=indim, name='critic_0', beta=self.critic_lr)
        self.critic_1 = Critic(indim=indim, name='critic_1', beta=self.critic_lr)
        self.critic_value = CriticValue(indim=indim, name='value', beta=self.critic_lr)
        self.critic_target_value = CriticValue(indim=indim, name='target_value', beta=self.critic_lr)

        # self.mse = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_0_optimizer = torch.optim.Adam(self.critic_0.parameters(), lr=self.critic_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_value_optimizer = torch.optim.Adam(self.critic_value.parameters(), lr=self.critic_lr)
        self.critic_target_value_optimizer = torch.optim.Adam(self.critic_target_value.parameters(), lr=self.critic_lr)

        self.reward_scale = reward_scale

        self.update_target_networks(tau=1)

    def update_target_networks(self, tau):
        if tau is None:
            tau = self.tau

        critic_value_params = self.critic_value.named_parameters()
        critic_target_value_params = self.critic_target_value.named_parameters()
        critic_value_params=dict(critic_value_params)
        critic_target_value_params=dict(critic_target_value_params)
        for name in critic_value_params:
            critic_value_params[name] = tau*critic_value_params[name].clone() + (1-tau)*critic_target_value_params[name].clone()

        self.critic_target_value.load_state_dict(critic_value_params)

    def add_to_replay_buffer(self, state, action, reward, new_state, done):
        self.replay_buffer.add_record(state, action, reward, new_state, done)

    def get_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.get_action_log_probs(state, reparameterization_trick=False)

        return actions.cpu().detach().numpy()[0]

    # def save(self):
    #     print('.... saving models ....')
    #     date_now = time.strftime("%Y%m%d%H%M")
    #     if not os.path.isdir(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}"):
    #         os.makedirs(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}")
    #     torch.save(self.actor, f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.actor.net_name}.pth")
    #     torch.save(self.critic_0, f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_0.net_name}.pth")
    #     torch.save(self.critic_1, f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_1.net_name}.pth")
    #     torch.save(self.critic_value, f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_value.net_name}.pth")
    #     torch.save(self.critic_target_value, f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_target_value.net_name}.pth")
    #     self.replay_buffer.save(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}")

    # def load(self):
    #     print('.... loading models ....')

    #     self.actor = torch.load(f"{self.path_load}/{self.actor.net_name}.pth")
    #     self.critic_0 = torch.load(f"{self.path_load}/{self.critic_0.net_name}.pth")
    #     self.critic_1 = torch.load(f"{self.path_load}/{self.critic_1.net_name}.pth")
    #     self.critic_value = torch.load(f"{self.path_load}/{self.critic_value.net_name}.pth")
    #     self.critic_target_value = torch.load(f"{self.path_load}/{self.critic_target_value.net_name}.pth")

    #     self.replay_buffer.load(f"{self.path_load}")

    def save(self):
        print('.... saving models ....')
        if not os.path.isdir(self.path_save):
            os.makedirs(self.path_save)
        self.actor.save_checkpoint()
        self.critic_0.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_value.save_checkpoint()
        self.critic_target_value.save_checkpoint()
        self.replay_buffer.save(self.path_save)
        self.path_load=self.path_save

    def load(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic_0.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_value.load_checkpoint()
        self.critic_target_value.load_checkpoint()
        self.replay_buffer.load(f"{self.path_load}")

    def learn(self):
        if self.replay_buffer.check_buffer_size() == False:
            return

        state, action, reward, new_state, done = self.replay_buffer.get_minibatch()
        # print("###############", type(reward))
        rewards = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        new_states = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        states = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(action, dtype=torch.float).to(self.actor.device)

        #Update critic_value
        value = torch.squeeze(self.critic_value(states), 1)
        target_value = torch.squeeze(self.critic_target_value(new_states), 1)
        target_value[done] = 0.0

        policy_actions, log_probs = self.actor.get_action_log_probs(states, reparameterization_trick=False)
        log_probs = torch.squeeze(log_probs,1)
        q_value_0 = self.critic_0(states, policy_actions)
        q_value_1 = self.critic_1(states, policy_actions)
        q_value = torch.squeeze(torch.minimum(q_value_0, q_value_1), 1)

        self.critic_value.optimizer.zero_grad()
        value_target = q_value - log_probs
        value_critic_loss = 0.5 * F.mse_loss(value, value_target)

        value_critic_loss.backward(retain_graph=True)
        self.critic_value.optimizer.step()

        #Update actor
        new_policy_actions, log_probs = self.actor.get_action_log_probs(states, reparameterization_trick=True)
        log_probs = torch.squeeze(log_probs, 1)
        new_q_value_0 = self.critic_0(states, new_policy_actions)
        new_q_value_1 = self.critic_1(states, new_policy_actions)
        new_q_value = torch.squeeze(torch.minimum(new_q_value_0, new_q_value_1), 1)

        actor_loss = log_probs - new_q_value
        actor_loss = torch.mean(actor_loss)

        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        #Update critics
        self.critic_0.optimizer.zero_grad()
        self.critic_1.optimizer.zero_grad()
        # print(reward.size(), target_value.size())
        q_pred = self.reward_scale * rewards + self.gamma * target_value
        old_q_value_0 = torch.squeeze(self.critic_0(states, actions), 1)
        old_q_value_1 = torch.squeeze(self.critic_1(states, actions), 1)
        critic_0_loss = 0.5 * F.mse_loss(old_q_value_0, q_pred)
        critic_1_loss = 0.5 * F.mse_loss(old_q_value_1, q_pred)

        critic_loss = critic_0_loss + critic_1_loss
        critic_loss.backward()
        self.critic_0.optimizer.step()
        self.critic_1.optimizer.step()

        self.update_target_networks(tau=self.tau)

        self.replay_buffer.update_n_games()