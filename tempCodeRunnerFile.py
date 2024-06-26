import sys
sys.path.append("../src")
import gym
import pybullet_envs
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
# import wandb
from config import *
from replay_buffer import *
from networks import *
from agent import *

config = dict(
  learning_rate_actor = ACTOR_LR,
  learning_rate_critic = ACTOR_LR,
  batch_size = BATCH_SIZE,
  architecture = "SAC",
  infra = "Colab",
  env = ENV_NAME
)

# wandb.init(
#   project=f"WalkRL {ENV_NAME.lower()}",
#   tags=["SAC", "FCL", "RL"],
#   config=config,
# )

env = gym.make(ENV_NAME)
agent = Agent(env)

scores = []
evaluation = True

if PATH_LOAD is not None:
    print("loading weights")
    observation = env.reset()
    action, log_probs = agent.actor.get_action_log_probs(observation[None, :], False)
    agent.actor(observation[None, :])
    agent.critic_0(observation[None, :], action)
    agent.critic_1(observation[None, :], action)
    agent.critic_value(observation[None, :])
    agent.critic_target_value(observation[None, :])
    agent.load()
    print(agent.replay_buffer.buffer_counter)
    print(agent.replay_buffer.n_games)

for _ in tqdm(range(MAX_GAMES)):
    start_time = time.time()
    states = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.get_action(states)
        new_states, reward, done, info = env.step(action)
        score += reward
        agent.add_to_replay_buffer(states, action, reward, new_states, done)
        agent.learn()
        states = new_states
    
    scores.append(score)
    agent.replay_buffer.update_n_games()

    # wandb.log({'Game number': agent.replay_buffer.n_games, '# Episodes': agent.replay_buffer.buffer_counter, 
    #            "Average reward": round(np.mean(scores[-10:]), 2), \
    #                   "Time taken": round(time.time() - start_time, 2)})
    
    if (_ + 1) % SAVE_FREQUENCY == 0:
        print("saving...")
        agent.save()
        print("saved")