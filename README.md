# Bipedal Walking through the Soft Actor-Critic Algorithm
Although remarkably common in the animal world, two-legged locomotion has
been a particularly challenging open research problem. The ultimate goal of bipedal
walking research is to develop a framework that enables an agent to learn to ‘walk’
in any unknown environment where this is permitted by the laws of physics. This implementation seeks to train an agent to maintain human gaits with maximum
horizontal velocity in the Mujoco Humanoid-env. A first-principles approach to
the implementation of the Soft Actor-Critic (SAC) stochastic policy optimization
algorithm has been discussed and it’s slow-convergence and potential implications
analyzed.

Please refer ```Project_Report.pdf``` for further details about the algorithm and it's implementation details.

# Acknowledgement

Please check out the original paper at [Soft Actor-Critic:
Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) <br>
You may also find OpenAI's [tutorials](https://spinningup.openai.com/en/latest/algorithms/sac.html) on this to be very helpful.
