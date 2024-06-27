import numpy as np
import torch as tr
import torch.optim as optim
torch.distributions.categorical.Categorical
from algos.ia2c.policies import ActorCriticPolicy
class IA2CAgent:
    def __init__(self, model, env_cfgs, model_cfgs, train_cfgs):
        self.models = [ActorCriticPolicy(model_cfgs) for _ in range(env_cfgs['num_agent'])]
        self.num_agent = env_cfgs['num_agent']
        self.num_process = env_cfgs['num_process']
        self.env_cfgs = env_cfgs
        self.model_cfgs = model_cfgs
        self.train_cfgs = train_cfgs

    def update(self, storage):
        pass

    def act(self, obs, deterministic=False):
        total_actions = tr.zeros(self.num_process, self.num_agent)
        for i in range(self.num_agent):
            action = self.models[i].act(obs, deterministic=deterministic)
            total_actions[i] = action
        return total_actions
