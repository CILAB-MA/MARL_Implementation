import numpy as np
import torch
import torch.optim as optim
torch.distributions.categorical.Categorical
from algos.ia2c.policies import ActorCriticPolicy
class IA2CAgent:
    def __init__(self, model, env_cfgs, model_cfgs, train_cfgs):
        self.model = ActorCriticPolicy(model_cfgs)
        self.env_cfgs = env_cfgs
        self.model_cfgs = model_cfgs
        self.train_cfgs = train_cfgs

    def update(self, storage):
        pass

    def act(self, obs, deterministic=False):
        action = self.model.act(obs, deterministic=deterministic)
        return action
