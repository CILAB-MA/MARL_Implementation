import numpy as np
import torch
import torch.optim as optim


class CA2CAgent:
    def __init__(self, model, env_cfgs, model_cfgs, train_cfgs):
        self.model = model
        self.env_cfgs = env_cfgs
        self.model_cfgs = model_cfgs
        self.train_cfgs = train_cfgs

    def update(self, storage):
        pass

    def act(self, obs):
        # actions = np.random.choice(self.env_cfgs['action_space'], self.env_cfgs['num_agent'])
        actions = self.env_cfgs['action_space'].sample()
        return actions