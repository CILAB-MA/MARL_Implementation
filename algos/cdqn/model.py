import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, cfgs):
        super(QNetwork, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(cfgs['joint_obs_dim'], cfgs['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Linear(cfgs['hidden_dim'], cfgs['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Linear(cfgs['hidden_dim'], cfgs['joint_action_dim'])
        )

    def forward(self, obs):
        return self.model(obs)

    def save(self, path):
        pass

    def load(self, path):
        pass
