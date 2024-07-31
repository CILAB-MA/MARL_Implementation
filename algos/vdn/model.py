import torch.nn as nn

from utils.model_func import MultiAgentFCNetwork


class QNetwork(nn.Module):
    def __init__(self, cfgs):
        super(QNetwork, self).__init__()

        self.obs_dim = cfgs['obs_dim']
        self.hidden_dim = cfgs['hidden_dim']
        self.action_dim = cfgs['action_dim']
        self.lr = cfgs['lr']
        self.epsilon = cfgs['epsilon']
        self.gamma = cfgs['gamma']
        self.batch_size = cfgs['batch_size']
        self.n_agents = cfgs['num_agent']
        self.target_update_freq = cfgs['target_update_freq']

        self.network = MultiAgentFCNetwork([self.obs_dim] * self.n_agents,
                                           [self.hidden_dim, self.hidden_dim],
                                           [self.action_dim] * self.n_agents)

    def forward(self, obs):
        return self.network(obs)
