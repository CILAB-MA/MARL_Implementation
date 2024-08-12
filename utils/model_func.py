from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class MultiCategorical:
    def __init__(self, categoricals):
        self.categoricals = categoricals

    def __getitem__(self, key):
        return self.categoricals[key]

    def sample(self):
        return [c.sample().unsqueeze(-1) for c in self.categoricals]

    def log_probs(self, actions):

        return [c.log_prob(a.squeeze(-1)).unsqueeze(-1) for c, a in zip(self.categoricals, actions)]

    def mode(self):
        return [c.mode for c in self.categoricals]

    def entropy(self):
        return [c.entropy() for c in self.categoricals]


def make_fc(dims, activation=nn.ReLU, final_activation=None, use_orthogonal_init=True):
    mods = []

    input_size = dims[0]
    h_sizes = dims[1:]

    mods = [nn.Linear(input_size, h_sizes[0])]
    for i in range(len(h_sizes) - 1):
        mods.append(activation())
        layer = nn.Linear(h_sizes[i], h_sizes[i + 1])
        mods.append(layer)

    if final_activation:
        mods.append(final_activation())

    return nn.Sequential(*mods)


class MultiAgentNetwork(ABC, nn.Module):
    def _make_fc(self, dims, activation=nn.ReLU, final_activation=None, use_orthogonal_init=True):
        return make_fc(dims, activation, final_activation, use_orthogonal_init)


class MultiAgentFCNetwork(MultiAgentNetwork):
    def __init__(self, input_sizes, idims, output_sizes, use_orthogonal_init=True):
        super().__init__()
        assert len(input_sizes) == len(output_sizes), "Expect same number of input and output sizes"
        self.independent = nn.ModuleList()

        for in_size, out_size in zip(input_sizes, output_sizes):
            dims = [in_size] + idims + [out_size]
            self.independent.append(self._make_fc(dims, use_orthogonal_init=use_orthogonal_init))


    def forward(self, inputs: List[torch.Tensor]):
        futures = [
            torch.jit.fork(model, x) for model, x in zip(self.independent, inputs)
        ]
        results = [torch.jit.wait(fut) for fut in futures]
        results = torch.stack(results)
        return results


class CriticFCNetwork(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim):
        super(CriticFCNetwork, self).__init__()

        # state, action, other_actions. index
        input_dim = (state_dim * agent_num) + action_dim + (action_dim * agent_num) + agent_num

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
