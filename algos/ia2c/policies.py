
from torch.distributions.categorical import Categorical
import torch.optim as optim
from utils.model_func import MultiAgentFCNetwork, MultiCategorical
import torch.nn as nn
import torch as tr
from gym.spaces import flatdim

class ActorCriticPolicy(nn.Module):

    def __init__(self, model_cfg):
        super(ActorCriticPolicy, self).__init__()
        obs_shape = [flatdim(o) for o in model_cfg['observation_space']]
        action_shape = [flatdim(a) for a in model_cfg['action_space']]
        self.actor = MultiAgentFCNetwork(obs_shape,
                                         [64, 64],
                                         action_shape, True)
        self.critic = MultiAgentFCNetwork(obs_shape,
                                         [64, 64], [1] * model_cfg['num_agent'], True)

        optimizer = getattr(optim, 'Adam')
        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer = optimizer(self.parameters(), lr=3.e-4)

    def act(self, inputs, action_mask=None):
        actor_features = self.actor(inputs)
        dist = self.get_dist(actor_features, action_mask)
        action = dist.sample()
        return action

    def get_value(self, inputs):
        return tr.cat(self.critic(inputs), dim=-1)

    def evaluate_actions(self, inputs, action, action_mask=None, state=None):
        if not state:
            state = inputs
        value = self.get_value(state)
        actor_features = self.actor(inputs)
        dist = self.get_dist(actor_features, action_mask)
        action_log_probs = tr.cat(dist.log_probs(action), dim=-1)
        dist_entropy = dist.entropy()
        dist_entropy = sum([d.mean() for d in dist_entropy])

        return (
            value,
            action_log_probs,
            dist_entropy,
        )

    def get_dist(self, actor_features, action_mask):
        if action_mask:
            action_mask = [-9999999 * (1 - action_mask) for a in action_mask]
        else:
            action_mask = len(actor_features) * [0]

        dist = MultiCategorical(
            [Categorical(logits=x + s) for x, s in zip(actor_features, action_mask)]
        )
        return dist