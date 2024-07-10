
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

    def act(self, obss, deterministic=False):
        actor_features = self.actor(obss)
        action_mask = len(actor_features) * [0]
        dist = MultiCategorical(
            [Categorical(logits=x + s) for x, s in zip(actor_features, action_mask)]
        )
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        log_prob = dist.log_probs(action)

        return action, log_prob

    def get_log_prob(self, obss, action):
        actor_features = self.actor(obss)
        action_mask = len(actor_features) * [0]
        dist = MultiCategorical(
            [Categorical(logits=x + s) for x, s in zip(actor_features, action_mask)]
        )
        log_prob = dist.log_probs(action)
        # print('log prob', action[0].shape, log_prob[0].shape, actor_features[0].shape)
        return log_prob

    def get_value(self, obss):
        return tr.cat(self.critic(obss), dim=-1)