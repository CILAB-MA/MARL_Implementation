from gym.spaces import flatdim
import torch
from torch.distributions import Categorical
import torch.nn as nn
from torch import optim

from utils import model_func


def _split_batch(splits):
    def thunk(batch):
        return torch.split(batch, splits, dim=-1)

    return thunk


@torch.jit.script
def compute_returns(rewards, done, next_value, gamma: float):
    returns = [next_value]
    for i in range(len(rewards) - 1, -1, -1):
        ret = rewards[i] + gamma * returns[0] * (1 - done[i, :].unsqueeze(1))
        returns.insert(0, ret)
    return returns


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space, cfg, device):
        super(ActorCritic, self).__init__()
        self.gamma = cfg.model_cfgs['gamma']
        self.entropy_coef = cfg.model_cfgs['entropy_coef']
        self.grad_clip = cfg.model_cfgs['grad_clip']
        self.value_loss_coef = cfg.model_cfgs['value_loss_coef']
        self.n_steps = cfg.env_cfgs['n_steps']

        self.n_agents = len(obs_space)
        obs_shape = [flatdim(o) for o in obs_space]  # [5, 5]
        action_shape = [flatdim(a) for a in action_space]  # [142, 142]
        critic_obs_shape = self.n_agents * [sum(obs_shape)]  # concat obs

        self.actor = model_func.MultiAgentFCNetwork(obs_shape, [64, 64], action_shape, True)
        self.critic = model_func.CriticFCNetwork(self.n_agents, critic_obs_shape[0], action_shape[0])
        self.target_critic = model_func.CriticFCNetwork(self.n_agents, critic_obs_shape[0], action_shape[0])

        self.soft_update(1.0)
        self.to(device)

        optimizer = getattr(optim, cfg.model_cfgs['optimizer'])
        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer_class = optimizer

        lr = cfg.model_cfgs['lr']
        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.target_update_interval_or_tau = 200

        self.split_obs = _split_batch([flatdim(s) for s in obs_space])
        self.split_act = _split_batch(self.n_agents * [1])

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError(
            "Forward not implemented. Use act, get_value, get_target_value or evaluate_actions instead.")

    def get_dist(self, actor_features, action_mask):
        if action_mask:
            action_mask = [-9999999 * (1 - action_mask) for a in action_mask]
        else:
            action_mask = len(actor_features) * [0]

        dist = model_func.MultiCategorical(
            [Categorical(logits=x + s) for x, s in zip(actor_features, action_mask)]
        )
        return dist

    def act(self, inputs, action_mask=None):
        actor_features = self.actor(inputs)
        dist = self.get_dist(actor_features, action_mask)
        action = dist.sample()
        return action

    '''수정 필요'''
    def get_value(self, inputs):
        inputs = self.n_agents * [torch.cat(inputs, dim=-1)]

        return torch.cat(list(torch.unbind(self.critic(inputs))), dim=-1)

    '''수정 필요'''
    def get_target_value(self, inputs):

        inputs = self.n_agents * [torch.cat(inputs, dim=-1)]

        return torch.cat(list(torch.unbind(self.target_critic(inputs))), dim=-1)

    def evaluate_actions(self, inputs, action, action_mask=None, state=None):
        if not state:
            state = inputs
        value = self.get_value(state)
        actor_features = self.actor(inputs)
        dist = self.get_dist(actor_features, action_mask)
        action_log_probs = torch.cat(dist.log_probs(action), dim=-1)
        dist_entropy = dist.entropy()
        dist_entropy = sum([d.mean() for d in dist_entropy])

        return (
            value,
            action_log_probs,
            dist_entropy,
        )

    def soft_update(self, t):
        source, target = self.critic, self.target_critic
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

    def update(self, batch_obs, batch_act, batch_rew, batch_done, step):

        with torch.no_grad():
            next_value = self.get_target_value(batch_obs[self.n_steps, :, :])


        returns = compute_returns(batch_rew, batch_done, next_value, self.gamma)

        values, action_log_probs, entropy = self.evaluate_actions(self.split_obs(batch_obs[:-1]),
                                                                  self.split_act(batch_act))

        returns = torch.stack(returns)[:-1]

        advantage = returns - values

        actor_loss = (
                -(action_log_probs * advantage.detach()).sum(dim=2).mean()
                - self.entropy_coef * entropy
        )
        value_loss = (returns - values).pow(2).sum(dim=2).mean()

        loss = actor_loss + self.value_loss_coef * value_loss

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        if self.target_update_interval_or_tau > 1.0 and step % self.target_update_interval_or_tau == 0:
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)

        return loss

