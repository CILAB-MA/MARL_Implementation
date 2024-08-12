from gym.spaces import flatdim
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
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
        self.obs_shape = [flatdim(o) for o in obs_space]  # [142, 142]
        self.action_shape = [flatdim(a) for a in action_space]  # [5, 5]

        self.actor = model_func.MultiAgentFCNetwork(self.obs_shape, [64, 64], self.action_shape, True)
        self.critic = model_func.CriticFCNetwork(self.n_agents, self.obs_shape[0], self.action_shape[0])
        self.target_critic = model_func.CriticFCNetwork(self.n_agents, self.obs_shape[0], self.action_shape[0])

        self.device = device

        self.soft_update(1.0)
        self.to(device)

        # TODO need to fix (optimizer)
        optimizer = getattr(optim, cfg.model_cfgs['optimizer'])
        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer_class = optimizer

        lr = cfg.model_cfgs['lr']
        self.optimizer = optimizer(self.parameters(), lr=lr)

        self.critic_optimizer = optimizer(self.parameters(), lr=lr)
        self.agent_optimizer = optimizer(self.parameters(), lr=lr)

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

    def soft_update(self, t):
        source, target = self.critic, self.target_critic
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

    def update(self, batch_obs, batch_act, batch_rew, batch_done, step):

        batch_critic_input = self.build_input(batch_obs, batch_act)

        '''train_critic'''
        with torch.no_grad():
            next_value = self.target_critic(batch_critic_input[self.n_steps-1])

        index_act = batch_act[self.n_steps - 1].long().unsqueeze(-1)  # (8, 2, 1)
        next_value_taken = torch.gather(next_value, dim=-1, index=index_act).squeeze(-1)  # (8, 2, 5)

        returns = compute_returns(batch_rew, batch_done, next_value_taken, self.gamma)  # (11, 8, 2)
        returns = torch.stack(returns)[:-1]  # (10, 8, 2)

        index_acts = batch_act.long().unsqueeze(-1)  # 10, 8, 2, 1
        values = self.critic(batch_critic_input)  # 10, 8, 2, 5
        values_taken = torch.gather(values, dim=-1, index=index_acts).squeeze(-1)
        critic_loss = (returns - values_taken).pow(2).sum(dim=2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''optimize agents(actor)'''
        # values, action_log_probs, entropy = self.evaluate_actions(self.split_obs(batch_obs[:-1]),
        #                                                           self.split_act(batch_act))
        #
        # advantage = returns - values
        #
        # actor_loss = (
        #         -(action_log_probs * advantage.detach()).sum(dim=2).mean()
        #         - self.entropy_coef * entropy
        # )
        #
        # loss = actor_loss + self.value_loss_coef * critic_loss

        if self.target_update_interval_or_tau > 1.0 and step % self.target_update_interval_or_tau == 0:
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)

        return critic_loss

    def build_input(self, batch_obs, batch_act):
        num_step, batch, _ = batch_act.shape

        batch_obs = batch_obs.unsqueeze(2).repeat(1, 1, self.n_agents, 1)  # (11, 8, 142) -> (11, 8, 2, 142)

        # joint action (8, 2, 5)
        batch_onehot_act = F.one_hot(batch_act.long(), num_classes=self.action_shape[0])  # joint action (10, 8, 2, 5)

        # masked other action (8, 2, 10(me, other))
        agent_mask = (1 - torch.eye(self.n_agents, device=self.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.action_shape[0]).view(self.n_agents, -1)

        agent_mask = agent_mask.unsqueeze(0).unsqueeze(0).expand(num_step, batch, -1, -1)  # 10, 8, 2, 10

        batch_onehot_act_reshape = batch_onehot_act.view(num_step, batch, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1)

        batch_other_act = batch_onehot_act_reshape * agent_mask

        # index (8, 2, 2)
        batch_index = torch.eye(self.n_agents, device=self.device).unsqueeze(0).unsqueeze(0).expand(num_step, batch, -1, -1)

        inputs = torch.cat([batch_obs[: num_step], batch_onehot_act, batch_other_act, batch_index], dim=-1)

        return inputs
