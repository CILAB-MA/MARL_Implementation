import numpy as np
import torch as tr
import torch.nn.functional as F
from algos.ia2c.policies import ActorCriticPolicy
from gym.spaces import flatdim
@tr.jit.script
def compute_returns(rewards, done, next_value, gamma: float):
    returns = [next_value]
    for i in range(len(rewards) - 1, -1, -1):
        ret = rewards[i] + gamma * returns[0] * (1 - done[i, :].unsqueeze(1))
        returns.insert(0, ret)
    return returns

def _split_batch(splits):
    def thunk(batch):
        return tr.split(batch, splits, dim=-1)

    return thunk


class IA2CAgent:
    def __init__(self, model, env_cfgs, model_cfgs, train_cfgs):
        self.model = ActorCriticPolicy(model_cfgs)
        self.num_agent = env_cfgs['num_agent']
        self.num_process = train_cfgs['num_process']
        self.env_cfgs = env_cfgs
        self.model_cfgs = model_cfgs
        self.train_cfgs = train_cfgs
        self.device = train_cfgs['device']

        self.model = self.model.to(self.device)
        self.split_obs = _split_batch([flatdim(s) for s in model_cfgs['observation_space']])
        self.split_act = _split_batch(model_cfgs['num_agent'] * [1])

    def update(self, batch_obs, batch_act, batch_rew, batch_done, step):
        with tr.no_grad():
            next_value = self.model.get_value(self.split_obs(batch_obs[-1, :, :]))

        returns = compute_returns(batch_rew, batch_done, next_value, 0.99)
        values, action_log_probs, entropy = self.model.evaluate_actions(self.split_obs(batch_obs[:-1]),
                                                                  self.split_act(batch_act))

        returns = tr.stack(returns)[:-1]


        advantage = returns - values

        actor_loss = (
                -(action_log_probs * advantage.detach()).sum(dim=2).mean()
                - 0.01 * entropy
        )
        value_loss = (returns - values).pow(2).sum(dim=2).mean()

        loss = actor_loss + 0.5 * value_loss
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        return actor_loss.item(), value_loss.item()

    def act(self, obss, deterministic=False):
        return self.model.act(obss)