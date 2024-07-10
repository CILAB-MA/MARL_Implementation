import numpy as np
import torch as tr
import torch.nn.functional as F
from algos.ia2c.policies import ActorCriticPolicy
class IA2CAgent:
    def __init__(self, model, env_cfgs, model_cfgs, train_cfgs):
        self.model = ActorCriticPolicy(model_cfgs)
        self.num_agent = env_cfgs['num_agent']
        self.num_process = train_cfgs['num_process']
        self.num_action = model_cfgs['action_space'][0].n
        self.env_cfgs = env_cfgs
        self.model_cfgs = model_cfgs
        self.train_cfgs = train_cfgs
        self.device = train_cfgs['device']

    def update(self, storage):
        data = storage.get()
        obss, acts, advs, cri_tar = data
        acts = acts.long()
        vals, log_probs = self.evaluate_action(obss, acts)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        policy_loss = -(advs * log_probs).mean()

        value_loss = F.mse_loss(vals, cri_tar)
        loss = policy_loss + value_loss
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        return value_loss.item(), policy_loss.item()

    def act(self, obss, deterministic=False):
        with tr.no_grad():
            obss = tr.cat([tr.from_numpy(o).unsqueeze(1) for o in obss], dim=1).transpose(0, 1)
            action, log_probs = self.model.act(obss, deterministic=deterministic)
            action = tr.cat(action, dim=-1)
            log_probs = tr.cat(log_probs, dim=-1)
            action = action.numpy()
        return action, log_probs

    def get_value(self, obss):
        with tr.no_grad():
            obss = tr.cat([tr.from_numpy(o).unsqueeze(1) for o in obss], dim=1).transpose(0, 1)
            value = self.model.get_value(obss)
        return value

    def evaluate_action(self, obss, acts):
        obss = obss.transpose(0, 1)
        acts = acts.transpose(0, 1)
        value = self.model.get_value(obss)
        log_prob = self.model.get_log_prob(obss, acts)
        log_prob = tr.cat(log_prob, dim=-1)
        return value, log_prob