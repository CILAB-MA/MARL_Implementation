import numpy as np
import torch as tr
import torch.nn.functional as F
from algos.ia2c.policies import ActorCriticPolicy
class IA2CAgent:
    def __init__(self, model, env_cfgs, model_cfgs, train_cfgs):
        self.models = [ActorCriticPolicy(model_cfgs) for _ in range(env_cfgs['num_agent'])]
        self.num_agent = env_cfgs['num_agent']
        self.num_process = train_cfgs['num_process']
        self.num_action = model_cfgs['num_action']
        self.env_cfgs = env_cfgs
        self.model_cfgs = model_cfgs
        self.train_cfgs = train_cfgs
        self.device = train_cfgs['device']

    def update(self, storage):
        data = storage.get()
        obss, acts, advs, cri_tar = data
        acts = acts.long()
        vals_0, log_probs_0 = self.evaluate_action(obss[:, 0], acts[:, 0], 0)
        vals_1, log_probs_1 = self.evaluate_action(obss[:, 1], acts[:, 1], 1)

        advs_0 = (advs[:, 0] - advs[:, 0].mean()) / (advs[:, 0].std() + 1e-8)
        advs_1 = (advs[:, 1] - advs[:, 1].mean()) / (advs[:, 1].std() + 1e-8)

        policy_loss_0 = -(advs_0 * log_probs_0).mean()

        value_loss_0 = F.mse_loss(vals_0, cri_tar[:, 0])

        policy_loss_1 = -(advs_1 * log_probs_1).mean()
        value_loss_1 = F.mse_loss(vals_1, cri_tar[:, 1])
        self.models[0].critic_optimizer.zero_grad()
        value_loss_0.backward()
        self.models[0].critic_optimizer.step()

        self.models[0].policy_optimizer.zero_grad()
        policy_loss_0.backward()
        self.models[0].policy_optimizer.step()

        self.models[1].critic_optimizer.zero_grad()
        value_loss_1.backward()
        self.models[1].critic_optimizer.step()

        self.models[1].policy_optimizer.zero_grad()
        policy_loss_1.backward()
        self.models[1].policy_optimizer.step()

    def act(self, obss, deterministic=False):
        with tr.no_grad():
            total_actions = tr.zeros(self.num_process, self.num_agent)
            total_log_probs = tr.zeros(self.num_process, self.num_agent)
            for i in range(self.num_agent):
                obs = tr.from_numpy(obss[i])
                action, log_prob = self.models[i].act(obs, deterministic=deterministic)
                print(action, log_prob)
                total_actions[:, i] = action
                total_log_probs[:, i] = log_prob
        total_actions = total_actions.numpy()
        return total_actions, total_log_probs

    def get_value(self, obss):
        total_values = tr.zeros(self.num_process, self.num_agent)
        with tr.no_grad():
            for i in range(self.num_agent):
                obs = tr.from_numpy(obss[i])
                value = self.models[i].get_value(obs)
                total_values[:, i] = value.squeeze(-1)
        return total_values

    def evaluate_action(self, obss, acts, i):

        value = self.models[i].get_value(obss)
        log_prob = self.models[i].get_log_prob(obss, acts)

        return value.squeeze(-1), log_prob