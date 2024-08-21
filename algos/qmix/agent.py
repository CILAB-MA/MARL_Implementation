import numpy as np

import torch
import torch.nn.functional as F

from algos.qmix.model import QMIXNet

from algos.idqn.agent import EpsilonScheduler, IDQNAgent


class QMIXAgent(IDQNAgent):
    def __init__(self, env_cfgs, model_cfgs, train_cfgs) -> None:
        # IDQN을 상속 받아서 사용함(QNetworks는 IDQN 것을 그대로 사용)
        super().__init__(env_cfgs, model_cfgs, train_cfgs)
        
        self.hypernet_layers = self.model_cfgs['hypernet_layers']
        
        self.mixer = QMIXNet(model_cfgs, env_cfgs).to(self.device)
        self.target_mixer = QMIXNet(model_cfgs, env_cfgs).to(self.device)
        self.soft_update(1.0)
        
        self.history = {"loss":[0]}
        self.updates = 0
        
        self.is_central_reward = env_cfgs['central_reward']
        
        self.optimizer = self.optimizer_class(
            list(self.model.parameters()) + list(self.mixer.parameters()), lr=model_cfgs['lr'],
        )
        
    def _compute_loss(self):
        minibatch = self.replay_buffer.sample()
        obss, actions, rewards, next_obss, dones = minibatch
        if self.is_central_reward:
            rewards = rewards[0]
        else:
            rewards = rewards.sum(dim=0)
        # obs = [batch[f"obs{i}"] for i in range(self.n_agents)]
        # nobs = [batch[f"next_obs{i}"] for i in range(self.n_agents)]
        # action = torch.stack([batch[f"act{i}"].long() for i in range(self.n_agents)])
        # rewards = batch["rew"]
        # dones = batch["done"]

        with torch.no_grad():
            # q_tp1_values = torch.stack(self.critic(nobs))
            # q_next_states = torch.stack(self.target(nobs))
        
            indvidual_target_q_vals = self.model.target_net(next_obss).max(dim=3)[0] 
        #all_q_states = torch.stack(self.critic(obs))
        central_target_q_vals = self.target_mixer(
            indvidual_target_q_vals, torch.concat(next_obss, dim=-1)
        ).detach()
        target_q_vals = rewards + self.gamma * central_target_q_vals * (1 - dones)
        
        
        
        # _, a_prime = q_tp1_values.max(-1)

        # target_next_states = self.target_mixer(
        #     q_next_states.gather(2, a_prime.unsqueeze(-1)), torch.concat(nobs, dim=-1)
        # ).detach()

        # target_states = rewards + self.gamma * target_next_states * (1 - dones)

        #Todo: implement standardize_returns
        # if self.standardize_returns:
        #     self.ret_ms.update(target_states)
        #     target_states = (target_states - self.ret_ms.mean.view(-1, 1)) / torch.sqrt(
        #         self.ret_ms.var.view(-1, 1)
        #     )
        indvidual_expected_q_vals = self.model.critic_net(obss).gather(3, actions).squeeze(3)
        expected_q_vals = self.mixer(indvidual_expected_q_vals, torch.concat(obss, dim=-1))
        
        loss = F.mse_loss(expected_q_vals, target_q_vals)
        
        with torch.no_grad():
            self.history['loss'].append(loss.item())
        
        return loss
    
    def soft_update(self, t):
        self.model.soft_update(t)
        try:
            source, target = self.mixer, self.target_mixer
        except AttributeError: # fix for when qmix has not initialised a mixer yet
            return
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)