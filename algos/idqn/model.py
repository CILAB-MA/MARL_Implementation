import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from gym.spaces import flatdim

from utils.model_func import MultiAgentFCNetwork

from typing import List


class QNet(nn.Module):
    def __init__(self, model_cfg, env_cfg) -> None:
        super(QNet, self).__init__()
        
        self.critic_net = MultiAgentFCNetwork(env_cfg['obs_space'], model_cfg['hidden_dim'], env_cfg['act_space'], True)
        self.target_net = MultiAgentFCNetwork(env_cfg['obs_space'], model_cfg['hidden_dim'], env_cfg['act_space'], True)
 
            
        self.soft_update(1.0)
        self.to(model_cfg['device'])
        
        for param in self.target_net.parameters():
            param.requires_grad = False
        
               
        optimizer = getattr(optim, 'Adam')
        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer = optimizer(self.parameters(), lr=model_cfg['lr'])

    
    def act(self, obss) -> List[int]:
        q_vals = self.critic_net(obss)
        actions = [q.argmax(-1) for q in q_vals] 
        return actions
    
    
    
    def soft_update(self, tau) -> None:
        actor, target = self.critic_net, self.target_net
        for target_param, source_param in zip(target.parameters(), actor.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

    
    
    
class FCNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128) -> None:
        super(FCNet, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.nn(x)
    
# class QNetwork(nn.Module):
#     def __init__(
#         self,
#         obs_space,
#         action_space,
#         cfg,
#         layers,
#         use_orthogonal_init,
#         device,
#     ):
#         super().__init__()
#         hidden_size = list(layers)
#         optimizer = getattr(optim, cfg.optimizer)
#         lr = cfg.lr

#         self.action_space = action_space

#         self.n_agents = len(obs_space)
#         obs_shape = [flatdim(o) for o in obs_space]
#         action_shape = [flatdim(a) for a in action_space]

#         # MultiAgentFCNetwork is much faster that MultiAgentSepsNetwork

#         self.critic = MultiAgentFCNetwork(obs_shape, hidden_size, action_shape, use_orthogonal_init)
#         self.target = MultiAgentFCNetwork(obs_shape, hidden_size, action_shape, use_orthogonal_init)


#         self.soft_update(1.0)
#         self.to(device)

#         for param in self.target.parameters():
#             param.requires_grad = False

#         if type(optimizer) is str:
#             optimizer = getattr(optim, optimizer)
#         self.optimizer_class = optimizer

#         self.optimizer = optimizer(self.critic.parameters(), lr=lr)

#         self.gamma = cfg.gamma
#         self.grad_clip = cfg.grad_clip
#         self.device = device

#         self.updates = 0
#         self.target_update_interval_or_tau = cfg.target_update_interval_or_tau

#         self.standardize_returns = cfg.standardize_returns
#         self.ret_ms = RunningMeanStd(shape=(self.n_agents,))

#         print(self)

#     def forward(self, inputs):
#         raise NotImplemented("Forward not implemented. Use act or update instead!")

#     def act(self, inputs, epsilon):
#         if epsilon > random.random():
#             actions = self.action_space.sample()
#             return actions
#         with torch.no_grad():
#             inputs = [torch.from_numpy(i).to(self.device) for i in inputs]
#             actions = [x.argmax(-1).cpu().item() for x in self.critic(inputs)]
#         return actions
    
#     def _compute_loss(self, batch):
#         obs = [batch[f"obs{i}"] for i in range(self.n_agents)]
#         nobs = [batch[f"next_obs{i}"] for i in range(self.n_agents)]
#         action = torch.stack([batch[f"act{i}"].long() for i in range(self.n_agents)])
#         rewards = torch.stack(
#             [batch["rew"][:, i].view(-1, 1) for i in range(self.n_agents)]
#         )
#         dones = batch["done"].unsqueeze(0).repeat(self.n_agents, 1, 1)

#         with torch.no_grad():
#             q_tp1_values = torch.stack(self.critic(nobs))
#             q_next_states = torch.stack(self.target(nobs))
#         all_q_states = torch.stack(self.critic(obs))

#         a_prime = q_tp1_values.argmax(-1)
#         target_next_states = q_next_states.gather(-1, a_prime.unsqueeze(-1))

#         target_states = rewards + self.gamma * target_next_states * (1 - dones)

#         if self.standardize_returns:
#             self.ret_ms.update(target_states)
#             target_states = (
#                 target_states - self.ret_ms.mean.view(-1, 1, 1)
#             ) / torch.sqrt(self.ret_ms.var.view(-1, 1, 1))

#         q_states = all_q_states.gather(-1, action)
#         return torch.nn.functional.mse_loss(q_states, target_states)


#     def update(self, batch):
#         loss = self._compute_loss(batch)
#         self.optimizer.zero_grad()
#         loss.backward()
#         if self.grad_clip:
#             torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
#         self.optimizer.step()

#         self.update_from_target()

#     def update_from_target(self):
#         if (
#             self.target_update_interval_or_tau > 1.0
#             and self.updates % self.target_update_interval_or_tau == 0
#         ):
#             # Hard update
#             self.soft_update(1.0)
#         elif self.target_update_interval_or_tau < 1.0:
#             # Soft update
#             self.soft_update(self.target_update_interval_or_tau)
#         self.updates += 1

#     def soft_update(self, t):
#         source, target = self.critic, self.target
#         for target_param, source_param in zip(target.parameters(), source.parameters()):
#             target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)
