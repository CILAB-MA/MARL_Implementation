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