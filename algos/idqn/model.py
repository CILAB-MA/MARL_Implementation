import torch.nn as nn
import torch.nn.functional as F
from utils.model_func import MultiAgentFCNetwork


class QNet(nn.Module):
    def __init__(self, model_cfg) -> None:
        super(QNet, self).__init__()
        
        
        self.actor_net = FCNet(model_cfg['obs_space'], model_cfg['act_space'], model_cfg['hidden_dim'])
        self.target_net = FCNet(model_cfg['obs_space'], model_cfg['act_space'], model_cfg['hidden_dim'])
        self.update_target_net()
    
    def forward(self, obs):
        q_val = self.actor_net(obs)
        return q_val
    
    def update_target_net(self) -> None:
        self.target_net.load_state_dict(self.actor_net.state_dict())
    
    
    
class FCNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128) -> None:
        super(FCNet, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.nn(x)