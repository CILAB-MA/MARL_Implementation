import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class QMIXNet(nn.Module):
    def __init__(self, model_cfg, env_cfg) -> None:
        super(QMIXNet, self).__init__()

        self.embed_dim = model_cfg["embed_dim"]
        self.n_agents = env_cfg["num_agent"]
        self.hyper_obs = sum(env_cfg["obs_space"])
        
        if model_cfg["hypernet_layers"] == 1:
            self.hyper_w_1 = nn.Linear(self.hyper_obs, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.hyper_obs, self.embed_dim)
        elif model_cfg["hypernet_layers"] == 2:
            hypernet_embed = model_cfg["hypernet_embed"]
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.hyper_obs, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents)
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.hyper_obs, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim)
            )
        
        # State dependent bias for hidden layer    
        self.hyper_b_1 = nn.Linear(self.hyper_obs, self.embed_dim)
        
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.hyper_obs, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
        
    def forward(self, agent_q_vals, states): #(N,B,E), (B,E,hyper_obs)
        agent_q_vals = rearrange(agent_q_vals, "N B E -> B E N")#(B,N,E), (B,E,hyper_obs)
        batch_size = agent_q_vals.size(0)
        states = states.reshape(-1, self.hyper_obs) #(B,N,E), (B*E,hyper_obs)
        agent_q_vals = agent_q_vals.view(-1, 1, self.n_agents) #(B*N,1,E), (B*E,hyper_obs)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states)) 
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_q_vals, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        central_q_val = y.view(batch_size, -1)
        return central_q_val
