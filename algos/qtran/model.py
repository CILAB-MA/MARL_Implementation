from torch import optim
import torch.nn as nn
import torch as tr

from utils.model_func import MultiAgentFCNetwork

class QTRANNet(nn.Module):
    def __init__(self, model_cfg, env_cfg) -> None:
        # todo: check dimensions for each network
        super(QTRANNet, self).__init__()
        # inidividual q (indiv obss embedding + action embedding)
        self.q = MultiAgentFCNetwork(env_cfg['obs_space'], model_cfg['hidden_dim'], env_cfg['act_space'], True)
        self.q_target = MultiAgentFCNetwork(env_cfg['obs_space'], model_cfg['hidden_dim'], env_cfg['act_space'], True)

        # joint q (whole state + indiv q embedding)
        self.joint_q = MultiAgentFCNetwork(env_cfg['obs_space'], model_cfg['hidden_dim'], env_cfg['act_space'], True)
        self.joint_q_target = MultiAgentFCNetwork(env_cfg['obs_space'], model_cfg['hidden_dim'], env_cfg['act_space'], True)

        # v (whole state)
        self.v = MultiAgentFCNetwork(env_cfg['obs_space'], model_cfg['hidden_dim'], env_cfg['act_space'], True)
        self.v_target = MultiAgentFCNetwork(env_cfg['obs_space'], model_cfg['hidden_dim'], env_cfg['act_space'], True)

        self.soft_update(1.0)
        self.to(model_cfg['device'])
        self.hidden_dim = q_output_dim + env_cfg['num_action']
        self.ae_embedding = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim))

        for param in self.target_net.parameters():
            param.requires_grad = False

        optimizer = getattr(optim, 'Adam')
        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer = optimizer(self.parameters(), lr=model_cfg['lr'])

    def forward(self, obss, actions=None):
        state = tr.cat(obss, dim=1) # todo: check dim is right
        indiv_qs = self.q(obss)
        inidiv_q_a = tr.cat([indiv_qs, actions], dim=-1) # todo: check action input
        indiv_embedding = self.ae_embedding(inidiv_q_a.reshape(-1, self.hidden_dim ))
        indiv_embedding = indiv_embedding.reshape(-1, self.num_agent, self.hidden_dim)
        indiv_embedding = indiv_embedding.sum(dim=1)

        joint_input = tr.cat([state, indiv_embedding], dim=-1)
        joint_q_ouptut = self.joint_q(joint_input)

        value_output = self.v(state)

        return joint_q_ouptut, value_output

    def act(self, obss):
        qs = self.q(obss)
        actions = [q.argmax(-1) for q in qs]
        return actions


