import torch
import torch.nn as nn
import copy


class CDQNAgent:
    def __init__(self, qnet: nn.Module, model_cfgs, env_cfgs):
        self.qnet = qnet
        self.qnet_target = copy.deepcopy(qnet)
        self.model_cfgs = model_cfgs
        self.env_cfgs = env_cfgs

    def act(self, obs):
        epsilon = self.model_cfgs['epsilon']

        if torch.rand(1).item() < epsilon:
            actions = self.env_cfgs['action_space'].sample()

        else:
            with torch.no_grad():
                q_values = self.qnet(obs)
                idx = int(q_values.argmax(dim=0))
                actions = self.env_cfgs['idx_action_dict'][idx]

        return actions

    def update(self, replay_buffer, optimizer):
        batch = replay_buffer.sample(self.model_cfgs['batch_size'])
        obs, actions, rewards, next_obs, dones = batch

        q_values = self.qnet(obs)
        indices = torch.tensor([self.env_cfgs["action_idx_dict"][tuple(x.tolist())] for x in actions]).unsqueeze(-1)
        q_values = torch.gather(q_values, 1, indices)

        with torch.no_grad():
            q_values_next = self.qnet_target(next_obs).max(dim=1).values
            target = rewards + self.model_cfgs['gamma'] * q_values_next * (1 - torch.all(dones, dim=1).int())

        loss = nn.functional.mse_loss(q_values.squeeze(-1), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
