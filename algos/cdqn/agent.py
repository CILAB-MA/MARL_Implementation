import torch
import torch.nn as nn
import copy


class CDQNAgent:
    def __init__(self, qnet: nn.Module, env_cfgs):
        self.qnet = qnet
        self.qnet_target = copy.deepcopy(qnet)
        self.env_cfgs = env_cfgs

    def act(self, obs, num_process):
        epsilon = self.qnet.epsilon
        actions = []

        if torch.rand(1).item() < epsilon:
            for _ in range(num_process):
                action = self.env_cfgs['action_space'].sample()
                actions.append(action)

        else:
            with torch.no_grad():
                q_values = torch.stack(self.qnet(obs))
                actions = q_values.argmax(dim=-1).transpose(0, 1).tolist()

        return actions

    def update(self, replay_buffer, optimizer):
        batch = replay_buffer.sample(self.qnet.batch_size)
        obs, actions, rewards, next_obs, dones = batch

        obs = obs.transpose(0, 1)
        next_obs = next_obs.transpose(0, 1)

        q_values = torch.stack(self.qnet(obs))
        q_values = torch.gather(q_values, 2, actions.transpose(0,1).unsqueeze(-1).long()).squeeze(-1)

        with torch.no_grad():
            q_values_next = torch.stack(self.qnet_target(next_obs)).max(dim=-1).values
            target = rewards + self.qnet.gamma * q_values_next * (1 - dones.int())

        loss = nn.functional.mse_loss(q_values, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def decay_epsilon(self, episode, total_episode):
        if self.qnet.epsilon > 0.1:
            self.qnet.epsilon = 1 - 3 / total_episode * episode
        else:
            self.qnet.epsilon = 0.1
