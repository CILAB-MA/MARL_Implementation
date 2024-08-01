import torch
import torch.nn as nn
import copy


class CDQNAgent:
    def __init__(self, qnet: nn.Module, env_cfgs):
        self.qnet = qnet
        self.qnet_target = copy.deepcopy(qnet)
        self.optimizer = torch.optim.Adam(qnet.parameters(), lr=qnet.lr)
        self.env_cfgs = env_cfgs

    def act(self, obs):
        actions = []
        for i in range(self.env_cfgs['num_process']):
            if torch.rand(1).item() < self.qnet.epsilon:
                actions.append(self.env_cfgs['action_space'].sample())

            else:
                with torch.no_grad():
                    q_values = self.qnet(obs[:, i, :])
                    actions.append(q_values.argmax(dim=-1).tolist())

        return actions

    def update(self, replay_buffer):
        batch = replay_buffer.sample(self.qnet.batch_size)
        obs, actions, rewards, next_obs, dones = batch

        q_values = self.qnet(obs)
        q_values = torch.gather(q_values, 3, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            q_values_next = self.qnet_target(next_obs).max(dim=-1).values
            target = rewards + self.qnet.gamma * q_values_next * (1 - dones.int())

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self, episode, total_episode, target_update_freq):
        if self.qnet.epsilon > 0.05:
            self.qnet.epsilon -= 3 / total_episode
        else:
            self.qnet.epsilon = 0.05

        # Update target network
        if episode % target_update_freq == 0:
            self.target_update()

    def target_update(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())
