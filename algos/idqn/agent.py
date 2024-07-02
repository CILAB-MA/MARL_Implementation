import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from algos.idqn.model import QNet
from algos.idqn.replay_buffer import ReplayBuffer


class IDQNAgent:
    def __init__(self, model, env_cfgs, model_cfgs, train_cfgs):
        self.env_cfgs = env_cfgs
        self.model_cfgs = model_cfgs
        self.train_cfgs = train_cfgs
        
        
        self.epsilon = train_cfgs['epsilon_start']
        self.gamma = train_cfgs['gamma']
        
    
        self.models = [QNet(self.model_cfgs) for _ in range(self.env_cfgs['num_agent'])]
        self.replay_buffers = [ReplayBuffer(train_cfgs['replay_buffer_size']) for _ in range(self.env_cfgs['num_agent'])]
        self.optimizers = [optim.Adam(model.parameters(), lr=self.model_cfgs['lr']) for model in self.models]
    
    def update_buffer(self, obss, actions, rewards, next_obss, dones):
        for idx, replay_buffer in enumerate(self.replay_buffers):
            replay_buffer.append(obss[idx], actions[idx], rewards[idx], next_obss[idx], dones[idx])

    def update_target(self) -> None:
        for model in self.models:
            model.update_target_net()
            
    def update(self):
        for idx, model in enumerate(self.models):
            minibatch = self.replay_buffers[idx].sample()
            obss, actions, rewards, next_obss, dones = minibatch
            
            expected_q_vals = model.actor_net(obss).gather(1, actions)
            target_q_vals = rewards + self.gamma * model.target_net(next_obss).max(dim=1)[0] * (1 - dones)
            loss = F.mse_loss(expected_q_vals, target_q_vals)
            self.optimizers[idx].zero_grad()
            loss.backward()
            self.optimizers[idx].step()
                
    def act(self, obss):
        actions = []
        for idx, model in enumerate(self.models):
            obs = obss[idx]
            if random.random() < self.epsilon:
                action = torch.tensor([random.randrange(self.model_cfgs['act_space'])], device=obs.device)
            else:
                q_vals = model(obs)
                action = q_vals.argmax()
            actions.append(action)
        return torch.stack(actions)
    
    def replay_buffer_size(self):
        return len(self.replay_buffers[0])  