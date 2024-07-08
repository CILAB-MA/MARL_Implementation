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
          
        self.gamma = train_cfgs['gamma']
        self.device = model_cfgs['device']
    
        self.models = [QNet(self.model_cfgs, self.env_cfgs).to(self.device) for _ in range(self.env_cfgs['num_agent'])]
        self.optimizers = [optim.Adam(model.parameters(), lr=self.model_cfgs['lr']) for model in self.models]
        self.replay_buffer = ReplayBuffer(train_cfgs, model_cfgs, env_cfgs)
        self.epsilon_scheduler = EpsilonScheduler(
            start=train_cfgs['epsilon'], 
            end=train_cfgs['epsilon_min'], 
            decay_steps=train_cfgs['epsilon_decay_steps'], 
            interval=train_cfgs['epsilon_decay_interval']
        )
    def update_buffer(self, obss, actions, rewards, next_obss, dones) -> bool:
        self.replay_buffer.append(obss, actions, rewards, next_obss, dones)
        return self.replay_buffer.is_full()

    def update_target(self) -> None:
        for model in self.models:
            model.update_target_net()
            
    def update(self):
        minibatch = self.replay_buffer.sample()
        obss, actions, rewards, next_obss, dones = minibatch
        self.epsilon_scheduler.step()
        for idx, model in enumerate(self.models):
            obs = obss[:,:,idx,:]
            action = actions[:,:,idx:idx+1]
            next_obs = next_obss[:,:,idx,:]
            reward = rewards[:,:,idx]
            
            expected_q_vals = model.actor_net(obs).gather(2, action).squeeze(2)
            with torch.no_grad():
                target_q_vals = reward + self.gamma * model.target_net(next_obs).max(dim=2)[0] * (1-dones)
            loss = F.mse_loss(expected_q_vals, target_q_vals)
            self.optimizers[idx].zero_grad()
            loss.backward()
            self.optimizers[idx].step()
                
    def act(self, obss):
        actions = []
        epsilon = self.epsilon_scheduler.get_epsilon()
        for idx, model in enumerate(self.models):
            obs = obss[idx]
            if len(obs.shape) == 1:
                obs = obs.unsqueeze(0)
            num_process = obs.shape[0]
            if random.random() < epsilon:
                action = torch.randint(
                    high=int(self.env_cfgs['act_space']), 
                    size=(num_process,),  
                    device=obs.device
                )
            else:
                q_vals = model(obs)
                action = q_vals.argmax(dim=1)
            actions.append(action)
        actions = torch.stack(actions)
        actions = actions.t().detach().cpu().numpy()
        return actions

    def save_agent(self, path):
        agent_state = {
            'models_state_dict': [model.state_dict() for model in self.models],
            'optimizers_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
            'epsilon_scheduler_state': {
                'epsilon': self.epsilon_scheduler.epsilon,
                'step_count': self.epsilon_scheduler.step_count
            },
            'env_cfgs': self.env_cfgs,
            'model_cfgs': self.model_cfgs,
            'train_cfgs': self.train_cfgs
        }

        torch.save(agent_state, path)
        
        
        
class EpsilonScheduler:
    def __init__(self, start=0.4, end=0.1, decay_steps=10000, interval=100):
        self.epsilon = start
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.interval = interval
        self.step_count = 0

    def step(self):
        if self.step_count % self.interval == 0 and self.epsilon > self.end:
            reduction = (self.start - self.end) / self.decay_steps
            self.epsilon = max(self.epsilon - reduction * self.interval, self.end)
        self.step_count += 1

    def get_epsilon(self):
        return self.epsilon