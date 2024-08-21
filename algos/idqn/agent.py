import numpy as np
import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from algos.idqn.model import QNet
from algos.idqn.replay_buffer import ReplayBuffer


class IDQNAgent:
    def __init__(self, env_cfgs, model_cfgs, train_cfgs):
        self.env_cfgs = env_cfgs
        self.model_cfgs = model_cfgs
        self.train_cfgs = train_cfgs
          
        self.gamma = train_cfgs['gamma']
        self.device = model_cfgs['device']
        self.target_update_interval_or_tau = train_cfgs['target_update_interval']

        self.model = QNet(model_cfgs, env_cfgs).to(self.device)
        self.replay_buffer = ReplayBuffer(train_cfgs, model_cfgs, env_cfgs)
        self.epsilon_scheduler = EpsilonScheduler(
            start=train_cfgs['epsilon'], 
            end=train_cfgs['epsilon_min'], 
            decay_steps=train_cfgs['epsilon_decay_steps'], 
            interval=train_cfgs['epsilon_decay_interval']
        )
        self.history = {"loss":[0]}
        self.updates = 0
        
        optimizer = getattr(optim, 'Adam')
        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)
        self.optimizer_class = optimizer
        self.optimizer = optimizer(self.model.parameters(), lr=model_cfgs['lr'])
        
    def update_buffer(self, obss, actions, rewards, next_obss, dones) -> bool:
        self.replay_buffer.append(obss, actions, rewards, next_obss, dones)
        return self.replay_buffer.is_full()

    def update_from_target(self) -> None:
        if (
            self.target_update_interval_or_tau > 1.0
            and self.updates % self.target_update_interval_or_tau == 0
        ):
            # Hard update
            self.model.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            # Soft update
            self.model.soft_update(self.target_update_interval_or_tau)
        self.updates += 1
    
    def _compute_loss(self):
        minibatch = self.replay_buffer.sample()
        obss, actions, rewards, next_obss, dones = minibatch

        
        expected_q_vals = self.model.critic_net(obss).gather(3, actions).squeeze(3)
        with torch.no_grad():
            target_q_vals = self.model.target_net(next_obss).max(dim=3)[0] * (1-dones)
            target_q_vals = rewards + self.gamma * target_q_vals
        loss = F.mse_loss(expected_q_vals, target_q_vals)
        
        with torch.no_grad():
            self.history['loss'].append(loss.item())
        
        return loss
           
            
    def update(self) -> None:
        self.epsilon_scheduler.step()
        loss = self._compute_loss()

        self.optimizer.zero_grad()
        loss.backward()
    
        self.optimizer.step()
        
        self.update_from_target()
                
    def act(self, obss):
        epsilon = self.epsilon_scheduler.get_epsilon()

        obss = [obs for obs in obss]
        actions = self.model.act(obss)

        for idx in range(self.env_cfgs["num_agent"]):
            if random.random() < epsilon:
                num_process = obss[idx].shape[0]
                action = torch.randint(
                        high=int(self.env_cfgs['act_space'][0]), 
                        size=(num_process,),  
                        device=obss[idx].device
                )
                actions[idx] = action
        actions = torch.stack(actions)
        actions = actions.t().detach().cpu().numpy()
        return actions

    def save_agent(self, path):
        agent_state = {
            'models_state_dict': self.model.state_dict(),
            'epsilon_scheduler_state': {
                'epsilon': self.epsilon_scheduler.epsilon,
                'step_count': self.epsilon_scheduler.step_count
            },
            'env_cfgs': self.env_cfgs,
            'model_cfgs': self.model_cfgs,
            'train_cfgs': self.train_cfgs
        }

        torch.save(agent_state, path)
    
    def load_agent(self, path):
        """Loads the agent's models, optimizers, and epsilon scheduler state from the given path."""
        agent_state = torch.load(path)

        # 모델 상태 복원
        self.model.load_state_dict(agent_state['models_state_dict'])

        # Epsilon 스케줄러 상태 복원
        self.epsilon_scheduler.epsilon = agent_state['epsilon_scheduler_state']['epsilon']
        self.epsilon_scheduler.step_count = agent_state['epsilon_scheduler_state']['step_count']

        # 환경 및 모델, 훈련 설정을 확인 (필요하다면)
        self.env_cfgs = agent_state['env_cfgs']
        self.model_cfgs = agent_state['model_cfgs']
        self.train_cfgs = agent_state['train_cfgs']        
    
    def get_mean_loss(self):
        mean_loss = np.mean(self.history['loss'])
        self.history['loss'] = [0]
        return mean_loss
        
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