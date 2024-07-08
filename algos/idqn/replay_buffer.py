
import collections
import random
import torch
from typing import Tuple
import numpy as np

class ReplayBuffer:
    """
    Store (S, A, R, S', D) 
    """
    def __init__(self, train_cfgs, model_cfgs, env_cfgs) -> None:
        self.obs_space = env_cfgs['obs_space']
        self.act_space = env_cfgs['act_space']
        self.buffer_size = int(train_cfgs['replay_buffer_size']/train_cfgs['num_process'])
        self.num_envs = train_cfgs['num_process']
        self.num_agents = env_cfgs['num_agent']
        self.device = model_cfgs['device']
        self.batch_size = train_cfgs['batch_size']
        self.reset()
    
    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.num_envs ,self.num_agents, self.obs_space),dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_envs, self.num_agents),dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, self.num_envs, self.num_agents),dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.num_envs ,self.num_agents, self.obs_space),dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_envs),dtype=np.float32)
        self.pos = 0
        self.full = False
    
    def append(self, obss, actions, rewards, next_obss, dones) -> None:
        obss = np.array(obss).transpose(1,0,2)
        next_obss = np.array(next_obss).transpose(1,0,2)
        
        self.observations[self.pos] = obss
        self.actions[self.pos] = actions.copy()
        self.rewards[self.pos] = rewards.copy()
        self.next_observations[self.pos] = next_obss
        self.dones[self.pos] = dones.copy()
        
        self.pos += 1
        if self.pos == self.buffer_size -1:
            self.full = True
            self.pos = 0
    
    def sample(self) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        max_idx = self.buffer_size if self.full else self.pos
        sample_idx = random.sample(range(max_idx), self.batch_size)
        
        s = self.observations[sample_idx]
        a = self.actions[sample_idx]
        r = self.rewards[sample_idx]
        s_prime = self.next_observations[sample_idx]
        done_mask = self.dones[sample_idx]     
        
        s_tensor = torch.tensor(s, dtype=torch.float, device=self.device)
        a_tensor = torch.tensor(a, dtype=torch.int64, device=self.device)
        r_tensor = torch.tensor(r, dtype=torch.float, device=self.device)
        s_prime_tensor = torch.tensor(s_prime, dtype=torch.float, device=self.device)
        done_mask_tensor = torch.tensor(done_mask, dtype=torch.float, device=self.device)
        
        return s_tensor, a_tensor, r_tensor, s_prime_tensor, done_mask_tensor
        
    
    def is_full(self) -> bool:
        return self.full