
import collections
import random
import torch
from typing import Tuple

class ReplayBuffer:
    """
    Store (S, A, R, S', D) 
    """
    def __init__(self, buffer_size) -> None:
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=buffer_size)
    
    def append(self, transition) -> None:
        self.buffer.append(transition)
    
    def sample(self, batch_size) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        mini_batch = random.sample(self.buffer, batch_size)
        
        s, a, r, s_prime, done_mask = zip(*mini_batch)
        
        s_tensor = torch.tensor(s, dtype=torch.float)
        a_tensor = torch.tensor(a, dtype=torch.float)
        r_tensor = torch.tensor(r, dtype=torch.float)
        s_prime_tensor = torch.tensor(s_prime, dtype=torch.float)
        done_mask_tensor = torch.tensor(done_mask, dtype=torch.float)
        
        return s_tensor, a_tensor, r_tensor, s_prime_tensor, done_mask_tensor
        
    
    def __len__(self) -> int:
        return len(self.buffer)