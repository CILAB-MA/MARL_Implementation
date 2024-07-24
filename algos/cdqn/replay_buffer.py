import torch
import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReplayBuffer:
    def __init__(self, size):
        self._storage = []
        self.maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs, action, reward, next_obs, done):
        data = (obs, action, reward, next_obs, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self.maxsize

    def _encode_sample(self, idxes):
        obss, actions, rewards, obss_next, dones = [], [], [], [], []

        for i in idxes:
            data = self._storage[i]
            obs, action, reward, obs_next, done = data
            obss.append(obs.clone().detach().float())
            actions.append(torch.tensor(action, dtype=torch.long, device=DEVICE).transpose(0, 1))
            rewards.append(reward.clone().detach().float())
            obss_next.append(obs_next.clone().detach().float())
            dones.append(done.clone().detach().float())
        return (torch.stack(obss).transpose(0, 1),
                torch.stack(actions).transpose(0, 1),
                torch.stack(rewards),
                torch.stack(obss_next).transpose(0, 1),
                torch.stack(dones))

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def full(self):
        return len(self._storage) == self.maxsize
