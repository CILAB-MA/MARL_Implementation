
import torch as tr
import numpy as np

class RolloutBuffer(object):

    def __init__(self, num_agent, num_obss, buffer_size, num_process, device, gamma=0.99):
        self.num_agent = num_agent
        self.num_obss = num_obss
        self.device = device
        self.num_process = num_process
        self.gamma = gamma
        self.buffer_size = buffer_size

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.num_process, self.num_agent, self.num_obss), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_process, self.num_agent), dtype=np.int16)
        self.rewards = np.zeros((self.buffer_size, self.num_process, self.num_agent), dtype=np.float16)
        self.dones = np.zeros((self.buffer_size, self.num_process, self.num_agent), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.num_process, self.num_agent), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.num_process, self.num_agent), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.num_process, self.num_agent, self.num_action), dtype=np.float32)
        self.critic_target = np.zeros((self.buffer_size, self.num_process, self.num_agent), dtype=np.float32)
        self.pos = 0

    def add(self, obss, actions, rewards, dones, values, log_probs):
        self.observations[self.pos] = obss.copy()
        self.actions[self.pos] = actions.clone().cpu().numpy() # todo: torch로 넘어오는지 체크
        self.rewards[self.pos] = np.array(rewards).copy()
        self.dones[self.pos] = np.array(dones).copy()
        self.values[self.pos] = values.clone().cpu().numpy()
        self.log_probs[self.pos] = log_probs.clone().cpu().numpy()

        self.pos += 1

    def compute_advantage(self, next_values, dones):
        '''
        rollout step이 꽉차면 돈다
        '''
        last_values = next_values.clone().cpu().numpy()

        for step in reversed(range(self.buffer_size)):

            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1 - self.dones[step + 1]
                next_values = self.values[step + 1]
            self.advantage[step] = self.rewards[step] + self.gamma * (1 - next_non_terminal) * next_values - self.values[step]
            self.critic_target[step] = self.rewards[step] + self.gamma * (1 - next_non_terminal) * next_values

