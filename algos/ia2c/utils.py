
import torch as tr
import numpy as np
import copy

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
        self.rewards = np.zeros((self.buffer_size, self.num_process, self.num_agent), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.num_process), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.num_process, self.num_agent), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.num_process, self.num_agent), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.num_process, self.num_agent), dtype=np.float32)
        self.critic_target = np.zeros((self.buffer_size, self.num_process, self.num_agent), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obss, actions, rewards, dones, values, log_probs):
        obss_tmp = np.array([obs.copy() for obs in obss])
        obss = obss_tmp.transpose(1, 0, 2)
        self.observations[self.pos] = obss
        self.actions[self.pos] = actions.copy()
        self.rewards[self.pos] = np.array(rewards).copy()
        self.dones[self.pos] = np.array(dones).copy()
        self.values[self.pos] = values.clone().cpu().numpy()
        self.log_probs[self.pos] = log_probs.clone().cpu().numpy()

        self.pos += 1
        if self.pos == self.buffer_size - 1:
            self.full =True

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
            next_non_terminal = np.expand_dims(next_non_terminal, axis=-1)
            next_non_terminal = np.tile(next_non_terminal, (1, 2))
            self.advantages[step] = self.rewards[step] + self.gamma * (1 - next_non_terminal) * next_values - self.values[step]
            self.critic_target[step] = self.rewards[step] + self.gamma * (1 - next_non_terminal) * next_values

    def swap_and_flatten(self, array):
        shape = array.shape
        return array.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def get(self):
        indices = np.random.permutation(self.buffer_size * self.num_process)
        _tensor_names = [
            "observations",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "critic_target",
        ]

        for tensor in _tensor_names:
            self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.__dict__[tensor] = tr.from_numpy(self.__dict__[tensor])#.to(self.device)
        data = (
            self.observations[indices],
            self.actions[indices],
            self.advantages[indices],
            self.critic_target[indices]
        )
        return copy.deepcopy(data)

def print_square(dictionary):
    for key in dictionary.keys():
        if "float" in str(type(dictionary[key])):
            newval = round(float(dictionary[key]), 6)
            dictionary[key] = newval
        elif "list" in str(type(dictionary[key])):
            dictionary[key] = str(dictionary[key])
    front_lens = []
    back_lens = []
    for key in dictionary.keys():
        front_lens.append(len(key))
        back_lens.append(len(str(dictionary[key])))
    front_len = max(front_lens)
    back_len = max(back_lens)

    strings = []
    for key in dictionary.keys():
        string = "| {0:<{2}} | {1:<{3}} |".format(key, dictionary[key], front_len, back_len)
        strings.append(string)

    max_len = max([len(i) for i in strings])
    print("-"*max_len)
    for string in strings:
        print(string)
    print("-" * max_len)