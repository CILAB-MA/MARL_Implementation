from collections import deque
from time import perf_counter
from functools import partial
import numpy as np
import gym
import rware
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
import os
from typing import Optional, Tuple
import numpy as np
import time

class VecRware(SubprocVecEnv):
    def __init__(self, num_envs, env_name):
        seed_list = [int.from_bytes(os.urandom(4), byteorder="little") for _ in range(num_envs)]
        env_fn = [partial(self.make_env_fn, env_name, seed_list[i]) for i in range(num_envs)]
        super().__init__(env_fn)

    @staticmethod
    def make_env_fn(name, seed):
        env = gym.make(name)
        env.seed(seed)
        env = RecordEpisodeStatistics(env)
        env = SquashDones(env)
        return env


class RwareWrapper(VecEnvWrapper):

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        return obs, rew, done, info



class SquashDones(gym.Wrapper):
    r"""Wrapper that squashes multiple dones to a single one using all(dones)"""

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, all(done), info


class RecordEpisodeStatistics(gym.Wrapper):
    """Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.episode_reward = 0
        self.episode_length = 0
        self.t0 = perf_counter()

        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.episode_reward += np.array(reward, dtype=np.float32)
        self.episode_length += 1
        if all(done):
            info["episode_returns"] = self.episode_reward
            if len(self.episode_reward) == self.n_agents:
                for i, agent_reward in enumerate(self.episode_reward):
                    info[f"agent{i}/episode_returns"] = agent_reward
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0

            self.reward_queue.append(self.episode_reward)
            self.length_queue.append(self.episode_length)
        return observation, reward, done, info