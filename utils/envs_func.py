from collections import deque
from time import perf_counter
from functools import partial
import numpy as np
import torch
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


class TorchRwareWrapper(VecEnvWrapper):
    def reset(self):
        obs = self.venv.reset()
        obs = np.array(obs)
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        return obs

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = np.array(obs)
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        done = torch.tensor(done, dtype=torch.bool, device=DEVICE)

        return obs, reward, done, info


class CentralRwareWrapper(TorchRwareWrapper):
    def __init__(self, envs):
        super().__init__(envs)

        self.joint_obs_dim = envs.observation_space[0].shape[0] * len(envs.observation_space)
        self.action_dim = envs.action_space[0].n
        self.num_agent = len(envs.observation_space)

    def reset(self):
        obs = super().reset()

        obs = obs.repeat(1, 1, self.num_agent)
        return obs

    def step_wait(self):
        obs, reward, done, info = super().step_wait()

        obs = obs.repeat(1, 1, self.num_agent)
        reward = reward.sum(axis=1)

        return obs, reward, done, info


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
