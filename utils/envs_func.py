from functools import partial
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

class RwareMonitor(VecEnvWrapper):
    def __init__(
            self,
            venv,
            filename: Optional[str] = None,
            num_agent: int = 2,
            info_keywords: Tuple[str, ...] = (),
    ):
        from stable_baselines3.common.monitor import Monitor, ResultsWriter

        VecEnvWrapper.__init__(self, venv)
        self.episode_count = 0
        self.t_start = time.time()

        env_id = None
        if hasattr(venv, "spec") and venv.spec is not None:
            env_id = venv.spec.id

        self.results_writer: Optional[ResultsWriter] = None
        if filename:
            self.results_writer = ResultsWriter(
                filename, header={"t_start": self.t_start, "env_id": str(env_id)}, extra_keys=info_keywords
            )
        self.num_agent = num_agent
        self.info_keywords = info_keywords
        self.episode_returns = np.zeros((self.num_envs, self.num_agent), dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        obs = self.venv.reset()
        self.episode_returns = np.zeros((self.num_envs, self.num_agent), dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_returns += rewards
        self.episode_lengths += 1
        new_infos = list(infos[:])
        for i in range(len(dones)):
            if dones[i]:
                info = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {"r": episode_return, "l": episode_length, "t": round(time.time() - self.t_start, 6)}
                for key in self.info_keywords:
                    episode_info[key] = info[key]
                info["episode"] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                if self.results_writer:
                    self.results_writer.write_row(episode_info)
                new_infos[i] = info
        return obs, rewards, dones, new_infos

    def close(self) -> None:
        if self.results_writer:
            self.results_writer.close()
        return self.venv.close()
