from functools import partial
import gym
import rware
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
import os


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
