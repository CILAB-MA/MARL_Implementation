import gym
from typing import Callable

def make_env() -> Callable:

    def _init() -> gym.Env:
        env = gym.make("rware-tiny-2ag-v1")
        return env
    return _init