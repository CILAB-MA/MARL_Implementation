from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv
import rware
from algos.ia2c.wrapper import make_env

def train(cfgs):
    train_cfgs = cfgs.train_cfgs
    print(train_cfgs)
    env_cfgs = cfgs.env_cfgs
    model_cfgs = cfgs.model_cfgs
    t = 0
    num_cpu = 2
    envs = SubprocVecEnv([make_env() for i in range(num_cpu)])
    obss = envs.reset()
    for _ in tqdm(range(train_cfgs['total_timesteps'])):
        pass
    print('Train completed!!')

