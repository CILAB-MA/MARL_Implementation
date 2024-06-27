from tqdm import tqdm
from collections import deque
from collections import defaultdict
from gym.spaces import flatdim
import torch
from utils import envs_func
from algos.ca2c.model import ActorCritic
import os

def train(cfgs):

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    envs = envs_func.VecRware(cfgs.train_cfgs['num_process'], "rware-tiny-2ag-v1") #agent 2
    envs = envs_func.RwareWrapper(envs)

    model = ActorCritic(envs.observation_space, envs.action_space, cfgs, device)


    # creates and initialises storage
    obs = envs.reset()
    parallel_envs = obs[0].shape[0]

    batch_obs = torch.zeros(cfgs.env_cfgs['n_steps'] + 1, parallel_envs, flatdim(envs.observation_space), device=device)
    batch_done = torch.zeros(cfgs.env_cfgs['n_steps'] + 1, parallel_envs, device=device)
    batch_act = torch.zeros(cfgs.env_cfgs['n_steps'], parallel_envs, len(envs.action_space), device=device)
    batch_rew = torch.zeros(cfgs.env_cfgs['n_steps'], parallel_envs, len(envs.observation_space), device=device)

    batch_obs[0, :, :] = torch.cat([torch.from_numpy(o) for o in obs], dim=1)

    storage = defaultdict(lambda: deque(maxlen=cfgs.env_cfgs['n_step']))
    storage["info"] = deque(maxlen=20)

    for step in tqdm(range(cfgs.env_cfgs['total_steps'] + 1)):

        for n in range(cfgs.env_cfgs['n_steps']):
            with torch.no_grad():
                actions = model.act(model.split_obs(batch_obs[n, :, :]))

            obs, reward, done, info = envs.step([x.squeeze().tolist() for x in torch.cat(actions, dim=1).split(1, dim=0)])
            done = torch.tensor(done, dtype=torch.float32)

            batch_obs[n + 1, :, :] = torch.cat([torch.from_numpy(o) for o in obs], dim=1)
            batch_act[n, :, :] = torch.cat(actions, dim=1)
            batch_done[n + 1, :] = all(done)
            batch_rew[n, :] = torch.tensor(reward)
            storage["info"].extend([i for i in info if "episode_returns" in i])

        model.update(batch_obs, batch_act, batch_rew, batch_done, step)

        batch_obs[0, :, :] = batch_obs[-1, :, :]
        batch_done[0, :] = batch_done[-1, :]

    envs.close()
