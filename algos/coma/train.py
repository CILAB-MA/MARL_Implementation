from tqdm import tqdm
from collections import deque
from collections import defaultdict
from gym.spaces import flatdim
import torch
from utils import envs_func
from algos.coma.model import COMA, RNNAgent
import numpy as np
import wandb
import yaml
import datetime


def start_wandb(cfg):
    with open("private.yaml") as f:
        private_info = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=private_info["wandb_key"])
    run = wandb.init(project=private_info["project"], entity=private_info["entity"])
    return run


def train(cfgs):
    train_cfgs = cfgs.train_cfgs
    env_cfgs = cfgs.env_cfgs
    model_cfgs = cfgs.model_cfgs

    run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if cfgs.train_cfgs['use_wandb']:
        wandb_run = start_wandb(cfgs)
        wandb_run.name = f"coma_{run_name}"
        wandb.config.update(env_cfgs)
        wandb.config.update(model_cfgs)
        wandb.config.update(train_cfgs)

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    envs = envs_func.VecRware(cfgs.train_cfgs['num_process'], "rware:rware-tiny-2ag-v1")
    envs = envs_func.RwareWrapper(envs)

    agent = RNNAgent(envs.observation_space, [64, 64], envs.action_space, device)
    agent.init_hidden(batch_size=cfgs.train_cfgs['num_process'], n_agents=len(envs.observation_space))
    model = COMA(agent, envs.observation_space, envs.action_space, cfgs, device)


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

    epi_rewards = deque(maxlen=20)
    epi_rewards_0 = deque(maxlen=20)
    epi_rewards_1 = deque(maxlen=20)

    for step in tqdm(range(cfgs.env_cfgs['total_steps'] + 1)):

        for n in range(cfgs.env_cfgs['n_steps']):
            with torch.no_grad():
                """actor"""
                actions = agent.act(batch_obs[n, :, :])

            obs, reward, done, infos = envs.step(actions.tolist())
            print('reward', reward)
            done = torch.tensor(done, dtype=torch.float32)

            batch_obs[n + 1, :, :] = torch.cat([torch.from_numpy(o) for o in obs], dim=1)
            batch_act[n, :, :] = actions
            batch_done[n + 1, :] = all(done)
            batch_rew[n, :] = torch.tensor(reward)
            storage["info"].extend([i for i in infos if "episode_returns" in i])

            for info in infos:
                if "episode_returns" in info:
                    epi_rewards.append(sum(info["episode_returns"]))
                    epi_rewards_0.append(info["episode_returns"][0])
                    epi_rewards_1.append(info["episode_returns"][1])

        """critic_update"""
        loss = model.update(agent, batch_obs, batch_act, batch_rew, batch_done, step)

        batch_obs[0, :, :] = batch_obs[-1, :, :]
        batch_done[0, :] = batch_done[-1, :]

        if cfgs.train_cfgs['use_wandb']:
            wandb.log({'epi_rewards': np.mean(epi_rewards)}, step=step)
            wandb.log({"loss": loss}, step=step)

    envs.close()
