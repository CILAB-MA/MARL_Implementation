import torch
import numpy as np
from tqdm import tqdm
import yaml
import wandb

from algos.cdqn.agent import CDQNAgent
from algos.cdqn.model import QNetwork
from algos.cdqn.replay_buffer import ReplayBuffer

from utils.envs_func import VecRware, CentralRwareWrapper

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(cfgs):
    # Initialize the environment
    num_process = cfgs.train_cfgs['num_process']
    envs = VecRware(num_process, "rware:rware-tiny-2ag-v1")
    envs = CentralRwareWrapper(envs)

    # Update the environment configurations
    env_cfgs = dict(action_space=envs.action_space,
                    observation_space=envs.observation_space,
                    num_agent=envs.num_agent,
                    num_process=num_process)
    cfgs.env_cfgs.update(env_cfgs)

    model_cfgs = dict(joint_obs_dim=envs.joint_obs_dim,
                      action_dim=envs.action_dim,
                      num_agent=envs.num_agent)
    cfgs.model_cfgs.update(model_cfgs)

    # Agent initialization
    qnet = QNetwork(cfgs.model_cfgs).to(DEVICE)
    agent = CDQNAgent(qnet=qnet, env_cfgs=cfgs.env_cfgs)
    replay_buffer = ReplayBuffer(cfgs.model_cfgs['buffer_size'])

    # Initialize wandb
    if cfgs.train_cfgs['use_wandb']:
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
        wandb.init(project=private_info["project"], entity=private_info["entity"], name='cdqn')
        wandb.config.update(cfgs)

    # Run the environment
    total_episode = (cfgs.train_cfgs['n_episodes'] + 1) // cfgs.train_cfgs['num_process']
    for episode in tqdm(range(1, total_episode), unit='episode'):
        obs = envs.reset()

        # reset values
        actions = agent.act(obs)
        done = [False for _ in range(num_process)]
        info = {}
        losses = [0]

        while not all(done):
            next_obs, reward, done, info = envs.step(actions)

            # Update replay buffer
            replay_buffer.add(obs, actions, reward, next_obs, done)

            # Update QNetwork
            if replay_buffer.full():
                loss = agent.update(replay_buffer)
                losses.append(loss)

            obs = next_obs
            actions = agent.act(obs)

        # Epsilon decay and target update
        if replay_buffer.full():
            agent.decay_epsilon(episode, total_episode, cfgs.model_cfgs['target_update_freq'])

        # Log wandb
        if cfgs.train_cfgs['use_wandb']:
            for i in range(envs.num_agent):
                wandb.log({f'epi_reward(agent{i})': np.mean([x['episode_returns'][i] for x in info])}, step=episode)
            wandb.log({'epi_rewards': np.mean([np.sum(x['episode_returns']) for x in info])}, step=episode)
            wandb.log({'epi_time': np.mean([x['episode_time'] for x in info])}, step=episode)
            wandb.log({'epsilon': agent.qnet.epsilon}, step=episode)
            wandb.log({'loss': np.mean(losses) if losses is not None else 0}, step=episode)
            wandb.log({'buffer_size': replay_buffer.__len__()}, step=episode)

    envs.close()
