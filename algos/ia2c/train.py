import wandb, yaml
from algos.ia2c.agent import IA2CAgent
import numpy as np
from utils.envs_func import VecRware, RwareWrapper, RwareMonitor
from algos.ia2c.utils import RolloutBuffer, print_square
from collections import deque
from collections import defaultdict
import torch as tr
from gym.spaces import flatdim

def train(cfgs): # todo: config 맞춰 만들어주기
    train_cfgs = cfgs.train_cfgs
    env_cfgs = cfgs.env_cfgs
    model_cfgs = cfgs.model_cfgs
    if cfgs.train_cfgs['use_wandb']:
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
        wandb.init(project=private_info["project"], entity=private_info["entity"],
                   name='ia2c_value_coef')
    t = 0
    envs = VecRware(train_cfgs['num_process'], "rware-tiny-2ag-v1") #agent 2
    envs = RwareWrapper(envs)
    obss = envs.reset()
    num_obss = envs.observation_space[0].shape[-1]

    model_cfgs['action_space'] = envs.action_space
    model_cfgs['observation_space'] = envs.observation_space

    agent = IA2CAgent(None, env_cfgs, model_cfgs, train_cfgs)
    storage = RolloutBuffer(num_agent=env_cfgs['num_agent'],
                            num_obss=num_obss,
                            buffer_size=train_cfgs['rollout_step'],
                            num_process=train_cfgs['num_process'],
                            device=train_cfgs['device'])
    storage.reset()
    iter_num = int(train_cfgs['total_timesteps'] / train_cfgs['num_process'])
    log_dict = dict(episode_reward=[], value_loss=[], policy_loss=[])
    epi_rewards = deque(maxlen=20)

    batch_obs = tr.zeros(train_cfgs['rollout_step'] + 1, train_cfgs['num_process'], flatdim(envs.observation_space),
                         device=train_cfgs['device'])
    batch_done = tr.zeros(train_cfgs['rollout_step'] + 1, train_cfgs['num_process'], device=train_cfgs['device'])
    batch_act = tr.zeros(train_cfgs['rollout_step'], train_cfgs['num_process'], len(envs.action_space), device=train_cfgs['device'])
    batch_rew = tr.zeros(train_cfgs['rollout_step'], train_cfgs['num_process'], len(envs.observation_space), device=train_cfgs['device'])

    batch_obs[0, :, :] = tr.cat([tr.from_numpy(o) for o in obss], dim=1)


    for step in range(iter_num):
        for n in range(train_cfgs['rollout_step'] ):
            with tr.no_grad():
                actions = agent.act(agent.split_obs(batch_obs[n, :, :]))
            obss, rews, dones, infos = envs.step([x.squeeze().tolist() for x in tr.cat(actions, dim=1).split(1, dim=0)])
            for info in infos:
                if "episode_returns" in info:
                    epi_rewards.append(sum(info["episode_returns"]))

            batch_obs[n + 1, :, :] = tr.cat([tr.from_numpy(o) for o in obss], dim=1)
            batch_act[n, :, :] = tr.cat(actions, dim=1)
            batch_done[n + 1, :] = tr.tensor(dones, dtype=tr.float32)
            batch_rew[n, :] = tr.tensor(rews)

        batch_obs[0, :, :] = batch_obs[-1, :, :]
        if train_cfgs['use_wandb']:
            wandb.log({'epi_rewards': np.mean(epi_rewards)}, step=step)

        val_loss, pol_loss = agent.update(batch_obs, batch_act, batch_rew, batch_done, step)
        batch_done[0, :] = batch_done[-1, :]
        t += train_cfgs['num_process']
        # for info in infos:
        #     if "episode_returns" in info:
        #         epi_rewards.append(sum(info["episode_returns"]))
        #
        #     storage.compute_advantage(next_values, dones)
        #     val_loss, pol_loss = agent.update(storage) # todo: rollout buffer 기반 update 함수 짜기
        #     storage.reset()
        #     log_dict['value_loss'].append(val_loss)
        #     log_dict['policy_loss'].append(pol_loss)
        if step > 0 and step % 1000 == 0: # log freq
            for log in log_dict.keys():
                log_dict[log] = np.mean(log_dict[log])
            log_dict['timestep'] = t
            log_dict['episode_reward'] = np.mean(epi_rewards)
            print_square(log_dict)
            log_dict = dict(episode_reward=[], value_loss=[], policy_loss=[])
    print('Train completed!!')
    envs.close()

