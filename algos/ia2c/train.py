import wandb, yaml
from algos.ia2c.agent import IA2CAgent
import numpy as np
from utils.envs_func import VecRware, RwareWrapper, RwareMonitor
from algos.ia2c.utils import RolloutBuffer, print_square
from collections import deque

def train(cfgs): # todo: config 맞춰 만들어주기
    train_cfgs = cfgs.train_cfgs
    env_cfgs = cfgs.env_cfgs
    model_cfgs = cfgs.model_cfgs
    # if cfgs.train_cfgs['use_wandb']:
    #     with open("private.yaml") as f:
    #         private_info = yaml.load(f, Loader=yaml.FullLoader)
    #     wandb.login(key=private_info["wandb_key"])
    #     wandb.init(project=private_info["project"], entity=private_info["entity"],
    #                name='"ia2c : {{{}}}'.format(cfgs.model_cfgs['centralised']))
    t = 0
    envs = VecRware(train_cfgs['num_process'], "rware-tiny-2ag-v1") #agent 2
    envs = RwareWrapper(envs)
    obss = envs.reset()
    num_obss = envs.observation_space[0].shape[-1]
    model_cfgs['observation_space'] = envs.observation_space
    model_cfgs['action_space'] = envs.action_space
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
    for n in range(iter_num):
        actions, log_probs = agent.act(obss)
        values = agent.get_value(obss)
        next_obss, rews, dones, infos = envs.step(actions)
        storage.add(obss, actions, rews, dones, values, log_probs)
        t += train_cfgs['num_process']
        for info in infos:
            if "episode_returns" in info:
                epi_rewards.append(sum(info["episode_returns"]))
        if storage.full:
            next_values = agent.get_value(next_obss)
            storage.compute_advantage(next_values, dones)
            val_loss, pol_loss = agent.update(storage) # todo: rollout buffer 기반 update 함수 짜기
            storage.reset()
            log_dict['value_loss'].append(val_loss)
            log_dict['policy_loss'].append(pol_loss)
        if n >0 and n % 1000 == 0: # log freq
            for log in log_dict.keys():
                log_dict[log] = np.mean(log_dict[log])
            log_dict['timestep'] = t
            log_dict['episode_reward'] = np.mean(epi_rewards)
            print_square(log_dict)
            log_dict = dict(episode_reward=[], value_loss=[], policy_loss=[])
        obss = next_obss
    print('Train completed!!')
    envs.close()

