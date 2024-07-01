from tqdm import tqdm
from algos.ia2c.agent import IA2CAgent
import torch as tr
from utils.envs_func import VecRware, RwareWrapper
from algos.ia2c.utils import RolloutBuffer

def train(cfgs): # todo: config 맞춰 만들어주기
    train_cfgs = cfgs.train_cfgs
    env_cfgs = cfgs.env_cfgs
    model_cfgs = cfgs.model_cfgs
    t = 0
    envs = VecRware(train_cfgs['num_process'], "rware-tiny-2ag-v1") #agent 2
    envs = RwareWrapper(envs)
    obss = envs.reset()
    num_obss = envs.observation_space[0].shape[-1]
    model_cfgs['num_obss'] = num_obss
    model_cfgs['num_action'] = envs.action_space[0].n
    agent = IA2CAgent(None, env_cfgs, model_cfgs, train_cfgs)
    storage = RolloutBuffer(num_agent=env_cfgs['num_agent'],
                            num_obss=model_cfgs['num_obss'],
                            buffer_size=train_cfgs['rollout_step'],
                            num_process=train_cfgs['num_process'],
                            device=train_cfgs['device'])
    storage.reset()
    print(train_cfgs, env_cfgs, model_cfgs)
    for _ in tqdm(range(train_cfgs['total_timesteps'])):
        actions, log_probs = agent.act(obss)
        values = agent.get_value(obss)
        next_obss, rews, dones, infos = envs.step(actions)
        storage.add(obss, actions, rews, dones, values, log_probs)
        t += 1
        if t > 0 and t % train_cfgs['rollout_step'] == 0:
            next_values = agent.get_value(next_obss)
            storage.compute_advantage(next_values, dones)
            agent.update(storage) # todo: rollout buffer 기반 update 함수 짜기
            storage.reset()
        obss = next_obss
    print('Train completed!!')

