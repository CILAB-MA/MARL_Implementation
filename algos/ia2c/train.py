from tqdm import tqdm
from algos.ia2c.agent import IA2CAgent
import torch as tr
from utils.envs_func import VecRware, RwareWrapper
from algos.ia2c.utils import RolloutBuffer
def train(cfgs):
    train_cfgs = cfgs.train_cfgs
    print(train_cfgs)
    env_cfgs = cfgs.env_cfgs
    model_cfgs = cfgs.model_cfgs
    t = 0
    envs = VecRware(train_cfgs['num_process'], "rware-tiny-2ag-v1") #agent 2
    envs = RwareWrapper(envs)
    obss = envs.reset()
    agent = IA2CAgent(None, env_cfgs, model_cfgs, train_cfgs)
    storage = RolloutBuffer() # todo: rollout buffer 제작
    for _ in tqdm(range(train_cfgs['total_timesteps'])):
        actions = agent.act(obss) # todo: action categorical 하게 샘플링 하는거 짜기
        next_obss, rews, dones, infos = envs.step(actions)
        storage.add(obss, actions, rews, dones, ) # todo: 더 넣을거 추려보기
        t += 1
        if t > 0 and t % train_cfgs['rollout_step'] == 0:
            next_values = agent.get_value(next_obss) # todo: get_value 함수 짜기
            storage.compute_advantage(next_values, dones) # todo: advantage 계산 짜기
            agent.update(storage) # todo: rollout buffer 기반 update 함수 짜기
            storage.clear(next_obss) # todo: storage clear 짜기
        obss = next_obss
    print('Train completed!!')

