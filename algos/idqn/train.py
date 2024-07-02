from tqdm import tqdm
from algos.idqn.agent import IDQNAgent

from utils.envs_func import VecRware, RwareWrapper
import torch


def train(cfgs):    
    train_cfgs = cfgs.train_cfgs
    env_cfgs = cfgs.env_cfgs
    model_cfgs = cfgs.model_cfgs
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model_cfgs['device'] = device
    
    envs = VecRware(train_cfgs['num_process'], "rware-tiny-2ag-v1") #agent 2
    envs = RwareWrapper(envs)
    obss = envs.reset()
    
    env_cfgs['num_agent'] = len(obss)
    env_cfgs['act_space'] = envs.action_space.n
    env_cfgs['obs_space'] = envs.observation_space.shape[0]
    
    agent = IDQNAgent(None, env_cfgs, model_cfgs, train_cfgs)
    
    t = 0
    for _ in tqdm(range(train_cfgs['total_timesteps'])):
        actions = agent.act(obss)
        next_obss, rewards, dones, infos = envs.step(actions)
        
        agent.update_buffer(obss, actions, rewards, next_obss, dones)
        
        if t > 0 and agent.replay_buffer_size() > train_cfgs['batch_size']:
            agent.update()
            obss = next_obss
            
        if t % train_cfgs['target_update_interval'] == 0:
            agent.update_target()
        t += 1
        
    print('Train completed!!')


#================
# Pseudo code

# N개의 네트워크 파라미터 초기화
# N개의 타겟 네트워크 동기화
# N개의 Relay Buffer 초기화

# for t in max_timesteps:
#   각 agent의 현재 observation 수집(N개)
#   for agent in agents:
#       epsilon-greedy로 action 선택
#   action을 사용한 observation, reward, done 수집
#   
#   for agent in agents:
#       relay buffer에 transition 저장
#       relay buffer에서 batch size만큼 transition 샘플링
#       if terminal state:
#           target_y = reward
#       else:
#           target_y = reward + gamma * max(Q(s', a'))
#       loss 계산(loss 수식 참고)
#       loss 역전파
#       if interval:
#           update target network
