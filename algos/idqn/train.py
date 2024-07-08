from tqdm import tqdm
from algos.idqn.agent import IDQNAgent

from utils.envs_func import VecRware, RwareWrapper, SquashRewards
import torch

import gym
import os

import datetime

def train(cfgs):    
    train_cfgs = cfgs.train_cfgs
    env_cfgs = cfgs.env_cfgs
    model_cfgs = cfgs.model_cfgs
    
    save_dir = f"./runs/idqn/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model_cfgs['device'] = device
    
    envs = VecRware(train_cfgs['num_process'], "rware-tiny-2ag-v1") #agent 2
    envs = RwareWrapper(envs)
    
    test_env = gym.make("rware-tiny-2ag-v1")
    
    if env_cfgs['central_reward']:
        envs = SquashRewards(envs)
    obss = envs.reset()    
    test_obs = test_env.reset()
    
    env_cfgs['num_agent'] = len(obss)
    env_cfgs['act_space'] = envs.action_space[0].n
    env_cfgs['obs_space'] = envs.observation_space[0].shape[0]
    
    agent = IDQNAgent(None, env_cfgs, model_cfgs, train_cfgs)
    train_reward_sum = 0.0
    train_reward_avg = 0.0
    test_rewards_sum = 0.0

    p_bar = tqdm(range(int(train_cfgs['total_timesteps']/train_cfgs['num_process'])))
    for t in p_bar:
        obss_tensor = torch.tensor(obss, device=device)
        actions = agent.act(obss_tensor)
        
        next_obss, rewards, dones, infos = envs.step(actions)
        is_full = agent.update_buffer(obss, actions, rewards, next_obss, dones)
        

        train_reward_sum += rewards.sum()
        if is_full:
            agent.update()
            obss = next_obss
            
        if t % train_cfgs['target_update_interval'] == 0:
            agent.update_target()
            
        if dones[0] == True:
            train_reward_avg = train_reward_sum / train_cfgs['num_process']
            train_reward_sum = 0.0
        p_bar.set_description(f"[{t:>03d} iter]: train_rewards: {train_reward_avg}, epsilon: {agent.epsilon_scheduler.get_epsilon():.3f}")
        
        if t % train_cfgs['save_interval'] == 0 and t > 0:
            agent.save_agent(os.path.join(save_dir, f"model_{t}_{train_reward_avg}.pt"))
        
        if t % train_cfgs['test_interval'] == 0 and t > 0:
            with torch.no_grad():
                test_done = False
                test_rewards_sum = 0
                while not test_done:
                    test_obs_tensor = torch.tensor(test_obs, device=device)
                    test_action = agent.act(test_obs_tensor).squeeze()
                    test_next_obs, test_reward, test_done, test_info = test_env.step(test_action)
                    test_done = test_done[0]
                    test_obs = test_next_obs
                    test_rewards_sum += test_reward[0] + test_reward[1]
                    test_env.render()
                test_env.reset()  
            print(f"test reward: {test_rewards_sum}")
    agent.save_agent(save_dir, os.path.join(save_dir, f"model_final_{train_reward_avg}.pt"))
          
       
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
