from tqdm import tqdm
from algos.idqn.agent import IDQNAgent

from utils.envs_func import VecRware, RwareWrapper
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder

import torch

import gym
import os, json
import wandb
import numpy as np

import datetime

def train(cfgs):    
    train_cfgs = cfgs.train_cfgs
    env_cfgs = cfgs.env_cfgs
    model_cfgs = cfgs.model_cfgs
    
    use_wandb = train_cfgs['use_wandb']
    
    run_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f"./runs/idqn/{run_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if use_wandb:
        wandb_run = start_wandb()
        wandb_run.name = f"idqn_{run_name}"
        wandb.config.update(env_cfgs)
        wandb.config.update(model_cfgs)
        wandb.config.update(train_cfgs)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model_cfgs['device'] = device
    
    envs = VecRware(train_cfgs['num_process'], "rware:rware-tiny-2ag-v1") #agent 2
    envs = RwareWrapper(envs)
    
    test_env = gym.make("rware:rware-tiny-2ag-v1")


    obss = envs.reset()    
    test_obs = test_env.reset()
    
    env_cfgs['num_agent'] = len(obss)
    env_cfgs['act_space'] = [act_space.n for act_space in envs.action_space]
    env_cfgs['obs_space'] = [env_space.shape[0] for env_space in envs.observation_space]

    agent = IDQNAgent(env_cfgs, model_cfgs, train_cfgs)
    
    train_reward_sum = [0., 0.]
    train_reward_avg = [0., 0.]
    train_reward_avg_history = [[0.,0.]]
    test_rewards_sum = 0.0

    p_bar = tqdm(range(int(train_cfgs['total_timesteps']/train_cfgs['num_process'])))
    for t in p_bar:
        obss_tensor = torch.tensor(obss, device=device)
        actions = agent.act(obss_tensor)
        
        next_obss, rewards, dones, infos = envs.step(actions)
        if env_cfgs['central_reward']:
            rewards_sum = np.sum(rewards, axis=1, keepdims=True)
            rewards_sum  = np.full(rewards.shape, rewards_sum )
        else:
            rewards_sum = rewards
        is_full = agent.update_buffer(obss, actions, rewards_sum, next_obss, dones)
        

        train_reward_sum += rewards.sum(axis=0)
        if is_full:
            agent.update()

        obss = next_obss


            
        if dones[0] == True:
            train_reward_avg = train_reward_sum / train_cfgs['num_process']
            train_reward_avg_history.append(train_reward_avg)
            if use_wandb:
                wandb.log(step=t ,data={"epi_rewards": train_reward_avg.sum()})
                wandb.log(step=t ,data={"epsilon": agent.epsilon_scheduler.get_epsilon()})
                wandb.log(step=t ,data={"buffer_size": agent.replay_buffer.buffer_size if agent.replay_buffer.is_full() else agent.replay_buffer.pos})
                wandb.log(step=t ,data={"epi_reward(agent0)": train_reward_avg[0]})
                wandb.log(step=t ,data={"epi_reward(agent1)": train_reward_avg[1]})
                wandb.log(step=t ,data={"loss": agent.get_mean_loss()})
            train_reward_sum = [0., 0.]
        p_bar.set_description(f"[{t:>03d} iter]: train_rewards: {np.mean(train_reward_avg_history,axis=0)}, epsilon: {agent.epsilon_scheduler.get_epsilon():.3f}")
        
        if t % train_cfgs['save_interval'] == 0 and t > 0:
            agent.save_agent(os.path.join(save_dir, f"model_{t}_{train_reward_avg}.pt"))
                
        if t % train_cfgs['test_interval'] == 0 and t > 0:
            train_reward_avg_history = [[0.,0.]]
            with torch.no_grad():
                test_done = False
                test_rewards_sum = 0
                while not test_done:
                    test_obs_tensor = torch.tensor(test_obs, device=device).unsqueeze(1)
                    test_action = agent.act(test_obs_tensor).squeeze()
                    test_next_obs, test_reward, test_done, test_info = test_env.step(test_action)
                    test_done = test_done[0]
                    test_obs = test_next_obs
                    test_rewards_sum += test_reward[0] + test_reward[1]
                    test_env.render()
                test_env.reset()  
            print(f"test reward: {test_rewards_sum}")
    agent.save_agent(os.path.join(save_dir, f"model_final_{train_reward_avg}.pt"))
          
       
    print('Train completed!!')


def start_wandb():
    with open('./wandb_key.json', 'r') as f:
        wandb_key = json.load(f)
    wandb.login(key=wandb_key["wandb_key"])
    run = wandb.init(project='marl_implement', entity="cilab-ma", monitor_gym=True)
    return run

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
