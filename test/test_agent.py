import gym
import rware
import os, sys
sys.path.append(os.getcwd())
from algos.base.agent import BaseAgent

def test_random_agent(model, model_cfgs, train_cfgs):
    env = gym.make("rware-tiny-2ag-v1")
    obs = env.reset()
    env_cfgs = dict(action_space=env.action_space, observation_space=env.observation_space,
                    num_agent=env.n_agents)
    agent = BaseAgent(model=model,
                      model_cfgs=model_cfgs,
                      train_cfgs=train_cfgs,
                      env_cfgs=env_cfgs)
    actions = agent.act(obs)
    done = [False, False]

    while not done[0]:
        print(f'actions: {actions}')
        n_obs, reward, done, info = env.step(actions)
        actions = agent.act(obs)
        print(f'done: {done}')  # [False, False]
        print(f'reward: {reward}')  # [0.0, 0.0]
    env.close()


if __name__ == '__main__':
    model = None
    model_cfgs = None
    train_cfgs = dict(device='cuda')
    test_random_agent(model, model_cfgs, train_cfgs)