import gym
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from itertools import product

from algos.cdqn.agent import CDQNAgent
from algos.cdqn.model import QNetwork
from algos.cdqn.replay_buffer import ReplayBuffer


def process_action_space(action_space):
    action_space = list(map(lambda x: range(x.n), action_space))
    action_space = product(*action_space)
    idx_to_action_dict = dict()
    action_to_idx_dict = dict()

    for i, a in enumerate(action_space):
        idx_to_action_dict[i] = a
        action_to_idx_dict[a] = i
    return idx_to_action_dict, action_to_idx_dict


def train(cfgs):
    # Initialize the environment
    env = gym.make("rware:rware-tiny-2ag-v1")

    # Update the environment configurations
    idx_to_action_dict, action_to_idx_dict = process_action_space(env.action_space)
    env_cfgs = dict(idx_action_dict=idx_to_action_dict,
                    action_idx_dict=action_to_idx_dict,
                    action_space=env.action_space,
                    observation_space=env.observation_space,
                    num_agent=env.n_agents)
    cfgs.env_cfgs.update(env_cfgs)

    model_cfgs = dict(obs_dim=env.observation_space[0].shape[0],
                      joint_obs_dim=env.observation_space[0].shape[0] * env.n_agents,
                      hidden_dim=128,
                      action_dim=env.action_space[0].n,
                      joint_action_dim=pow(env.action_space[0].n, env.n_agents),
                      writer=SummaryWriter())
    cfgs.model_cfgs.update(model_cfgs)

    # Initialize the agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qnet = QNetwork(cfgs.model_cfgs).to(device)
    agent = CDQNAgent(qnet=qnet, model_cfgs=cfgs.model_cfgs, env_cfgs=cfgs.env_cfgs)
    replay_buffer = ReplayBuffer(cfgs.model_cfgs['buffer_size'])
    optimizer = optim.Adam(qnet.parameters(), lr=cfgs.model_cfgs['lr'])

    # Run the environment
    for episode in tqdm(range(1, cfgs.train_cfgs['n_episodes'] + 1), unit='episode'):
        obs = env.reset()
        obs = [torch.tensor(ob, dtype=torch.float32) for ob in obs]
        obs = torch.cat(obs, dim=0)
        actions = agent.act(obs)
        done = [False for _ in range(env.n_agents)]
        rewards = 0

        while not all(done):
            next_obs, reward, done, info = env.step(actions)
            env.render()

            next_obs = [torch.tensor(n_ob, dtype=torch.float32) for n_ob in next_obs]
            next_obs = torch.cat(next_obs, dim=0)
            reward = sum(reward)
            rewards += reward
            replay_buffer.add(obs, actions, reward, next_obs, done)

            # Update QNetwork
            if len(replay_buffer) == cfgs.model_cfgs['buffer_size']:
                agent.update(replay_buffer, optimizer)

            obs = next_obs
            actions = agent.act(obs)

        # Update target
        if episode % 2 == 0:
            agent.qnet_target.load_state_dict(agent.qnet.state_dict())

        # Epsilon decay
        agent.decay_epsilon(episode, cfgs.train_cfgs['n_episodes'])

        # Log the rewards
        cfgs.model_cfgs['writer'].add_scalar('reward', rewards, episode)

    env.close()
