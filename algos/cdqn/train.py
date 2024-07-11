import torch
from torch import optim
from tqdm import tqdm
import yaml
import wandb

from algos.cdqn.agent import CDQNAgent
from algos.cdqn.model import QNetwork
from algos.cdqn.replay_buffer import ReplayBuffer

from utils.envs_func import VecRware, RwareWrapper


def train(cfgs):
    # Initialize the environment
    num_process = cfgs.train_cfgs['num_process']
    envs = VecRware(num_process, "rware-tiny-2ag-v1")
    envs = RwareWrapper(envs)
    num_agent = len(envs.observation_space)

    # Update the environment configurations
    env_cfgs = dict(action_space=envs.action_space,
                    observation_space=envs.observation_space,
                    num_agent=num_agent)
    cfgs.env_cfgs.update(env_cfgs)

    model_cfgs = dict(joint_obs_dim=envs.observation_space[0].shape[0] * len(envs.observation_space),
                      hidden_dim=128,
                      action_dim=envs.action_space[0].n,
                      num_agent=num_agent)
    cfgs.model_cfgs.update(model_cfgs)

    # Initialize the agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qnet = QNetwork(cfgs.model_cfgs).to(device)
    agent = CDQNAgent(qnet=qnet, env_cfgs=cfgs.env_cfgs)
    replay_buffer = ReplayBuffer(cfgs.model_cfgs['buffer_size'])
    optimizer = optim.Adam(qnet.parameters(), lr=cfgs.model_cfgs['lr'])

    # Initialize wandb
    with open("private.yaml") as f:
        private_info = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=private_info["wandb_key"])
    wandb.init(project=private_info["project"], entity=private_info["entity"], name='cdqn', monitor_gym=True)

    # Run the environment
    for episode in tqdm(range(1, (cfgs.train_cfgs['n_episodes'] + 1) // cfgs.train_cfgs['num_process']), unit='episode'):
        obs = envs.reset()
        obs = torch.tensor(obs, device=device, dtype=torch.float32).reshape(num_process, -1)
        obs = obs.unsqueeze(0).repeat(2, 1, 1)
        actions = agent.act(obs, num_process)
        done = [False for _ in range(num_process)]
        rewards = 0

        while not all(done):
            next_obs, reward, done, info = envs.step(actions)
            envs.render()

            next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32).reshape(num_process, -1)
            next_obs = next_obs.unsqueeze(0).repeat(2, 1, 1)
            reward = reward.sum(axis=1)
            rewards += reward.mean()

            for i in range(num_process):
                replay_buffer.add(obs[:, i, :], actions[i], reward[i], next_obs[:, i, :], done[i])

            # Update QNetwork
            if len(replay_buffer) == cfgs.model_cfgs['buffer_size']:
                agent.update(replay_buffer, optimizer)

            obs = next_obs
            actions = agent.act(obs, num_process)

        # Update target
        if episode % 2 == 0:
            agent.qnet_target.load_state_dict(agent.qnet.state_dict())

        # Epsilon decay
        agent.decay_epsilon(episode, cfgs.train_cfgs['n_episodes'])

        wandb.log({'reward': rewards, 'epsilon': agent.qnet.epsilon}, step=episode)

    envs.close()
