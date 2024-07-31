import wandb, yaml, os
import torch as tr
from utils.envs_func import VecRware, RwareWrapper
from algos.qtran.agent import QTRANAgent

def train(cfgs):
    train_cfgs = cfgs.train_cfgs
    env_cfgs = cfgs.env_cfgs
    model_cfgs = cfgs.model_cfgs

    use_wandb = train_cfgs['use_wandb']
    device = tr.device('cuda:0' if tr.cuda.is_available() else "cpu")
    if use_wandb:
        if cfgs.train_cfgs['use_wandb']:
            with open("private.yaml") as f:
                private_info = yaml.load(f, Loader=yaml.FullLoader)
            wandb.login(key=private_info["wandb_key"])
            wandb.init(project=private_info["project"], entity=private_info["entity"],
                       name='qtran')

    envs = VecRware(train_cfgs['num_process'], "rware-tiny-2ag-v1")
    envs = RwareWrapper(envs)
    obss = envs.reset()

    env_cfgs['num_agent'] = len(obss)
    env_cfgs['act_space'] = [act_space.n for act_space in envs.action_space]
    env_cfgs['obs_space'] = [env_space.shape[0] for env_space in envs.observation_space]


    # todo: make saving model
    agent = QTRANAgent(None, env_cfgs, model_cfgs, train_cfgs)
    cumulative_reward_mean = 0
    cumulative_reward_means = [[0.,0.]]

    num_timesteps = int(train_cfgs['total_timesteps']/train_cfgs['num_process'])
    for step in range(num_timesteps):
        obss_tensor = tr.tensor(obss, device=device)
        actions = agent.act(obss_tensor)
        next_obss, rewards, dones, infos = envs.step(actions)
        cumulative_reward_mean += rewards.sum(axis=0)
        buffer.update(obss, actions, rewards, next_obss, dones) # todo: should make buffer

        if buffer.full:
            loss = agent.update(buffer)
        obss = next_obss
        if dones[0] == True:
            cumulative_reward_mean = cumulative_reward_mean / train_cfgs['num_process']
            cumulative_reward_means.append(cumulative_reward_mean)
            if use_wandb:
                wandb.log(step=step, data={"epi_rewards": cumulative_reward_mean.sum()})
                wandb.log(step=step, data={"loss": loss}) # todo: make loss to mean()

        if step % train_cfgs['save_interval'] == 0 and step > 0: # todo: make saving
            agent.save_model(os.path.join(save_dir, f'model_{step}.pt'))

    agent.save_model(os.path.join(save_dir, f'model_{step}.pt'))

    print('Train Completed!!')

