from gym.spaces import flatdim
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def _split_batch(splits):
    def thunk(batch):
        return torch.split(batch, splits, dim=-1)

    return thunk

@torch.jit.script
def compute_returns(rewards, done, next_value, gamma: float):
    returns = [next_value]
    for i in range(len(rewards) - 1, -1, -1):
        ret = rewards[i] + gamma * returns[0] * (1 - done[i, :].unsqueeze(1))
        returns.insert(0, ret)
    return returns


class CriticFCNetwork(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim):
        super(CriticFCNetwork, self).__init__()

        # state, action, other_actions, index
        input_dim = (state_dim * agent_num) + action_dim + (action_dim * agent_num) + agent_num

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RNNAgent(nn.Module):
    def __init__(self, obs_space, hidden_dim, action_space, device):
        super(RNNAgent, self).__init__()

        obs_shape = [flatdim(o) for o in obs_space]
        action_shape = [flatdim(a) for a in action_space]
        self.split_obs = _split_batch([flatdim(s) for s in obs_space])
        self.n_agents = len(obs_shape)

        self.fc1 = nn.Linear(obs_shape[0] + self.n_agents, hidden_dim[0])
        self.rnn = nn.GRUCell(hidden_dim[0], hidden_dim[1])
        self.fc2 = nn.Linear(hidden_dim[1], action_shape[0])

        self.device = device
        self.hidden_dim = hidden_dim[0]
        self.hidden_states = None

    def init_hidden(self, batch_size, n_agents):
        # make hidden states on same device as model
        hidden_states = self.fc1.weight.new(1, self.hidden_dim).zero_()
        self.hidden_states = hidden_states.unsqueeze(0).expand(batch_size, n_agents, -1)

    def forward(self, inputs):

        batch_size, _ = inputs.shape

        build_inputs = self.build_actor_input(inputs)
        x = F.relu(self.fc1(build_inputs))
        h_in = self.hidden_states.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        # policy_logit for choose action
        action_pi = F.softmax(q, dim=-1)
        action_pi = action_pi.view(batch_size, self.n_agents, -1)
        return action_pi, self.hidden_states

    def act(self, inputs):

        q, h = self.forward(inputs)
        dist = Categorical(q)
        actions = dist.sample().long()

        return actions

    def build_actor_input(self, batch_obs):
        batch, _ = batch_obs.shape

        batch_split_obs = self.split_obs(batch_obs)  # num_agent, batch, feature
        batch_split_obs = torch.cat(batch_split_obs, dim=0)  # num_agent * batch, feature

        # index (8, 2, 2)
        batch_index = torch.eye(self.n_agents, device=self.device).unsqueeze(0).expand(batch, -1,  -1)
        batch_index = batch_index.reshape(batch * self.n_agents, -1)  # batch * num_agent , feature

        inputs = torch.cat([batch_split_obs, batch_index], dim=-1)

        return inputs


class COMA(nn.Module):
    def __init__(self, actor, obs_space, action_space, cfg, device):
        super(COMA, self).__init__()
        self.gamma = cfg.model_cfgs['gamma']
        self.entropy_coef = cfg.model_cfgs['entropy_coef']
        self.grad_clip = cfg.model_cfgs['grad_clip']
        self.value_loss_coef = cfg.model_cfgs['value_loss_coef']
        self.n_steps = cfg.env_cfgs['n_steps']

        self.n_agents = len(obs_space)
        self.obs_shape = [flatdim(o) for o in obs_space]  # [71, 71]
        self.action_shape = [flatdim(a) for a in action_space]  # [5, 5]

        self.critic = CriticFCNetwork(self.n_agents, self.obs_shape[0], self.action_shape[0])
        self.target_critic = CriticFCNetwork(self.n_agents, self.obs_shape[0], self.action_shape[0])
        self.actor = actor

        self.device = device

        self.soft_update(1.0)
        self.to(device)

        optimizer = getattr(optim, cfg.model_cfgs['optimizer'])
        if type(optimizer) is str:
            optimizer = getattr(optim, optimizer)

        lr = cfg.model_cfgs['lr']

        self.critic_optimizer = optimizer(self.critic.parameters(), lr=lr)
        self.actor_optimizer = optimizer(self.actor.parameters(), lr=lr)

        self.target_update_interval_or_tau = 200

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError(
            "Forward not implemented. Use act, get_value, get_target_value or evaluate_actions instead.")

    def soft_update(self, t):
        source, target = self.critic, self.target_critic
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

    def update(self, actor, batch_obs, batch_act, batch_rew, batch_done, step):
        n_steps, batch_size, feature = batch_act.shape

        batch_critic_input = self.build_critic_input(batch_obs, batch_act)

        '''train_critic''' #TODO mask 적용하기?
        with torch.no_grad():
            next_value = self.target_critic(batch_critic_input[self.n_steps-1])

        index_act = batch_act[self.n_steps - 1].long().unsqueeze(-1)  # (8, 2, 1)
        next_value_taken = torch.gather(next_value, dim=-1, index=index_act).squeeze(-1)  # (8, 2, 5)

        returns = compute_returns(batch_rew, batch_done, next_value_taken, self.gamma)  # (11, 8, 2)
        returns = torch.stack(returns)[:-1]  # (10, 8, 2)

        index_acts = batch_act.long().unsqueeze(-1)  # 10, 8, 2, 1

        values = self.critic(batch_critic_input)  # 10, 8, 2, 5
        values_taken = torch.gather(values, dim=-1, index=index_acts).squeeze(-1)

        critic_loss = (returns - values_taken).pow(2).sum(dim=2).mean()  # TODO check

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        '''optimize agents(actor)'''
        actor_out = []
        actor.init_hidden(batch_size=batch_size, n_agents=self.n_agents)
        for i in range(n_steps):
            agent_outs, hidden_states = actor.forward(batch_obs[i])
            actor_out.append(agent_outs)
        actor_out = torch.stack(actor_out, dim=0)  # (10, 8, 2, 5)

        # calculated baseline
        values = values.reshape(-1, self.action_shape[0])
        pi = actor_out.view(-1, self.action_shape[0])
        baseline = (pi * values).sum(-1).detach()

        # Calculate policy grad with mask
        values_taken_reshape = values_taken.reshape(-1, 1).squeeze(1)
        pi_taken = torch.gather(actor_out, dim=-1, index=index_acts).squeeze(1)
        log_pi_taken = torch.log(pi_taken)

        # advantage
        advantage = (values_taken_reshape - baseline).detach()

        entropy = torch.sum(pi * torch.log(pi + 1e-10), dim=-1)

        actor_loss = (
                -(advantage * log_pi_taken + self.entropy_coef * entropy)
                .sum(dim=2).mean()
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.target_update_interval_or_tau > 1.0 and step % self.target_update_interval_or_tau == 0:
            self.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            self.soft_update(self.target_update_interval_or_tau)

        return critic_loss

    def build_critic_input(self, batch_obs, batch_act):
        num_step, batch, _ = batch_act.shape

        batch_obs = batch_obs.unsqueeze(2).repeat(1, 1, self.n_agents, 1)  # (11, 8, 142) -> (11, 8, 2, 142)

        # joint action (8, 2, 5)
        batch_onehot_act = F.one_hot(batch_act.long(), num_classes=self.action_shape[0])  # joint action (10, 8, 2, 5)

        # masked other action (8, 2, 10(me, other))
        agent_mask = (1 - torch.eye(self.n_agents, device=self.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.action_shape[0]).view(self.n_agents, -1)

        agent_mask = agent_mask.unsqueeze(0).unsqueeze(0).expand(num_step, batch, -1, -1)  # 10, 8, 2, 10

        batch_onehot_act_reshape = batch_onehot_act.view(num_step, batch, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1)

        batch_other_act = batch_onehot_act_reshape * agent_mask

        # index (8, 2, 2)
        batch_index = torch.eye(self.n_agents, device=self.device).unsqueeze(0).unsqueeze(0).expand(num_step, batch, -1, -1)

        inputs = torch.cat([batch_obs[: num_step], batch_onehot_act, batch_other_act, batch_index], dim=-1)

        return inputs
