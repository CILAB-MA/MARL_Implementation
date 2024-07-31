from algos.qtran.model import QTRANNet
from algos.qtran.replay_buffer import ReplayBuffer
from utils.train_func import EpsilonScheduler
import random
import torch as tr
import torch.nn.functional as F

class QTRANAgent(object):

    def __init__(self, model, env_cfgs, model_cfgs, train_cfgs):
        self.model = QTRANNet()
        self.replay_buffer = ReplayBuffer(train_cfgs, model_cfgs, env_cfgs)
        self.epsilon_scheduler = EpsilonScheduler(
            start=train_cfgs['epsilon'],
            end=train_cfgs['epsilon_min'],
            decay_steps=train_cfgs['epsilon_decay_steps'],
            interval=train_cfgs['epsilon_decay_interval']
        )
        self.gamma = 0.99
    def update(self):
        minibatch = self.replay_buffer.sample()
        obss, actions, rewards, next_obss, dones = minibatch
        indiv_q = self.model.get_indiv_q(obss)
        indiv_q_target = self.model.get_target_q(next_obss)
        next_action_target = self.model.target_act(next_obss)
        chosen_indiv_q = tr.gather(indiv_q, dim=-1, index=actions)
        target_max_actions = indiv_q_target.max(dim=-1, keepdim=True)[1]
        max_next_action_oh = target_max_actions.scatter(3, target_max_actions[:, :], 1) # todo: change to onehot
        # TD loss
        joint_qs, vs = self.model(obss, actions)
        # max_action_qvals, max_action_current = self.model.act(obss, max_actions_onehot) # todo: what is difference between argmax and just get actions?

        target_joint_qs, target_vs = self.model.get_joint_target_q(next_obss, max_next_action_oh)
        td_target = rewards + self.gamma * ( 1- dones) * target_joint_qs
        td_error = F.mse_loss(joint_qs, td_target.detach())

        # Opt loss
        max_action_current = indiv_q.max(dim=-1, keepdim=True)[1]
        max_actions_oh = target_max_actions.scatter(3, max_action_current[:, :], 1) # todo: change to onehot

        max_joint_qs, max_vs = self.model.get_joint_q(obss, max_actions_oh)

    def act(self, obss):
        epsilon = self.epsilon_scheduler.get_epsilon()

        obss = [obs for obs in obss]
        actions = self.model.act(obss)

        for idx in range(self.env_cfgs["num_agent"]):
            if random.random() < epsilon:
                num_process = obss[idx].shape[0]
                action = tr.randint(
                    high=int(self.env_cfgs['act_space'][0]),
                    size=(num_process,),
                    device=obss[idx].device
                )
                actions[idx] = action

        actions = tr.stack(actions)
        actions = actions.t().detach().cpu().numpy()
        return actions

    def update_from_target(self) -> None:
        if (
            self.target_update_interval_or_tau > 1.0
            and self.updates % self.target_update_interval_or_tau == 0
        ):
            # Hard update
            self.model.soft_update(1.0)
        elif self.target_update_interval_or_tau < 1.0:
            # Soft update
            self.model.soft_update(self.target_update_interval_or_tau)
        self.updates += 1