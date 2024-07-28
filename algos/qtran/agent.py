from algos.qtran.model import QTRANNet
from algos.qtran.replay_buffer import ReplayBuffer
from utils.train_func import EpsilonScheduler
import random

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

    def update(self):
        pass

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