from algos.ia2c.model import MLPNetwork
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch.optim as optim
class ActorCriticPolicy:

    def __init__(self, model_cfg):
        self.actor = MLPNetwork(model_cfg['num_obss'],
                                num_output=model_cfg['num_action'],
                                num_hidden=model_cfg['num_hidden'])
        self.critic = MLPNetwork(model_cfg['num_obss'],
                                num_output=1,
                                num_hidden=model_cfg['num_hidden'])

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0005)
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=0.0005)

    def act(self, obss, deterministic=False):
        logits = F.softmax(self.actor(obss))
        dist = Categorical(logits)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def get_log_prob(self, obss, action):
        logits = F.softmax(self.actor(obss), dim=-1)
        dist = Categorical(logits)
        log_prob = dist.log_prob(action)
        return log_prob

    def get_value(self, obss):
        return self.critic(obss)