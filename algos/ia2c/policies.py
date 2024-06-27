from algos.ia2c.model import MLPNetwork


class ActorCriticPolicy:

    def __init__(self, model_cfg):
        self.actor = MLPNetwork(model_cfg['num_obss'],
                                num_output=model_cfg['num_action'],
                                num_hidden=model_cfg['num_hidden'])
        self.critic = MLPNetwork(model_cfg['num_obss'],
                                num_output=1,
                                num_hidden=model_cfg['num_hidden'])

    def act(self, obss):
        pass