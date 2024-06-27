
import torch as tr
import numpy as np

class RolloutBuffer(object):

    def __init__(self, num_agent, num_obss):

        self.obss = np.zeros(())