import torch.nn as nn
import torch as tr

class MLPNetwork(nn.Module):

    def __init__(self, num_input, num_output, num_hidden=128):

        super(MLPNetwork, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_input, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_output)
        )

    def forward(self, obss):
        return self.nn(obss)
