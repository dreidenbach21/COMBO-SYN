import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions



class MLPPolicy(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 device = "cuda:0",
                 discrete=True,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.logits_na = build_mlp(input_size=self.ob_dim,
                                       output_size=self.ac_dim,
                                       n_layers=self.n_layers,
                                       size=self.size)
        self.logits_na.to(device)
        self.mean_net = None
        self.logstd = None
        self.optimizer = optim.Adam(self.logits_na.parameters(), self.learning_rate)

#         if nn_baseline:
# #             self.baseline = ptu.build_mlp(
# #                 input_size=self.ob_dim,
# #                 output_size=1,
# #                 n_layers=self.n_layers,
# #                 size=self.size,
# #             )
# #             self.baseline.to(ptu.device)
# #             self.baseline_optimizer = optim.Adam(
# #                 self.baseline.parameters(),
# #                 self.learning_rate,
#             )
#         else:
        self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

#     # query the policy with observation(s) to get selected action(s)
#     def get_action(self, obs: np.ndarray) -> np.ndarray:
#         if len(obs.shape) > 1:
#             observation = obs
#         else:
#             observation = obs[None]
#         observation = ptu.from_numpy(observation)
#         dist = self(observation)
#         if self.discrete:
#             d = dist.sample()
#         else:
#             d =  dist.rsample()
#         return d.detach().cpu().numpy()

#     # update/train this policy
#     def update(self, observations, actions, **kwargs):
        # raise NotImplementedError

    def forward(self, observation: torch.FloatTensor):
        logits = self.logits_na(observation)
        action_distribution = distributions.Categorical(logits=logits)
        return action_distribution

from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
    'softmax':nn.Softmax()
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)

