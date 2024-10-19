import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torchkit.distributions import TanhNormal
from torchkit.networks import Mlp

from spikingjelly.activation_based import neuron, surrogate, layer, functional
from torchkit.snn_layer import *
import time
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
PROB_MIN = 1e-8


class MarkovPolicyBase(Mlp):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes,
        init_w=1e-3,
        image_encoder=None,
        **kwargs
    ):
        self.save_init_params(locals())
        self.action_dim = action_dim

        if image_encoder is None:
            self.input_size = obs_dim
        else:
            self.input_size = image_encoder.embed_size

        # first register MLP
        super().__init__(
            hidden_sizes,
            input_size=self.input_size,
            output_size=self.action_dim,
            init_w=init_w,
            **kwargs,
        )

        # then register image encoder
        self.image_encoder = image_encoder  # None or nn.Module

    def forward(self, obs, hidden_state=None):
        """
        :param obs: Observation, usually 2D (B, dim), but maybe 3D (T, B, dim)
        return action (*, dim)
        """
        x = self.preprocess(obs)
        return super().forward(x, hidden_state)

    def preprocess(self, obs):
        x = obs
        if self.image_encoder is not None:
            x = self.image_encoder(x)
        return x


class DeterministicPolicy(MarkovPolicyBase):
    """
    Usage: TD3
    ```
    policy = DeterministicPolicy(...)
    action = policy(obs)
    ```
    NOTE: action space must be [-1,1]^d
    """

    def forward(
        self,
        obs,
    ):
        h = super().forward(obs)
        action = torch.tanh(h)  # map into [-1, 1]
        return action


class TanhGaussianPolicy(MarkovPolicyBase):
    """
    Usage: SAC
    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.
    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    NOTE: action space must be [-1,1]^d
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_sizes,
        std=None,
        init_w=1e-3,
        image_encoder=None,
        **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            obs_dim, action_dim, hidden_sizes, init_w, image_encoder, **kwargs
        )
       

        self.log_std = None
        self.std = std
        # self.recs = 0.   #last added recurrency from output spikes to input
        # self.fc_0 = nn.Linear(hidden_sizes[-1], 1) # weights to act on the output spikes to add them to the input
        if std is None:  # learn std
            last_hidden_size = self.input_size
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            # initialized near zeros, https://arxiv.org/pdf/2005.05719v1.pdf fig 7.a
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:  # fix std
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

        #r.s.o
        if "T" in kwargs:
            self.T = kwargs["T"]


    def forward(
        self,
        obs,
        hidden_state=None, #r.s.o
        reparameterize=True,
        deterministic=False,
        return_log_prob=False,
    ):
        """
        :param obs: Observation, usually 2D (B, dim), but maybe 3D (T, B, dim)
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        # Measure time
        h = self.preprocess(obs)
        # h+=self.recs


        for i, fc in enumerate(self.fcs):

            h= fc(h) # r.s.o

            if isinstance(self.hidden_activation, list):    #r.s.o
                # No internal mem load - jelly
                if isinstance(self.hidden_activation[0], neuron.IFNode) or isinstance(self.hidden_activation[0], layer.LinearRecurrentContainer):

                    if len(obs.shape) == 2:   #[BS*T, dim] = [BS, dim]
                        # self.hidden_activation[i].step_mode= 's'
                        self.hidden_activation[i].step_mode= 'm'
                        h = h.reshape(self.T, -1, h.shape[-1])  #[T, BS, dim]
                        h = self.hidden_activation[i](h)
                        h = h.reshape(-1, h.shape[-1])  #[BS*T, dim]
                    else:                       #[episode, BS*T, dim] = [episode, BS, dim]
                        self.hidden_activation[i].step_mode= 'm'
                        h = self.hidden_activation[i](h)

            else:
                h = self.hidden_activation(h)

            # h = self.hidden_activation(fc(h))          # original



        mean = self.last_fc(h)
        # self.recs = self.fc_0(h)


        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        if deterministic:
            action = torch.tanh(mean)
            assert (
                return_log_prob == False
            )  # NOTE: cannot be used for estimating entropy
        else:
            tanh_normal = TanhNormal(mean, std)  # (*, B, dim)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
                log_prob = log_prob.sum(dim=-1, keepdim=True)  # (*, B, 1)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return action, mean, log_std, log_prob


class CategoricalPolicy(MarkovPolicyBase):
    """Based on https://github.com/ku2482/sac-discrete.pytorch/blob/master/sacd/model.py
    Usage: SAC-discrete
    ```
    policy = CategoricalPolicy(...)
    action, _, _ = policy(obs, deterministic=True)
    action, _, _ = policy(obs, deterministic=False)
    action, prob, log_prob = policy(obs, deterministic=False, return_log_prob=True)
    ```
    NOTE: action space must be discrete
    """

    def forward(
        self,
        obs,
        hidden_state=None,
        deterministic=False,
        return_log_prob=False,
    ):
        """
        :param obs: Observation, usually 2D (B, dim), but maybe 3D (T, B, dim)
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        return: action (*, B, A), prob (*, B, A), log_prob (*, B, A)
        """
        action_logits = super().forward(obs, hidden_state)  # (*, A)

        prob, log_prob = None, None
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)  # (*)
            assert (
                return_log_prob == False
            )  # NOTE: cannot be used for estimating entropy
        else:
            prob = F.softmax(action_logits, dim=-1)  # (*, A)
            distr = Categorical(prob)
            # categorical distr cannot reparameterize
            action = distr.sample()  # (*)
            if return_log_prob:
                log_prob = torch.log(torch.clamp(prob, min=PROB_MIN))

        # convert to one-hot vectors
        action = F.one_hot(action.long(), num_classes=self.action_dim).float()  # (*, A)

        return action, prob, log_prob
