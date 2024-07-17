import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from torchkit.constant import *
import torchkit.pytorch_utils as ptu


class Actor_SNN(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        algo,
        policy_layers,
        hidden_activation = F.relu,
        image_encoder=None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algo = algo

        self.policy = self.algo.build_actor(
            input_size=obs_dim,
            action_dim=action_dim,
            hidden_sizes=policy_layers,
            hidden_activation=hidden_activation      #r.s.o
        )


    # def forward(self, observs):

    #     probs_list = []
    #     log_probs_list = []
        
    #     T = len(observs)    # observs [T, BS, obs_dim]
    #     for t in range(T):

    #         observ = observs[t]
    #         probs, log_probs = self.algo.forward_actor(actor=self.policy, observ=observ)

    #         probs_list.append(probs)
    #         log_probs_list.append(log_probs)

    #     return torch.stack(probs_list), torch.stack(log_probs_list)

    def forward(self, observs, return_log_prob=True):

        return self.policy(observs, return_log_prob=return_log_prob)

    @torch.no_grad()
    def get_initial_info(self):
        return [act.init_mem for act in self.policy.hidden_activation]
        # return [act.init_leaky() for act in self.policy.hidden_activation]    # snnTorch

    
    @torch.no_grad()
    def act(self, obs, prev_internal_state=None, deterministic=False, return_log_prob=False):

        a, prob, log_prob, _ = self.algo.select_action(
            actor=self.policy,
            observ=obs,
            hidden_state = prev_internal_state,
            deterministic=deterministic,
            return_log_prob=return_log_prob,
        )

        current_state = [act.current_mem for act in self.policy.hidden_activation]
        # current_state = [act.mem for act in self.policy.hidden_activation] # snnTorch
        
        return (a, prob, log_prob, _), current_state
