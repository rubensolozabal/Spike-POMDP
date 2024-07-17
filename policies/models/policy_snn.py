""" 
Episodic stateful SNN implementation for POMDP environments

Date: 2024-06-01
"""
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from policies.models import *
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu

from torchkit.snn_layer import LIF
from policies.models.snn_actor import Actor_SNN


class ModelFreeOffPolicy_Separate_SNN(nn.Module):

    ARCH = "snn"
    type_actor = "snn"
    type_critic = "snn"

    def __init__(
        self,
        obs_dim,
        action_dim,
        algo_name,
        dqn_layers,
        policy_layers,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.algo = RL_ALGORITHMS[algo_name](**kwargs[algo_name], action_dim=action_dim)

        # Get hidden activation - r.s.o
        if kwargs.get("hidden_activation") is None:
            hidden_activation = F.relu
        else:
            hidden_activation = []
            for i, value in enumerate(kwargs["hidden_activation"]):
                f_call = eval(value)
                # f_call = torch.jit.script(f_call)
                hidden_activation.append(f_call)


        # Markov q networks
        self.qf1, self.qf2 = self.algo.build_critic(
            obs_dim=obs_dim,
            hidden_sizes=dqn_layers,
            action_dim=action_dim,
            hidden_activation=hidden_activation      #r.s.o
        )
        self.qf1_optim = Adam(self.qf1.parameters(), lr=lr)
        self.qf2_optim = Adam(self.qf2.parameters(), lr=lr)
        # target networks
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)

        # Markov Actor
        self.actor = Actor_SNN(
            obs_dim=obs_dim,
            action_dim=action_dim,
            algo=self.algo,
            policy_layers=policy_layers,
            hidden_activation=hidden_activation      #r.s.o
        )
        self.actor_optim = Adam(self.actor.parameters(), lr=lr)
        # target network
        self.actor_target = copy.deepcopy(self.actor)

    @torch.no_grad()
    def act(self, obs, prev_internal_state=None, deterministic=False, return_log_prob=False):
        return self.actor.act(obs, prev_internal_state, deterministic, return_log_prob)
    
    @torch.no_grad()
    def get_initial_info(self):
        return self.actor.get_initial_info()
    
    def update(self, batch):
        observs, next_observs = batch["obs"], batch["obs2"]  # (B, dim)
        actions, rewards, dones = batch["act"], batch["rew"], batch["term"]  # (B, dim)

        ### 1. Critic loss
        start = time.time()
        (q1_pred, q2_pred), q_target = self.algo.critic_loss(
            type_actor=self.type_actor,
            type_critic=self.type_critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=(self.qf1, self.qf2),
            critic_target=(self.qf1_target, self.qf2_target),
            observs=observs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            next_observs=next_observs,
        )
        print("Critic loss time: ", time.time() - start)

        qf1_loss = F.mse_loss(q1_pred, q_target)  # TD error
        qf2_loss = F.mse_loss(q2_pred, q_target)  # TD error


        # update q networks
        start = time.time()
        self.qf1_optim.zero_grad()
        self.qf2_optim.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        self.qf1_optim.step()
        self.qf2_optim.step()
        print("Critic update time: ", time.time() - start)


        # soft update
        self.soft_target_update()

        ### 2. Actor loss
        start = time.time()
        actor_loss, log_probs = self.algo.actor_loss(
            type_actor=self.type_actor,
            type_critic=self.type_critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=(self.qf1, self.qf2),
            critic_target=(self.qf1_target, self.qf2_target),
            observs=observs,
        )
        actor_loss = actor_loss.mean()
        print("Actor loss time: ", time.time() - start)

        # update policy network
        start = time.time()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        print("Actor update time: ", time.time() - start)

        outputs = {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "policy_loss": actor_loss.item(),
        }

        # update others like alpha
        if log_probs is not None:
            current_log_probs = log_probs.mean().item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)
