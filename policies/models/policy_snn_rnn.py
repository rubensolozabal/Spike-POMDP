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
from copy import deepcopy

from torchkit.snn_layer import LIF, STC_LIF
from policies.models.snn_actor import Actor_SNN
from policies.models.recurrent_critic import Critic_RNN

class ModelFreeOffPolicy_SNN_RNN(nn.Module):

    ARCH = "snn"
    type_actor = "snn"
    type_critic = "rnn"

    def __init__(
        self,
        obs_dim,
        action_dim,
        encoder,
        algo_name,
        action_embedding_size,
        observ_embedding_size,
        reward_embedding_size,
        rnn_hidden_size,
        dqn_layers,
        policy_layers,
        rnn_num_layers=1,
        lr=3e-4,
        gamma=0.99,
        tau=5e-3,
        # pixel obs
        image_encoder_fn=lambda: None,
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



        self.critic = Critic_RNN(
            obs_dim,
            action_dim,
            encoder,
            self.algo,
            action_embedding_size,
            observ_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            dqn_layers,
            rnn_num_layers,
            image_encoder=image_encoder_fn(),  # separate weight
        )


        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        # target networks
        self.critic_target = deepcopy(self.critic)


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


        _, batch_size, _ = actions.shape
        if not self.algo.continuous_action:
            # for discrete action space, convert to one-hot vectors
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T, B, A)

        masks = batch["mask"]

        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss


        # extend observs, actions, rewards, dones from len = T to len = T+1
        _observs = torch.cat((observs[[0]], next_observs), dim=0)  # (T+1, B, dim)
        _actions = torch.cat(
            (ptu.zeros((1, batch_size, self.action_dim)).float(), actions), dim=0
        )  # (T+1, B, dim)
        _rewards = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), rewards), dim=0
        )  # (T+1, B, dim)
        _dones = torch.cat(
            (ptu.zeros((1, batch_size, 1)).float(), dones), dim=0
        )  # (T+1, B, dim)


        ### 1. Critic loss
        (q1_pred, q2_pred), q_target = self.algo.critic_loss(
            type_actor=self.type_actor,     #r.s.o
            type_critic=self.type_critic,      #r.s.o
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=_observs,
            actions=_actions,
            rewards=_rewards,
            dones=_dones,
            gamma=self.gamma,
            # next_observs=next_observs,
        )


        # masked Bellman error: masks (T,B,1) ignore the invalid error
        # this is not equal to masks * q1_pred, cuz the denominator in mean()
        # 	should depend on masks > 0.0, not a constant B*T
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks
        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        self.critic_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optimizer.step()


        # soft update
        self.soft_target_update()

        ### 2. Actor loss
        start = time.time()
        actor_loss, log_probs = self.algo.actor_loss(
            type_actor=self.type_actor,
            type_critic=self.type_critic,
            actor=self.actor,
            actor_target=self.actor_target,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
        )


        # masked policy_loss
        actor_loss = (actor_loss * masks).sum() / num_valid

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

        ### 4. update others like alpha
        if log_probs is not None:
            # extract valid log_probs
            with torch.no_grad():
                if self.type_actor != 'snn':
                    log_probs = log_probs[:-1] # r.s.o
                current_log_probs = (log_probs * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.actor, self.actor_target, self.tau)
