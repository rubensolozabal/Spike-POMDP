import torch
import torch.nn as nn
from torch.optim import Adam
from .base import RLAlgorithmBase
from policies.models.actor import TanhGaussianPolicy
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu
import time
import torch.nn.functional as F

from torchkit.snn_layer import LIF, STC_LIF


class SAC(RLAlgorithmBase):
    name = "sac"
    continuous_action = True
    use_target_actor = False

    def __init__(
        self,
        entropy_alpha=0.1,
        automatic_entropy_tuning=True,
        target_entropy=None,
        alpha_lr=3e-4,
        action_dim=None,
    ):

        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy is not None:
                self.target_entropy = float(target_entropy)
            else:
                self.target_entropy = -float(action_dim)
            self.log_alpha_entropy = torch.zeros(
                1, requires_grad=True, device=ptu.device
            )
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
            self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
        else:
            self.alpha_entropy = entropy_alpha

    def update_others(self, current_log_probs):
        if self.automatic_entropy_tuning:
            alpha_entropy_loss = -self.log_alpha_entropy.exp() * (
                current_log_probs + self.target_entropy
            )

            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()
            self.alpha_entropy = self.log_alpha_entropy.exp().item()

        return {"policy_entropy": -current_log_probs, "alpha": self.alpha_entropy}

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return TanhGaussianPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None, hidden_activation=F.relu):       #r.s.o: add hidden activation
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        qf1 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes, hidden_activation=hidden_activation
        )
        qf2 = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes, hidden_activation=hidden_activation
        )
        return qf1, qf2

    def select_action(self, actor, observ, deterministic: bool, return_log_prob: bool,hidden_state=None):
        
        # return actor(observ, False, deterministic, return_log_prob)
        return actor(observ, hidden_state, False, deterministic, return_log_prob)

    @staticmethod
    def forward_actor(actor, observ):
        new_actions, _, _, log_probs = actor(observ, return_log_prob=True)
        return new_actions, log_probs  # (T+1, B, dim), (T+1, B, 1)

    def critic_loss(
        self,
        type_actor, # r.s.o
        type_critic, # r.s.o
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        dones,
        gamma,
        next_observs=None,  # used in markov_critic
    ):
        markov_critic = False # r.s.o
        if type_critic == "mlp":
            markov_critic = True

        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from current policy,
            if type_actor == "mlp":
                new_actions, new_log_probs = self.forward_actor(
                    actor, next_observs if markov_critic else observs
                )
            elif type_actor == "snn":
                new_actions, new_log_probs = self.forward_actor(actor, next_observs if markov_critic else observs)
            else:
                # (T+1, B, dim) including reaction to last obs
                new_actions, new_log_probs = actor(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=next_observs if markov_critic else observs,
                )

            if type_critic == "mlp":  # (B, A)
                next_q1 = critic_target[0](next_observs, new_actions)
                next_q2 = critic_target[1](next_observs, new_actions)
            elif type_critic == "snn":
                start = time.time()
                next_q1 = critic_target[0](next_observs, new_actions)
                next_q2 = critic_target[1](next_observs, new_actions)
                # print("Critic_target time: ", time.time()-start)
            else:
                next_q1, next_q2 = critic_target(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=new_actions,
                )  # (T+1, B, 1)

            min_next_q_target = torch.min(next_q1, next_q2)
            min_next_q_target += self.alpha_entropy * (-new_log_probs)  # (T+1, B, 1)

            # q_target: (T, B, 1)
            q_target = rewards + (1.0 - dones) * gamma * min_next_q_target  # next q
            if type_critic == "rnn":
                q_target = q_target[1:]  # (T, B, 1)

        if type_critic == "mlp":  # (B, A)
            q1_pred = critic[0](observs, actions)
            q2_pred = critic[1](observs, actions)
        elif type_critic == "snn":
            start = time.time()
            q1_pred = critic[0](observs, actions)
            q2_pred = critic[1](observs, actions)
            # print("Critic time: ", time.time()-start)
        else:
            # Q(h(t), a(t)) (T, B, 1)
            q1_pred, q2_pred = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=actions[1:],
            )  # (T, B, 1)

        return (q1_pred, q2_pred), q_target

    def actor_loss(
        self,
        type_actor, # r.s.o
        type_critic, # r.s.o
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions=None,
        rewards=None,
        _observs=None,
        _actions=None,
        _rewards=None,
    ):
        markov_critic = False
        if type_critic == "mlp":
            markov_critic = True

        if type_actor == "mlp":
            new_actions, log_probs = self.forward_actor(actor, observs)
        elif type_actor == "snn":
            new_actions, log_probs = self.forward_actor(actor, observs)
        else:
            new_actions, log_probs = actor(
                prev_actions=actions, rewards=rewards, observs=observs
            )  # (T+1, B, A)

        if type_critic == "mlp":
            q1 = critic[0](observs, new_actions)
            q2 = critic[1](observs, new_actions)
        elif type_critic == "snn":
            start = time.time()
            q1 = critic[0](observs, new_actions)
            q2 = critic[1](observs, new_actions)    
            # print("Critic time: ", time.time()-start)     
        elif type_actor == "snn" and type_critic == "rnn": 
            q1, q2 = critic(
                prev_actions=_actions,
                rewards=_rewards,
                observs=_observs,
                current_actions=new_actions,
            )   
        else:
            q1, q2 = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=new_actions,
            )  # (T+1, B, 1)
        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1)

        policy_loss = -min_q_new_actions
        policy_loss += self.alpha_entropy * log_probs
        if type_critic == "rnn":
            if type_actor != "snn": #r.s.o 
                policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs

        return policy_loss, log_probs

    #### Below are used in shared RNN setting
    def forward_actor_in_target(self, actor, actor_target, next_observ):
        return self.forward_actor(actor, next_observ)

    def entropy_bonus(self, log_probs):
        return self.alpha_entropy * (-log_probs)
