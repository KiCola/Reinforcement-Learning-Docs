"""
[标准答案] Day 23：PPO 完整实现
参考：https://arxiv.org/abs/1707.06347
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Config:
    env_id: str           = "LunarLander-v2"
    total_timesteps: int  = 500_000
    lr: float             = 3e-4
    n_steps: int          = 2048
    n_epochs: int         = 10
    batch_size: int       = 64
    gamma: float          = 0.99
    gae_lambda: float     = 0.95
    clip_eps: float       = 0.2
    vf_coef: float        = 0.5
    ent_coef: float       = 0.01
    max_grad_norm: float  = 0.5
    seed: int             = 42


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            self._layer(nn.Linear(obs_dim, 64)), nn.Tanh(),
            self._layer(nn.Linear(64, 64)),      nn.Tanh(),
        )
        self.actor_head  = self._layer(nn.Linear(64, n_actions), std=0.01)
        self.critic_head = self._layer(nn.Linear(64, 1), std=1.0)

    @staticmethod
    def _layer(layer, std=np.sqrt(2), bias=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias)
        return layer

    def forward(self, x):
        feat   = self.shared(x)
        logits = self.actor_head(feat)
        value  = self.critic_head(feat).squeeze(-1)
        return logits, value

    def get_action(self, 
