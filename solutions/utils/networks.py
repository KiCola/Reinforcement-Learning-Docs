"""
[标准答案] 通用神经网络模块
"""
import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256), activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), activation()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dims=(128, 128)):
        super().__init__()
        self.net = MLP(obs_dim, n_actions, hidden_dims)
    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dims=(128, 128)):
        super().__init__()
        self.feature      = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1])
        self.value_stream = nn.Linear(hidden_dims[-1], 1)
        self.adv_stream   = nn.Linear(hidden_dims[-1], n_actions)
    def forward(self, x):
        feat = self.feature(x)
        V = self.value_stream(feat)
        A = self.adv_stream(feat)
        return V + A - A.mean(dim=1, keepdim=True)


class ActorContinuous(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256), action_scale=1.0):
        super().__init__()
        self.net = MLP(obs_dim, action_dim, hidden_dims)
        self.action_scale = action_scale
    def forward(self, x):
        return torch.tanh(self.net(x)) * self.action_scale


class ActorStochastic(nn.Module):
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.net           = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1])
        self.mean_layer    = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
    def forward(self, x):
        feat    = self.net(x)
        mean    = self.mean_layer(feat)
        log_std = self.log_std_layer(feat).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    def get_action(self, x):
        mean, log_std = self.forward(x)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t  = dist.rsample()
        y_t  = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob, torch.tanh(mean)


class CriticQ(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.net = MLP(obs_dim + action_dim, 1, hidden_dims)
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

