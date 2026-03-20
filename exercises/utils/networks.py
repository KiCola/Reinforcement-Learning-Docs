"""
【练习】通用神经网络模块
完成各网络类后，后续阶段直接 import 使用
标准答案：solutions/utils/networks.py
"""
import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化（PPO 推荐）
    TODO：用 nn.init.orthogonal_ 和 nn.init.constant_ 实现
    """
    # TODO
    raise NotImplementedError
    return layer


class MLP(nn.Module):
    """通用多层感知机
    例：MLP(4, 2, (128,128)) => Linear(4,128)-ReLU-Linear(128,128)-ReLU-Linear(128,2)
    TODO：用循环动态构建，nn.Sequential 打包
    """
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256), activation=nn.ReLU):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError


class QNetwork(nn.Module):
    """DQN Q 网络：s -> Q(s,·)
    TODO：用 MLP 实现
    """
    def __init__(self, obs_dim, n_actions, hidden_dims=(128, 128)):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError


class DuelingQNetwork(nn.Module):
    """Dueling DQN：Q = V + A - mean(A)
    TODO：共享特征层 + value_stream + adv_stream
    """
    def __init__(self, obs_dim, n_actions, hidden_dims=(128, 128)):
        super().__init__()
        # feature: MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1])
        # value_stream: Linear(hidden_dims[-1], 1)
        # adv_stream:   Linear(hidden_dims[-1], n_actions)
        # TODO
        raise NotImplementedError

    def forward(self, x):
        # TODO: Q = V + A - A.mean(dim=1, keepdim=True)
        raise NotImplementedError


class ActorContinuous(nn.Module):
    """确定性 Actor（DDPG/TD3）：输出 tanh(net(x))*action_scale"""
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256), action_scale=1.0):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError


class ActorStochastic(nn.Module):
    """随机 Actor（SAC）：输出高斯分布参数，支持重参数化采样"""
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        # TODO: 共享 net + mean_layer + log_std_layer
        raise NotImplementedError

    def forward(self, x):
        # TODO: 返回 (mean, log_std)，log_std clamp 到 [MIN, MAX]
        raise NotImplementedError

    def get_action(self, x):
        """重参数化采样 + tanh + Jacobian 修正
        返回 (action, log_prob, mean_action)
        """
        # TODO
        raise NotImplementedError


class CriticQ(nn.Module):
    """连续动作 Q 网络（DDPG/TD3/SAC）：(s,a) -> Q"""
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        # TODO: MLP(obs_dim + action_dim, 1, hidden_dims)
        raise NotImplementedError

    def forward(self, state, action):
        # TODO: cat([state, action], dim=-1) 再过网络
        raise NotImplementedError

