"""
【练习】Day 9：DQN 从零实现
环境：CartPole-v1
参考论文：Human-level control through deep reinforcement learning (Nature 2015)
https://www.nature.com/articles/nature14236
CleanRL 参考：https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
标准答案：solutions/phase2_dqn/dqn.py
"""
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


@dataclass
class Config:
    env_id: str          = "CartPole-v1"
    total_timesteps: int = 50_000
    learning_rate: float = 2.5e-4
    buffer_size: int     = 10_000
    gamma: float         = 0.99
    tau: float           = 1.0          # 目标网络更新系数（1.0=硬更新）
    target_update_freq: int = 500
    batch_size: int      = 128
    start_learning: int  = 1_000
    eps_start: float     = 1.0
    eps_end: float       = 0.05
    eps_decay: int       = 10_000
    seed: int            = 42


# ──────────────────────────────────────────────
# TODO 1：实现 Q 网络
# 输入：obs_dim 维状态向量
# 输出：n_actions 维 Q 值向量（每个动作对应一个 Q 值）
# 建议结构：Linear(obs_dim,128)->ReLU->Linear(128,128)->ReLU->Linear(128,n_actions)
# ──────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        # TODO: 定义网络层
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 前向传播
        raise NotImplementedError


# ──────────────────────────────────────────────
# TODO 2：实现经验回放池
# 功能：存储 (obs, action, reward, next_obs, done) 转移，随机采样 minibatch
# ──────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self.ptr      = 0
        self.size     = 0
        # TODO: 初始化存储数组（用 np.zeros 预分配内存）
        # self.obs      = ...
        # self.actions  = ...
        # self.rewards  = ...
        # self.next_obs = ...
        # self.dones    = ...
        raise NotImplementedError

    def push(self, obs, action, reward, next_obs, done):
        # TODO: 将一条转移写入 ptr 位置，并更新 ptr 和 size
        # 提示：ptr 到达 capacity 时需要循环（环形缓冲区）
        raise NotImplementedError

    def sample(self, batch_size: int):
        # TODO: 随机采样 batch_size 条转移，返回 5 个 torch.Tensor
        raise NotImplementedError

    def __len__(self):
        return self.size


def train(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    env       = gym.make(cfg.env_id)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # TODO 3：创建在线网络 q_net 和目标网络 target_q_net
    # 目标网络初始权重与在线网络相同，且不参与梯度计算
    q_net        = None  # TODO
    target_q_net = None  # TODO

    optimizer = optim.Adam(q_net.parameters(), lr=cfg.learning_rate)
    buffer    = ReplayBuffer(cfg.buffer_size, obs_dim, device)

    obs, _          = env.reset(seed=cfg.seed)
    episode_reward  = 0.0
    episode_count   = 0
    episode_rewards = []

    for step in range(1, cfg.total_timesteps + 1):

        # TODO 4：ε-greedy 动作选择
        # eps 从 eps_start 线性衰减到 eps_end（在 eps_decay 步内）
        eps    = None  # TODO
        action = None  # TODO

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        episode_reward += reward

        if done:
            episode_count += 1
            episode_rewards.append(episode_reward)
            mean100 = np.mean(episode_rewards[-100:])
            print(f"Step {step:6d} | Ep {episode_count:4d} | "
                  f"Reward: {episode_reward:7.2f} | Mean100: {mean100:7.2f} | ε: {eps:.3f}")
            episode_reward = 0.0
            obs, _ = env.reset()

        if len(buffer) < cfg.start_learning:
            continue

        b_obs, b_act, b_rew, b_next_obs, b_done = buffer.sample(cfg.batch_size)

        # TODO 5：计算 TD target（用 target_q_net，不计算梯度）
        # 公式：td_target = r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            td_target = None  # TODO

        # TODO 6：计算当前 Q 值 Q(s, a)
        # 提示：用 .gather(1, b_act.unsqueeze(1)).squeeze(1)
        current_q = None  # TODO

        # TODO 7：MSE 损失 + 反向传播 + 梯度裁剪
        loss = None  # TODO

        # TODO 8：每隔 target_update_freq 步软更新目标网络
        # θ_target = τ·θ + (1-τ)·θ_target
        if step % cfg.target_update_freq == 0:
            pass  # TODO

    env.close()
    print(f"\n训练完成！最终 Mean100: {np.mean(episode_rewards[-100:]):.2f}")
    return episode_rewards


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cfg     = Config()
    rewards = train(cfg)
    window  = 20
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(rewards,  alpha=0.3, color='steelblue', label='Episode Reward')
    plt.plot(smoothed, color='steelblue', label=f'Smoothed ({window}-ep)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN on CartPole-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('dqn_cartpole.png', dpi=150)
    plt.show()

"""
思考题：
1. 去掉目标网络直接用 q_net 算 td_target，训练为何会不稳定？
2. buffer_size 太小/太大各有什么副作用？
3. DQN 为什么会对 Q 值过估计？Double DQN 如何解决？
"""

