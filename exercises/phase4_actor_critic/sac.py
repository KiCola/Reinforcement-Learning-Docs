"""
【练习】Day 28：SAC (Soft Actor-Critic) 从零实现
环境：Pendulum-v1
参考论文：SAC v2 https://arxiv.org/abs/1812.05905
CleanRL 参考：https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
标准答案：solutions/phase4_actor_critic/sac.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Config:
    env_id: str          = "Pendulum-v1"
    total_timesteps: int = 100_000
    lr: float            = 3e-4
    buffer_size: int     = 100_000
    batch_size: int      = 256
    gamma: float         = 0.99
    tau: float           = 0.005
    alpha: float         = 0.2
    auto_alpha: bool     = True
    start_learning: int  = 5_000
    update_freq: int     = 1
    seed: int            = 42


# TODO 1：实现 SoftQNetwork
# 输入：(state, action) 拼接；输出：标量 Q(s,a)
# 建议：256-256 MLP，ReLU 激活
class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, state, action):
        # TODO: cat([state, action], dim=-1) 再过网络
        raise NotImplementedError


# TODO 2：实现随机策略 Actor
# forward 输出 (mean, log_std)
# get_action 使用重参数化采样 + tanh 压缩 + Jacobian 修正
# 修正：log_prob -= log(1 - tanh(x)^2 + 1e-6)
class Actor(nn.Module):
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # TODO: 共享特征层 + mean_layer + log_std_layer
        raise NotImplementedError

    def forward(self, x):
        # TODO: 返回 (mean, log_std)，log_std 需 clamp
        raise NotImplementedError

    def get_action(self, x):
        """
        1. rsample() 得到 x_t
        2. y_t = tanh(x_t)
        3. log_prob 做 Jacobian 修正后沿 action_dim 求和
        返回：(y_t, log_prob, tanh(mean))
        """
        # TODO
        raise NotImplementedError


# TODO 3：实现经验回放池（连续动作版本）
# actions 是 float 向量，rewards/dones shape 为 (N,1)
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.device   = device
        self.ptr = self.size = 0
        # TODO: 初始化数组
        raise NotImplementedError

    def push(self, obs, action, reward, next_obs, done):
        # TODO
        raise NotImplementedError

    def sample(self, batch_size):
        # TODO: 返回 5 个 Tensor
        raise NotImplementedError

    def __len__(self): return self.size


def train_sac(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: {cfg.env_id}")

    env          = gym.make(cfg.env_id)
    obs_dim      = env.observation_space.shape[0]
    action_dim   = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    # TODO 4：创建 actor / qf1 / qf2 / qf1_tgt / qf2_tgt
    # 目标网络初始权重与 qf1/qf2 相同
    raise NotImplementedError

    actor_opt = optim.Adam(actor.parameters(), lr=cfg.lr)
    qf_opt    = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.lr)

    # TODO 5：自动温度调节设置
    # target_entropy = -action_dim
    # log_alpha 可学习，alpha = log_alpha.exp()
    if cfg.auto_alpha:
        target_entropy = -action_dim
        raise NotImplementedError  # 初始化 log_alpha 和 alpha_opt
    else:
        alpha = cfg.alpha

    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, action_dim, device)
    obs, _ = env.reset(seed=cfg.seed)
    episode_rewards, ep_reward, episode_count = [], 0.0, 0

    for step in range(1, cfg.total_timesteps + 1):
        if len(buffer) < cfg.start_learning:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                # TODO: actor.get_action() 采样，乘以 action_scale
                action = None  # TODO

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_reward += reward

        if done:
            episode_count += 1
            episode_rewards.append(ep_reward)
            if episode_count % 20 == 0:
                print(f"Step {step:7d} | Ep {episode_count:4d} | "
                      f"Reward: {ep_reward:8.2f} | "
                      f"Mean50: {np.mean(episode_rewards[-50:]):8.2f}")
            ep_reward = 0.0
            obs, _ = env.reset()

        if len(buffer) < cfg.start_learning:
            continue

        b_obs, b_act, b_rew, b_next, b_done = buffer.sample(cfg.batch_size)

        # TODO 6：更新 Q 网络
        # td_target = r + γ*(1-done)*(min(Q1_tgt,Q2_tgt)(s',a') - α*next_logp)
        # qf_loss = MSE(Q1,td_target) + MSE(Q2,td_target)
        raise NotImplementedError

        # TODO 7：更新 Actor
        # actor_loss = mean(α*log_prob - min(Q1,Q2))
        raise NotImplementedError

        # TODO 8：自动调节 α
        # alpha_loss = mean(-log_alpha * (log_prob + target_entropy).detach())
        if cfg.auto_alpha:
            raise NotImplementedError

        # TODO 9：软更新目标网络
        # θ_tgt = τ*θ + (1-τ)*θ_tgt
        raise NotImplementedError

    env.close()
    return episode_rewards


if __name__ == "__main__":
    cfg     = Config()
    rewards = train_sac(cfg)

    def smooth(r, w=20):
        return np.convolve(r, np.ones(w) / w, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(rewards,        alpha=0.3, color='darkorange')
    plt.plot(smooth(rewards),           color='darkorange', label='SAC')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'SAC on {cfg.env_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sac_curve.png', dpi=150)
    plt.show()

"""
思考题：
1. 最大熵目标为什么能提升探索效率和 sample efficiency？
2. Twin Q 网络的作用是什么？和 TD3 有何异同？
3. target_entropy 为什么设为 -dim(A)？
4. SAC 和 TD3 的核心区别是什么？
"""

