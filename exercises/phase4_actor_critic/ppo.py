"""
【练习】Day 23：PPO (Proximal Policy Optimization) 从零实现
环境：LunarLander-v2
参考论文：https://arxiv.org/abs/1707.06347
37个实现细节：https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
标准答案：solutions/phase4_actor_critic/ppo.py
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


# ──────────────────────────────────────────────
# TODO 1：实现 ActorCritic 网络
# 共享底层 MLP（Tanh激活）+ Actor头（logits）+ Critic头（V(s)）
# 建议使用正交初始化：Actor头 std=0.01，Critic头 std=1.0，共享层 std=sqrt(2)
# ──────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        # TODO: 定义 shared / actor_head / critic_head
        raise NotImplementedError

    @staticmethod
    def _layer(layer, std=np.sqrt(2), bias=0.0):
        """正交初始化"""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias)
        return layer

    def forward(self, x):
        # TODO: 返回 (logits, value)
        raise NotImplementedError

    def get_action(self, x):
        """采样动作，返回 (action, log_prob, entropy, value)"""
        # TODO: Categorical(logits=logits).sample()
        raise NotImplementedError

    def evaluate_actions(self, x, actions):
        """评估给定动作，返回 (log_prob, entropy, value)"""
        # TODO
        raise NotImplementedError


# ──────────────────────────────────────────────
# TODO 2：实现 GAE（广义优势估计）
# δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
# A_t = δ_t + γλ·A_{t+1}  （从 t=T-1 往前递推）
# 返回 (advantages, returns)，其中 returns = advantages + values
# ──────────────────────────────────────────────
def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    # TODO: 从后往前递推
    raise NotImplementedError
    returns = advantages + values
    return advantages, returns


def train_ppo(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: {cfg.env_id}")

    env       = gym.make(cfg.env_id)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # TODO 3：创建 ActorCritic 和 Adam 优化器（eps=1e-5）
    model     = None  # TODO
    optimizer = None  # TODO

    obs_buf  = np.zeros((cfg.n_steps, obs_dim), dtype=np.float32)
    act_buf  = np.zeros(cfg.n_steps,            dtype=np.int64)
    rew_buf  = np.zeros(cfg.n_steps,            dtype=np.float32)
    done_buf = np.zeros(cfg.n_steps,            dtype=np.float32)
    val_buf  = np.zeros(cfg.n_steps,            dtype=np.float32)
    logp_buf = np.zeros(cfg.n_steps,            dtype=np.float32)

    obs, _         = env.reset(seed=cfg.seed)
    ep_reward      = 0.0
    episode_count  = 0
    episode_rewards = []
    n_updates = cfg.total_timesteps // cfg.n_steps

    for update in range(1, n_updates + 1):

        # TODO 4：收集 n_steps 步 rollout
        # 每步用 model.get_action()（no_grad），存入各 buf
        # episode 结束时记录奖励并 reset
        for step in range(cfg.n_steps):
            raise NotImplementedError

        # TODO 5：计算 GAE 并归一化 advantage
        # next_value 由 model.get_action(last_obs) 得到
        # 归一化：adv = (adv - mean) / (std + 1e-8)
        raise NotImplementedError

        b_obs  = torch.tensor(obs_buf,    device=device)
        b_act  = torch.tensor(act_buf,    device=device)
        b_adv  = torch.tensor(advantages, device=device)
        b_ret  = torch.tensor(returns,    device=device)
        b_logp = torch.tensor(logp_buf,   device=device)

        # TODO 6：n_epochs 轮 minibatch 更新
        # PPO-Clip loss + Value loss + Entropy bonus
        # 梯度裁剪 max_grad_norm
        indices = np.arange(cfg.n_steps)
        for epoch in range(cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, cfg.n_steps, cfg.batch_size):
                mb_idx = indices[start: start + cfg.batch_size]
                raise NotImplementedError

    env.close()
    print(f"\n训练完成！最终 Mean50: {np.mean(episode_rewards[-50:]):.2f}")
    return episode_rewards


if __name__ == "__main__":
    cfg     = Config()
    rewards = train_ppo(cfg)

    def smooth(r, w=20):
        return np.convolve(r, np.ones(w) / w, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(rewards,        alpha=0.3, color='steelblue')
    plt.plot(smooth(rewards),           color='steelblue', label='PPO')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'PPO on {cfg.env_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ppo_curve.png', dpi=150)
    plt.show()

"""
思考题：
1. clip_eps=0.2 的含义？ratio 超出 [0.8,1.2] 后会发生什么？
2. 为什么要对 advantage 归一化？
3. n_epochs>1 时同一批数据被重复使用，PPO 为何仍是 on-policy 方法？
4. 去掉 entropy bonus，策略探索行为如何变化？
"""

