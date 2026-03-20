"""
【练习】Day 16：REINFORCE 与带 Baseline 的 REINFORCE
环境：CartPole-v1
参考论文：Policy Gradient Methods for RL with Function Approximation (Sutton 2000)
参考书：《深度强化学习》王树森 第6章
Easy-RL Chapter 9: https://datawhalechina.github.io/easy-rl/#/chapter9/chapter9
标准答案：solutions/phase3_policy_gradient/reinforce.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# TODO 1：实现策略网络 PolicyNet
# 输入：obs_dim 维状态
# 输出：n_actions 维动作概率分布（末层用 Softmax）
# ──────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        # TODO: 定义网络层（建议 Linear->ReLU->Linear->Softmax）
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError


# ──────────────────────────────────────────────
# TODO 2：实现价值网络 ValueNet（用作 Baseline）
# 输入：obs_dim 维状态
# 输出：标量 V(s)
# ──────────────────────────────────────────────
class ValueNet(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x):
        # TODO：注意 squeeze 掉最后一维，输出形状 (batch,)
        raise NotImplementedError


def reinforce(env, n_episodes=800, gamma=0.99, lr=1e-3, use_baseline=False):
    """
    REINFORCE 算法（Monte Carlo 策略梯度）

    核心公式：∇J(θ) ≈ Σ_t ∇log π_θ(a_t|s_t) · G_t
    其中 G_t = Σ_{k=t}^{T} γ^{k-t} r_k（折扣回报）

    use_baseline=True 时：用 A_t = G_t - V(s_t) 代替 G_t，降低方差
    """
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy     = PolicyNet(obs_dim, n_actions).to(device)
    policy_opt = optim.Adam(policy.parameters(), lr=lr)

    # TODO 3：若 use_baseline=True，创建 baseline 网络和对应优化器
    baseline     = None  # TODO
    baseline_opt = None  # TODO

    all_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        log_probs, rewards, states = [], [], []
        done = False

        # ── 采集完整轨迹 ──
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            # TODO 4：用策略网络得到动作概率，创建 Categorical 分布，采样动作
            # 保存 log_prob 和 obs_t 到列表
            raise NotImplementedError

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        all_rewards.append(sum(rewards))
        T = len(rewards)

        # TODO 5：从后往前计算折扣回报 G_t
        # G[T-1] = r[T-1]
        # G[t]   = r[t] + gamma * G[t+1]
        G   = np.zeros(T, dtype=np.float32)
        G_t = None  # TODO: 转为 torch.Tensor

        # TODO 6：计算 advantage
        # use_baseline=True：advantage = (G_t - V(s)).detach()，并更新 baseline
        # use_baseline=False：advantage = G_t
        advantage = None  # TODO

        # TODO 7：计算策略梯度 loss 并更新
        # loss = -mean( log_prob(a_t) * advantage_t )
        raise NotImplementedError

        if (ep + 1) % 50 == 0:
            mean100 = np.mean(all_rewards[-100:])
            label   = "w/ baseline" if use_baseline else "no baseline"
            print(f"[REINFORCE {label}] Ep {ep+1:4d} | Mean100: {mean100:.2f}")

    return all_rewards


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    print("=== 训练 REINFORCE（无 baseline）===")
    rewards_no_bl = reinforce(env, n_episodes=800, use_baseline=False)

    print("\n=== 训练 REINFORCE（有 baseline）===")
    rewards_bl = reinforce(env, n_episodes=800, use_baseline=True)
    env.close()

    def smooth(r, w=20):
        return np.convolve(r, np.ones(w) / w, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(smooth(rewards_no_bl), color='tomato',    label='REINFORCE (no baseline)')
    plt.plot(smooth(rewards_bl),    color='steelblue', label='REINFORCE (w/ V baseline)')
    plt.axhline(y=475, color='gray', linestyle='--', alpha=0.5, label='Solved (475)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (smoothed)')
    plt.title('REINFORCE vs REINFORCE w/ Baseline on CartPole-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reinforce_comparison.png', dpi=150)
    plt.show()

"""
思考题：
1. 为什么 Baseline 能降低方差但不引入偏差？
   提示：E_a[∇log π(a|s) · b(s)] = 0
2. REINFORCE 需要等整条轨迹结束才能更新，这带来了什么问题？
3. 如何把 REINFORCE 改造成 Actor-Critic？
"""
