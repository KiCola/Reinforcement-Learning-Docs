"""
【练习】Day 53：DAgger（Dataset Aggregation）
环境：LunarLander-v2
参考论文：A Reduction of Imitation Learning and Structured Prediction
         to No-Regret Online Learning (Ross 2011)
         https://arxiv.org/abs/1011.0686
标准答案：solutions/phase6_imitation/dagger.py
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
    env_id: str            = "LunarLander-v2"
    dagger_iterations: int = 10         # DAgger 迭代轮数 N
    episodes_per_iter: int = 20         # 每轮收集的 episode 数
    bc_epochs_per_iter: int = 10        # 每轮在聚合数据集上训练的轮数
    batch_size: int        = 256
    lr: float              = 1e-3
    beta_start: float      = 1.0        # 初始专家混合比例（第1轮全用专家）
    beta_decay: float      = 0.9        # 每轮衰减
    eval_episodes: int     = 20
    seed: int              = 42


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        # TODO：与 BC 中相同的网络结构
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


# ──────────────────────────────────────────────
# TODO 1：实现 DAgger 数据收集
# 关键：以概率 β 使用专家动作，以概率 (1-β) 使用当前策略动作
# 但无论执行哪个动作，都要记录专家对当前状态的标注 a* = expert(s)
# ──────────────────────────────────────────────
def collect_dagger_data(env, policy, expert, beta: float,
                        n_episodes: int, device):
    """
    DAgger 数据收集：
    - 执行动作：以 β 概率选专家动作，(1-β) 概率选策略动作
    - 记录标注：无论执行什么，都记录 (s_t, a*_t = expert(s_t))

    返回：(obs_list, expert_action_list)

    TODO：实现此函数
    提示：
      - expert 为 None 时用随机动作模拟专家
      - beta=1.0 时等同于纯专家数据收集（DAgger 第一轮）
      - beta=0.0 时等同于纯策略执行，专家只做标注
    """
    all_obs, all_expert_actions = [], []
    # TODO
    raise NotImplementedError
    return np.array(all_obs), np.array(all_expert_actions)


# ──────────────────────────────────────────────
# TODO 2：实现单轮 BC 训练（在聚合数据集 D 上）
# ──────────────────────────────────────────────
def train_one_iter(obs_data, action_data, policy, optimizer,
                  n_epochs: int, batch_size: int, device):
    """
    在聚合数据集上训练 n_epochs 轮
    返回：平均 loss
    TODO
    """
    # TODO
    raise NotImplementedError


def evaluate_policy(env, policy, n_episodes: int, device):
    """
    TODO：评估策略平均 reward（贪心，不探索）
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg    = Config()

    env       = gym.make(cfg.env_id)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy    = PolicyNet(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    # 用 None 代替真实专家（请替换为训练好的 PPO 模型）
    expert = None

    # 聚合数据集 D（从空开始）
    all_obs     = np.zeros((0, obs_dim), dtype=np.float32)
    all_actions = np.zeros(0,            dtype=np.int64)

    beta           = cfg.beta_start
    eval_rewards   = []

    print("=== DAgger 训练 ===")
    for i in range(1, cfg.dagger_iterations + 1):
        # TODO 3：调用 collect_dagger_data，将新数据聚合到 D
        # TODO 4：在聚合数据集 D 上调用 train_one_iter
        # TODO 5：衰减 beta：beta *= beta_decay
        # TODO 6：评估当前策略
        raise NotImplementedError

        print(f"Iter {i:2d} | Dataset: {len(all_obs):5d} | "
              f"β: {beta:.3f} | Reward: {eval_rewards[-1]:.2f}")

    env.close()

    plt.figure(figsize=(8, 4))
    plt.plot(eval_rewards, marker='o', color='steelblue', label='DAgger')
    plt.xlabel('DAgger Iteration')
    plt.ylabel('Mean Reward')
    plt.title(f'DAgger on {cfg.env_id}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('dagger_curve.png', dpi=150)
    plt.show()

"""
思考题：
1. DAgger 相比 BC 的核心改进是什么？体现在代码的哪一行？
2. β 的衰减策略对训练有什么影响？β 保持为 1.0 会怎样？
3. DAgger 的主要局限是什么？（提示：需要专家在线查询）
4. 如果没有专家在线，如何用 IRL/GAIL 替代？→ 见 Day 55-56
"""

