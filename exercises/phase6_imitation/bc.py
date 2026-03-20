"""
【练习】Day 52：行为克隆（Behavior Cloning, BC）
环境：LunarLander-v2（离散）
标准流程：先用 PPO 生成专家数据，再用 BC 从数据中学习策略
参考论文：A Reduction of Imitation Learning (Ross 2010)
         https://arxiv.org/abs/1011.0686
标准答案：solutions/phase6_imitation/bc.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Config:
    env_id: str           = "LunarLander-v2"
    expert_episodes: int  = 200       # 收集多少条专家轨迹
    bc_epochs: int        = 50        # BC 训练轮数
    batch_size: int       = 256
    lr: float             = 1e-3
    eval_episodes: int    = 20        # 评估轮数
    seed: int             = 42


# ──────────────────────────────────────────────
# TODO 1：实现策略网络（与 REINFORCE 中的 PolicyNet 相同）
# 输入：obs_dim；输出：n_actions 维 logits（不加 Softmax）
# ──────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x):
        # TODO：返回 logits
        raise NotImplementedError


def load_expert_policy(env_id: str, device: torch.device):
    """
    加载预训练的专家策略。
    实际使用时替换为你自己训练好的 PPO/SAC checkpoint。
    这里提供一个随机策略作为占位，请先用 PPO 训练一个专家模型后再替换。
    """
    # TODO（可选）：加载你在 Phase 4 训练好的 PPO checkpoint
    # 示例：
    # from phase4_actor_critic.ppo import ActorCritic, Config as PPOConfig
    # model = ActorCritic(obs_dim, n_actions).to(device)
    # model.load_state_dict(torch.load('checkpoints/ppo_lunarlander.pth'))
    # return model
    print("[警告] 使用随机策略作为专家，请替换为训练好的 PPO 模型")
    return None


def collect_expert_data(env, expert_policy, n_episodes: int, device):
    """
    用专家策略收集轨迹数据。
    返回：(obs_list, action_list) — numpy arrays

    TODO：实现数据收集循环
    1. 重置环境
    2. 用专家策略选择动作（若 expert_policy 为 None，用随机动作）
    3. 将 (obs, action) 存入列表
    4. episode 结束后继续直到收集满 n_episodes
    """
    all_obs, all_actions = [], []
    # TODO
    raise NotImplementedError
    return np.array(all_obs), np.array(all_actions)


def train_bc(obs_data, action_data, policy, cfg: Config, device):
    """
    行为克隆训练：将 (obs, action) 对视为监督学习数据集
    损失函数：交叉熵 L = -E_{(s,a)~D}[log π_θ(a|s)]

    TODO：
    1. 构建 TensorDataset 和 DataLoader
    2. 训练 cfg.bc_epochs 轮
    3. 每轮记录平均 loss
    4. 返回 loss 历史
    """
    # TODO
    raise NotImplementedError


def evaluate_policy(env, policy, n_episodes: int, device):
    """
    评估策略的平均 episode reward。
    TODO：实现评估循环（贪心选动作，不探索）
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

    # Step 1：收集专家数据
    print("=== Step 1: 收集专家数据 ===")
    expert = load_expert_policy(cfg.env_id, device)
    obs_data, action_data = collect_expert_data(
        env, expert, cfg.expert_episodes, device
    )
    print(f"收集完成：{len(obs_data)} 条 (obs, action) 对")

    # Step 2：行为克隆训练
    print("\n=== Step 2: 行为克隆训练 ===")
    policy    = PolicyNet(obs_dim, n_actions).to(device)
    loss_hist = train_bc(obs_data, action_data, policy, cfg, device)

    # Step 3：评估
    print("\n=== Step 3: 评估 BC 策略 ===")
    mean_reward = evaluate_policy(env, policy, cfg.eval_episodes, device)
    print(f"BC 策略平均 Reward: {mean_reward:.2f}")

    env.close()

    # 绘制训练曲线
    plt.figure(figsize=(8, 4))
    plt.plot(loss_hist, color='steelblue')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Behavior Cloning Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bc_loss.png', dpi=150)
    plt.show()

"""
思考题：
1. 为什么 BC 在简单任务上能工作，但在长程任务上会失败？
   （提示：Covariate Shift / 误差累积）
2. 专家数据量对 BC 性能的影响是什么？
3. BC 和监督学习有什么根本区别？
4. 如何改进 BC 以减轻分布偏移问题？→ 见 Day 53 (DAgger)
"""

