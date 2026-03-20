"""
【练习】Day 56：GAIL（Generative Adversarial Imitation Learning）
环境：LunarLander-v2
参考论文：Generative Adversarial Imitation Learning (Ho & Ermon 2016)
         https://arxiv.org/abs/1606.03476
标准答案：solutions/phase6_imitation/gail.py
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
    env_id: str          = "LunarLander-v2"
    total_updates: int   = 300          # 总更新轮数
    expert_episodes: int = 50           # 专家轨迹数量
    rollout_steps: int   = 1024         # 每轮策略收集步数
    disc_epochs: int     = 5            # 每轮判别器更新次数
    ppo_epochs: int      = 4            # 每轮 PPO 更新次数
    batch_size: int      = 128
    lr_disc: float       = 1e-3
    lr_policy: float     = 3e-4
    gamma: float         = 0.99
    gae_lambda: float    = 0.95
    clip_eps: float      = 0.2
    seed: int            = 42


# ──────────────────────────────────────────────
# TODO 1：实现判别器 Discriminator
# 输入：(obs, action_onehot) 拼接向量
# 输出：标量概率 D(s,a) ∈ (0,1)，表示"是专家"的概率
# 建议：64-64 MLP + Sigmoid
# ──────────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        # TODO：输入维度 = obs_dim + n_actions（action 做 one-hot）
        raise NotImplementedError

    def forward(self, obs, action_onehot):
        # TODO：cat([obs, action_onehot], dim=-1) 再过网络
        raise NotImplementedError


# ──────────────────────────────────────────────
# TODO 2：实现 Actor-Critic（与 PPO 中相同）
# ──────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x):
        # TODO：返回 (logits, value)
        raise NotImplementedError

    def get_action(self, x):
        # TODO：返回 (action, log_prob, entropy, value)
        raise NotImplementedError

    def evaluate_actions(self, x, actions):
        # TODO：返回 (log_prob, entropy, value)
        raise NotImplementedError


def get_onehot(actions, n_actions, device):
    """将整数动作转为 one-hot 向量"""
    one_hot = torch.zeros(len(actions), n_actions, device=device)
    one_hot.scatter_(1, actions.unsqueeze(1), 1)
    return one_hot


# ──────────────────────────────────────────────
# TODO 3：实现判别器训练步
# 目标：区分专家轨迹（标签 1）和策略轨迹（标签 0）
# 损失：BCE loss
# D 越好 → 能越精准区分专家与策略
# ──────────────────────────────────────────────
def train_discriminator(disc, disc_opt, expert_obs, expert_acts,
                        policy_obs, policy_acts, n_actions,
                        n_epochs, batch_size, device):
    """
    TODO：
    1. 将专家数据标签设为 1，策略数据标签设为 0
    2. 用 BCE loss 训练判别器 n_epochs 轮
    3. 返回平均 disc_loss
    """
    # TODO
    raise NotImplementedError


def train_gail(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: {cfg.env_id}")

    env       = gym.make(cfg.env_id)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model    = ActorCritic(obs_dim, n_actions).to(device)
    disc     = Discriminator(obs_dim, n_actions).to(device)
    pol_opt  = optim.Adam(model.parameters(), lr=cfg.lr_policy, eps=1e-5)
    disc_opt = optim.Adam(disc.parameters(),  lr=cfg.lr_disc)

    # ── 加载专家数据（请替换为真实专家轨迹）──
    # 这里用随机数据占位，实际应从 BC 练习中收集的专家轨迹加载
    print("[警告] 使用随机数据作为专家轨迹，请替换为真实专家数据")
    expert_obs  = torch.randn(cfg.expert_episodes * 50, obs_dim, device=device)
    expert_acts = torch.randint(0, n_actions, (cfg.expert_episodes * 50,), device=device)

    obs, _        = env.reset(seed=cfg.seed)
    episode_rewards = []
    ep_reward     = 0.0

    obs_buf  = np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32)
    act_buf  = np.zeros(cfg.rollout_steps,            dtype=np.int64)
    rew_buf  = np.zeros(cfg.rollout_steps,            dtype=np.float32)
    done_buf = np.zeros(cfg.rollout_steps,            dtype=np.float32)
    val_buf  = np.zeros(cfg.rollout_steps,            dtype=np.float32)
    logp_buf = np.zeros(cfg.rollout_steps,            dtype=np.float32)

    for update in range(1, cfg.total_updates + 1):

        # ── TODO 6：收集策略轨迹（与 PPO rollout 相同）──
        for step in range(cfg.rollout_steps):
            obs_buf[step] = obs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action(obs_t)
            act_buf[step]  = action.item()
            logp_buf[step] = log_prob.item()
            val_buf[step]  = value.item()
            obs, _, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            done_buf[step] = float(done)
            ep_reward += _  # 注意：这里用环境真实 reward 仅用于监控
            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()

        # ── TODO 7：用判别器计算模仿奖励，替换 rew_buf ──
        pol_obs_t  = torch.tensor(obs_buf,  dtype=torch.float32, device=device)
        pol_acts_t = torch.tensor(act_buf,  dtype=torch.long,    device=device)
        # rew_buf = compute_imitation_reward(disc, pol_obs_t, pol_acts_t, n_actions, device)
        raise NotImplementedError  # 替换上一行注释，删除此行

        # ── TODO 8：计算 GAE ──
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, _, next_val = model.get_action(obs_t)
        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_val.item(), cfg.gamma, cfg.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── TODO 9：训练判别器 ──
        disc_loss = train_discriminator(
            disc, disc_opt,
            expert_obs, expert_acts,
            pol_obs_t, pol_acts_t,
            n_actions, cfg.disc_epochs, cfg.batch_size, device
        )

        # ── TODO 10：PPO 更新策略 ──
        ppo_update(
            model, pol_opt,
            pol_obs_t, pol_acts_t,
            torch.tensor(logp_buf, device=device),
            torch.tensor(advantages, device=device),
            torch.tensor(returns,    device=device),
            cfg, device
        )

        if update % 20 == 0 and episode_rewards:
            mean_r = np.mean(episode_rewards[-20:])
            print(f"Update {update:4d} | Mean Reward: {mean_r:8.2f} | Disc Loss: {disc_loss:.4f}")

    env.close()
    return episode_rewards


if __name__ == "__main__":
    cfg     = Config()
    rewards = train_gail(cfg)

    def smooth(r, w=10):
        return np.convolve(r, np.ones(w) / w, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(rewards,        alpha=0.3, color='purple')
    plt.plot(smooth(rewards),           color='purple', label='GAIL')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (env)')
    plt.title(f'GAIL on {cfg.env_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gail_curve.png', dpi=150)
    plt.show()

"""
思考题：
1. GAIL 中判别器的输入是 (s,a)，为什么不只输入 s？
2. 模仿奖励 r = -log(1 - D(s,a)) 的直觉是什么？
   当 D(s,a) → 1（很像专家）时，r 如何变化？
3. GAIL 和 GAN 的训练不稳定问题有何相似之处？如何缓解？
4. AIRL 为何比 GAIL 的奖励函数更适合 sim-to-real 迁移？
5. 对比 BC、DAgger、GAIL 三种方法：
   - 哪种对专家数据量最不敏感？
   - 哪种不需要专家在线？
   - 哪种最容易训练？
"""


# ──────────────────────────────────────────────
# TODO 4：用判别器输出计算模仿奖励
# 公式：r_imitation = -log(1 - D(s,a))
# 直觉：D(s,a) 越接近 1（越像专家），奖励越高
# ──────────────────────────────────────────────
def compute_imitation_reward(disc, obs, actions, n_actions, device):
    """
    TODO：计算每个 (s,a) 对的模仿奖励
    注意：不需要梯度（用 torch.no_grad）
    """
    # TODO
    raise NotImplementedError


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """GAE 计算（与 PPO 相同）"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nv   = next_value if t == T - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * mask - values[t]
        gae   = delta + gamma * lam * mask * gae
        advantages[t] = gae
    return advantages, advantages + values


# ──────────────────────────────────────────────
# TODO 5：实现 PPO 更新（与 Day 23 相同）
# 区别：reward 来自判别器而非环境
# ──────────────────────────────────────────────
def ppo_update(model, optimizer, obs_buf, act_buf, logp_buf,
               adv_buf, ret_buf, cfg: Config, device):
    """
    TODO：标准 PPO-Clip 更新
    """
    # TODO
    raise NotImplementedError

