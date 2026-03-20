"""
Day 9：DQN 从零实现
环境：CartPole-v1
参考论文：Human-level control through deep reinforcement learning (Nature 2015)
https://www.nature.com/articles/nature14236

CleanRL 参考：https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
"""
import random
from collections import deque
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
    tau: float           = 1.0
    target_update_freq: int = 500
    batch_size: int      = 128
    start_learning: int  = 1_000
    eps_start: float     = 1.0
    eps_end: float       = 0.05
    eps_decay: int       = 10_000
    seed: int            = 42


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self.ptr      = 0
        self.size     = 0
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx],      device=self.device),
            torch.tensor(self.actions[idx],  device=self.device),
            torch.tensor(self.rewards[idx],  device=self.device),
            torch.tensor(self.next_obs[idx], device=self.device),
            torch.tensor(self.dones[idx],    device=self.device),
        )

    def __len__(self):
        return self.size


def train(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    env = gym.make(cfg.env_id)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net        = QNetwork(obs_dim, n_actions).to(device)
    target_q_net = QNetwork(obs_dim, n_actions).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    target_q_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=cfg.learning_rate)
    buffer    = ReplayBuffer(cfg.buffer_size, obs_dim, device)

    obs, _ = env.reset(seed=cfg.seed)
    episode_reward = 0.0
    episode_count  = 0
    episode_rewards = []

    for step in range(1, cfg.total_timesteps + 1):
        eps = max(cfg.eps_end,
                  cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / cfg.eps_decay)
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = q_net(torch.tensor(obs, device=device).unsqueeze(0))
                action = int(q_vals.argmax(dim=1).item())

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

        with torch.no_grad():
            max_next_q = target_q_net(b_next_obs).max(dim=1).values
            td_target  = b_rew + cfg.gamma * max_next_q * (1.0 - b_done)

        current_q = q_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(current_q, td_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
        optimizer.step()

        if step % cfg.target_update_freq == 0:
            for param, target_param in zip(q_net.parameters(), target_q_net.parameters()):
                target_param.data.copy_(
                    cfg.tau * param.data + (1 - cfg.tau) * target_param.data
                )

    env.close()
    print(f"\n训练完成！最终 Mean100 Reward: {np.mean(episode_rewards[-100:]):.2f}")
    return episode_rewards


if __name__ == "__main__":
    cfg = Config()
    rewards = train(cfg)

