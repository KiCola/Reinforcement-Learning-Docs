"""
[标准答案] Day 28：SAC 完整实现
参考：https://arxiv.org/abs/1812.05905
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


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                  nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class Actor(nn.Module):
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

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


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.device   = device
        self.ptr = self.size = 0
        self.obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),          dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx],      device=self.device),
            torch.tensor(self.actions[idx],  device=self.device),
            torch.tensor(self.rewards[idx],  device=self.device),
            torch.tensor(self.next_obs[idx], device=self.device),
            torch.tensor(self.dones[idx],    device=self.device),
        )
    def __len__(self): return self.size


def train_sac(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(cfg.env_id)
    obs_dim      = env.observation_space.shape[0]
    action_dim   = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    actor   = Actor(obs_dim, action_dim).to(device)
    qf1     = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2     = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_tgt = SoftQNetwork(obs_dim, action_dim).to(device)
    qf2_tgt = SoftQNetwork(obs_dim, action_dim).to(device)
    qf1_tgt.load_state_dict(qf1.state_dict())
    qf2_tgt.load_state_dict(qf2.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=cfg.lr)
    qf_opt    = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.lr)

    if cfg.auto_alpha:
        target_entropy = -action_dim
        log_alpha      = torch.zeros(1, requires_grad=True, device=device)
        alpha_opt      = optim.Adam([log_alpha], lr=cfg.lr)
        alpha          = log_alpha.exp().item()
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
                obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _ = actor.get_action(obs_t)
                action = action.squeeze(0).cpu().numpy() * action_scale

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_reward += reward

        if done:
            episode_count += 1
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()

        if len(buffer) < cfg.start_learning:
            continue

        b_obs, b_act, b_rew, b_next, b_done = buffer.sample(cfg.batch_size)
        with torch.no_grad():
            next_act, next_logp, _ = actor.get_action(b_next)
            next_act  = next_act * action_scale
            min_q     = torch.min(qf1_tgt(b_next, next_act), qf2_tgt(b_next, next_act)) - alpha * next_logp
            td_target = b_rew + cfg.gamma * (1 - b_done) * min_q

        qf_loss = nn.functional.mse_loss(qf1(b_obs, b_act), td_target) + \
                  nn.functional.mse_loss(qf2(b_obs, b_act), td_target)
        qf_opt.zero_grad(); qf_loss.backward(); qf_opt.step()

        curr_act, log_pi, _ = actor.get_action(b_obs)
        curr_act = curr_act * action_scale
        actor_loss = (alpha * log_pi - torch.min(qf1(b_obs, curr_act), qf2(b_obs, curr_act))).mean()
        actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

        if cfg.auto_alpha:
            with torch.no_grad():
                _, log_pi, _ = actor.get_action(b_obs)
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
            alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
            alpha = log_alpha.exp().item()

        for p, tp in zip(qf1.parameters(), qf1_tgt.parameters()):
            tp.data.copy_(cfg.tau * p.data + (1 - cfg.tau) * tp.data)
        for p, tp in zip(qf2.parameters(), qf2_tgt.parameters()):
            tp.data.copy_(cfg.tau * p.data + (1 - cfg.tau) * tp.data)

    env.close()
    return episode_rewards

