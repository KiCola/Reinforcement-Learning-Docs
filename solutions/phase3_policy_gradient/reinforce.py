"""
[标准答案] Day 16：REINFORCE 与带 Baseline 的 REINFORCE
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def reinforce(env, n_episodes=800, gamma=0.99, lr=1e-3, use_baseline=False):
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy     = PolicyNet(obs_dim, n_actions).to(device)
    policy_opt = optim.Adam(policy.parameters(), lr=lr)
    baseline     = ValueNet(obs_dim).to(device) if use_baseline else None
    baseline_opt = optim.Adam(baseline.parameters(), lr=lr) if use_baseline else None
    all_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        log_probs, rewards, states = [], [], []
        done = False
        while not done:
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device)
            probs  = policy(obs_t)
            dist   = Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            states.append(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        all_rewards.append(sum(rewards))
        T = len(rewards)
        G = np.zeros(T, dtype=np.float32)
        G[-1] = rewards[-1]
        for t in reversed(range(T - 1)):
            G[t] = rewards[t] + gamma * G[t + 1]
        G_t = torch.tensor(G, dtype=torch.float32, device=device)

        if use_baseline:
            states_t  = torch.stack(states)
            V_s       = baseline(states_t)
            advantage = (G_t - V_s).detach()
            baseline_loss = nn.functional.mse_loss(V_s, G_t)
            baseline_opt.zero_grad()
            baseline_loss.backward()
            baseline_opt.step()
        else:
            advantage = G_t

        log_probs_t = torch.stack(log_probs)
        policy_loss = -(log_probs_t * advantage).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        if (ep + 1) % 50 == 0:
            mean100 = np.mean(all_rewards[-100:])
            label   = "w/ baseline" if use_baseline else "no baseline"
            print(f"[REINFORCE {label}] Ep {ep+1:4d} | Mean100: {mean100:.2f}")

    return all_rewards

