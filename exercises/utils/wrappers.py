"""
常用 Gymnasium 环境 Wrapper — 已完成，直接使用
"""
import numpy as np
import gymnasium as gym
from collections import deque


class NormalizeObservation(gym.ObservationWrapper):
    """在线归一化观测（running mean/std）"""
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.mean  = np.zeros(env.observation_space.shape)
        self.var   = np.ones(env.observation_space.shape)
        self.count = epsilon

    def observation(self, obs):
        self.count += 1
        last_mean   = self.mean.copy()
        self.mean  += (obs - self.mean) / self.count
        self.var   += (obs - last_mean) * (obs - self.mean)
        std = np.sqrt(self.var / self.count + self.epsilon)
        return (obs - self.mean) / std


class FrameStack(gym.Wrapper):
    """将最近 n_frames 帧叠加作为状态（Atari 常用）"""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames   = deque(maxlen=n_frames)
        obs_shape     = env.observation_space.shape
        new_shape     = (obs_shape[0] * n_frames,) + obs_shape[1:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)


class RecordEpisodeStats(gym.Wrapper):
    """记录 episode 统计信息到 info"""
    def __init__(self, env):
        super().__init__(env)
        self.ep_reward = 0.0
        self.ep_len    = 0

    def reset(self, **kwargs):
        self.ep_reward = 0.0
        self.ep_len    = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.ep_reward += reward
        self.ep_len    += 1
        if terminated or truncated:
            info["episode"] = {"r": self.ep_reward, "l": self.ep_len}
        return obs, reward, terminated, truncated, info

