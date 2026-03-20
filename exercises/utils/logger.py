"""
训练日志工具 — 封装 TensorBoard，已完成，直接使用
"""
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str, algo_name: str = "RL"):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.algo_name = algo_name
        self.start_time = time.time()
        self.ep_rewards = deque(maxlen=100)
        self.global_step = 0

    def log_episode(self, episode: int, ep_reward: float, ep_len: int):
        self.ep_rewards.append(ep_reward)
        mean_reward = sum(self.ep_rewards) / len(self.ep_rewards)
        self.writer.add_scalar("charts/episode_reward", ep_reward, episode)
        self.writer.add_scalar("charts/mean_reward_100ep", mean_reward, episode)
        self.writer.add_scalar("charts/episode_length", ep_len, episode)
        elapsed = time.time() - self.start_time
        print(f"[{self.algo_name}] Ep {episode:5d} | "
              f"Reward: {ep_reward:8.2f} | "
              f"Mean100: {mean_reward:8.2f} | "
              f"Time: {elapsed:.0f}s")

    def log_train(self, tag: str, value: float, step: int = None):
        step = step if step is not None else self.global_step
        self.writer.add_scalar(f"train/{tag}", value, step)

    def step(self):
        self.global_step += 1

    def close(self):
        self.writer.close()

