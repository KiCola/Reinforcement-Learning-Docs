<div align="center">

# 🎯 Deep RL From Scratch

**从零手写深度强化学习 — 理论 + 代码 + 50天成长计划**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-black)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[English](#english) | [快速开始](#快速开始) | [学习计划](#学习计划) | [算法列表](#算法列表)

</div>

---

## 这是什么？

本仓库是一个面向**算法程序员**的深度强化学习自学项目，特点如下：

- **理论 + 代码并行**：每个算法都有完整的公式推导笔记和对应的 PyTorch 实现
- **练习驱动**：`exercises/` 目录提供带 `TODO` 的代码框架，自己填写核心逻辑
- **标准答案**：`solutions/` 目录提供完整参考实现，做完再看
- **50天计划**：每天不超过 2 小时，从 MDP 基础到具身智能全覆盖
- **极简依赖**：仅 PyTorch + Gymnasium，不依赖 SB3 等封装库

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/KiCola/Reinforcement-Learning-Docs.git
cd Reinforcement-Learning-Docs
```

### 2. 创建环境

```bash
conda create -n drl python=3.10 -y
conda activate drl
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python scripts/verify_install.py
```

### 4. 开始第一个练习

```bash
# 打开练习文件，完成 TODO
code exercises/phase1_basics/q_learning_sarsa.py

# 完成后运行验证
python exercises/phase1_basics/q_learning_sarsa.py
```

---

## 仓库结构

```
deep-rl-from-scratch/
│
├── exercises/                      # 练习文件（含 TODO，自己填写）
│   ├── phase1_basics/
│   │   └── q_learning_sarsa.py     # Day 5：Q-Learning vs SARSA
│   ├── phase2_dqn/
│   │   └── dqn.py                  # Day 9：DQN
│   ├── phase3_policy_gradient/
│   │   └── reinforce.py            # Day 16：REINFORCE
│   ├── phase4_actor_critic/
│   │   ├── ppo.py                  # Day 23：PPO
│   │   └── sac.py                  # Day 28：SAC
│   └── utils/
│       ├── networks.py             # 通用网络模块（练习）
│       ├── logger.py               # TensorBoard 日志（已完成）
│       └── wrappers.py             # 环境 Wrapper（已完成）
│
├── solutions/                      # 标准答案 ⚠️ 做完再看！
│   ├── phase1_basics/
│   ├── phase2_dqn/
│   ├── phase3_policy_gradient/
│   ├── phase4_actor_critic/
│   └── utils/
│
├── notes/                          # 学习笔记（Markdown）
│   └── 深度强化学习成长计划.md
│
├── scripts/
│   └── verify_install.py           # 环境验证脚本
│
├── train.py                        # 统一训练入口
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 算法列表

| 阶段 | 算法 | 动作空间 | 练习文件 | 参考论文 |
|------|------|----------|----------|----------|
| Phase 1 | Q-Learning | 离散 | `exercises/phase1_basics/q_learning_sarsa.py` | [Watkins 1989](http://www.cs.rhul.ac.uk/~chrisw/thesis.pdf) |
| Phase 1 | SARSA | 离散 | `exercises/phase1_basics/q_learning_sarsa.py` | [Rummery 1994](https://www.researchgate.net/publication/2500611) |
| Phase 2 | DQN | 离散 | `exercises/phase2_dqn/dqn.py` | [Mnih 2015](https://www.nature.com/articles/nature14236) |
| Phase 3 | REINFORCE | 离散/连续 | `exercises/phase3_policy_gradient/reinforce.py` | [Williams 1992](https://link.springer.com/article/10.1007/BF00992696) |
| Phase 4 | PPO | 离散/连续 | `exercises/phase4_actor_critic/ppo.py` | [Schulman 2017](https://arxiv.org/abs/1707.06347) |
| Phase 4 | SAC | 连续 | `exercises/phase4_actor_critic/sac.py` | [Haarnoja 2018](https://arxiv.org/abs/1812.05905) |
| Phase 6 | BC | 离散/连续 | `exercises/phase6_imitation/bc.py` | [Pomerleau 1989](https://proceedings.neurips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf) |
| Phase 6 | DAgger | 离散/连续 | `exercises/phase6_imitation/dagger.py` | [Ross 2011](https://arxiv.org/abs/1011.0686) |
| Phase 6 | GAIL | 连续 | `exercises/phase6_imitation/gail.py` | [Ho 2016](https://arxiv.org/abs/1606.03476) |

<details>
<summary>更多算法（计划中）</summary>

| 算法 | 阶段 | 论文 |
|------|------|------|
| Double DQN | Phase 2 | [van Hasselt 2016](https://arxiv.org/abs/1509.06461) |
| Dueling DQN | Phase 2 | [Wang 2016](https://arxiv.org/abs/1511.06581) |
| PER | Phase 2 | [Schaul 2016](https://arxiv.org/abs/1511.05952) |
| A2C | Phase 3 | [Mnih 2016](https://arxiv.org/abs/1602.01783) |
| DDPG | Phase 4 | [Lillicrap 2016](https://arxiv.org/abs/1509.02971) |
| TD3 | Phase 4 | [Fujimoto 2018](https://arxiv.org/abs/1802.09477) |
| Dreamer v3 | Phase 5 | [Hafner 2023](https://arxiv.org/abs/2301.04104) |

</details>

---

## 学习计划

> 每天不超过 2 小时：第 1 小时理论推导，第 2 小时动手编码

| 阶段 | 周数 | 核心内容 | 关键算法 |
|------|------|----------|----------|
| **Phase 1**：RL 基础 | Week 1–2 | MDP、贝尔曼方程、DP、MC、TD | Q-Learning、SARSA |
| **Phase 2**：值函数方法 | Week 3–5 | 深度 Q 网络、经验回放、目标网络 | DQN 及变体 |
| **Phase 3**：策略梯度 | Week 6–8 | 策略梯度定理、GAE、TRPO | REINFORCE、A2C |
| **Phase 4**：AC 进阶 | Week 9–11 | PPO/SAC/TD3，连续控制 | PPO、SAC、TD3 |
| **Phase 5**：前沿专题 | Week 12–16 | MBRL、MARL、Offline RL、具身智能 | Dreamer、QMIX、CQL |
| **Phase 6**：模仿学习 | Week 17–20 | BC、DAgger、IRL、GAIL、Decision Transformer | GAIL、AIRL、DT |

详细的每日任务、参考资料和思考题见 👉 [notes/深度强化学习成长计划.md](notes/深度强化学习成长计划.md)

---

## 使用方法

### 练习模式（推荐）

每个练习文件都包含：
- 完整的网络结构框架
- `TODO N：` 注释说明需要实现的内容
- `raise NotImplementedError` 占位符（实现后删除）
- 文件末尾的**思考题**

```python
# 示例：dqn.py 中的一个 TODO

# TODO 5：计算 TD target（用 target_q_net，不计算梯度）
# 公式：td_target = r + γ * max_a' Q_target(s', a') * (1 - done)
with torch.no_grad():
    td_target = None  # TODO
```

完成后运行，图像会自动保存到对应目录。

### 统一训练入口

```bash
# 用练习文件中的实现训练
python train.py --algo dqn --env CartPole-v1
python train.py --algo ppo --env LunarLander-v2
python train.py --algo sac --env Pendulum-v1

# 可选参数
python train.py --algo ppo --env HalfCheetah-v4 --steps 1000000 --seed 0
```

### 对照答案

卡壳时的建议顺序：
1. 重新阅读参考论文/博客的对应章节
2. 在文件顶部的参考链接中查找 CleanRL 实现
3. 最后再看 `solutions/` 中的标准答案

---

## 环境要求

```
Python     >= 3.9
PyTorch    >= 2.0.0
Gymnasium  >= 0.29.0
NumPy      >= 1.24.0
Matplotlib >= 3.7.0
TensorBoard>= 2.13.0
```

可选（MuJoCo 连续控制）：
```bash
pip install gymnasium[mujoco]
```

可选（Atari）：
```bash
pip install gymnasium[atari] ale-py
```

---

## 推荐学习资料

### 教材

| 书名 | 作者 | 链接 |
|------|------|------|
| Reinforcement Learning: An Introduction | Sutton & Barto | [免费 PDF](http://incompleteideas.net/book/the-book-2nd.html) |
| 深度强化学习 | 王树森 | [GitHub](https://github.com/wangshusen/DeepLearning) |
| 动手学强化学习 | 张伟楠 et al. | [在线阅读](https://datawhalechina.github.io/easy-rl/) |

### 视频课程

| 课程 | 平台 | 链接 |
|------|------|------|
| 王树森 深度强化学习 | Bilibili | [链接](https://www.bilibili.com/video/BV12o4y197US/) |
| 张伟楠 强化学习 | Bilibili | [链接](https://space.bilibili.com/3546754433681656) |
| David Silver RL Course | YouTube | [链接](https://www.youtube.com/watch?v=2pWv7GOvuf0) |

### 优质代码库

| 项目 | 特点 | 链接 |
|------|------|------|
| CleanRL | 单文件极简实现，最佳参考 | [GitHub](https://github.com/vwxyzjn/cleanrl) |
| Spinning Up | OpenAI 出品，文档详尽 | [链接](https://spinningup.openai.com/) |
| Stable-Baselines3 | 工业级实现，调参用 | [GitHub](https://github.com/DLR-RM/stable-baselines3) |

---

## 贡献

欢迎提交 Issue 和 PR！

- 发现 bug 或笔误 → 提 Issue
- 完善练习题或新增算法 → 提 PR
- 分享你的学习笔记 → 提 PR 到 `notes/`

---

## License

MIT License — 自由使用，欢迎 star ⭐

---

<div id="english"></div>

## English Summary

This repository provides a **50-day self-study plan** for Deep Reinforcement Learning with:

- **Exercise templates** (`exercises/`) with `TODO` placeholders for core algorithm logic
- **Reference solutions** (`solutions/`) to check after completing exercises  
- **Detailed study plan** covering: MDP basics → DQN family → Policy Gradient → PPO/SAC/TD3 → Advanced topics
- All implementations use only **PyTorch + Gymnasium** (no high-level wrappers)

```bash
git clone https://github.com/YOUR_USERNAME/deep-rl-from-scratch.git
cd deep-rl-from-scratch
pip install -r requirements.txt
python train.py --algo ppo --env LunarLander-v2
```
