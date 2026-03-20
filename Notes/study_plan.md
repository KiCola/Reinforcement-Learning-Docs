# 深度强化学习算法程序员成长计划

> 目标：系统掌握深度强化学习核心算法，具备自主科研与工程落地能力
> 节奏：每天不超过 2 小时，理论 + 代码并行推进
> 工具链：Python / PyTorch / Gymnasium

---

## 总体路线图

```
强化学习基础
    │
    ▼
值函数方法 (DQN 族)
    │
    ▼
策略梯度方法 (REINFORCE / A2C)
    │
    ▼
Actor-Critic 进阶 (PPO / SAC / TD3)
    │
    ▼
前沿专题 (MBRL / MARL / 具身智能)
```

---

## 第一阶段：强化学习基础巩固（Week 1–2）

### Day 1 — 环境搭建 & MDP 复习（≤2h）
**任务**
- [ ] 安装 conda 环境，配置 PyTorch + Gymnasium
- [ ] 在 CartPole-v1 上跑通随机策略，观察 reward 曲线
- [ ] 复习 MDP 五元组：$\mathcal{S, A, P, R, \gamma}$

**参考**
- 书：《深度强化学习》王树森 — 第1章
- 书：Sutton《Reinforcement Learning》— Chapter 3
- 代码：https://gymnasium.farama.org/introduction/basic_usage/

### Day 2 — 贝尔曼方程推导（≤2h）
**任务**
- [ ] 手推贝尔曼期望方程（$V^\pi$、$Q^\pi$）
- [ ] 手推贝尔曼最优方程
- [ ] 矩阵形式求解小型 GridWorld

**参考**
- Sutton — Chapter 3.5–3.6
- https://datawhalechina.github.io/easy-rl/#/chapter2/chapter2

### Day 3 — 动态规划（≤2h）
**任务**
- [ ] 实现策略评估、策略迭代、值迭代
- [ ] 在 FrozenLake-v1 验证收敛

**参考**
- Sutton — Chapter 4
- https://github.com/dennybritz/reinforcement-learning/tree/master/DP
- https://datawhalechina.github.io/easy-rl/#/chapter3/chapter3

### Day 4 — Monte Carlo 方法（≤2h）
**任务**
- [ ] 实现 First-Visit MC 在 Blackjack-v1 上的策略评估
- [ ] 对比 MC 与 DP 的适用场景

**参考**
- Sutton — Chapter 5
- https://github.com/dennybritz/reinforcement-learning/tree/master/MC

### Day 5 — Q-Learning & SARSA（≤2h）✅ 有练习文件
**任务**
- [ ] 完成 `exercises/phase1_basics/q_learning_sarsa.py`
- [ ] 在 CliffWalking-v0 对比两者路径差异

**参考**
- Sutton — Chapter 6
- https://datawhalechina.github.io/easy-rl/#/chapter4/chapter4

### Day 6 — 函数近似基础（≤2h）
**任务**
- [ ] 理解线性函数近似与神经网络近似的区别
- [ ] 理解 deadly triad（离轨 + 函数近似 + 自举）

**参考**
- Sutton — Chapter 9–10
- DQN 原始论文：https://arxiv.org/abs/1312.5602

### Day 7 — 第一阶段总结（≤2h）
**任务**
- [ ] 写 Markdown 总结：MDP → DP → MC → TD 的演进逻辑
- [ ] 徒手写出 Q-Learning 更新式

---

## 第二阶段：DQN 族（Week 3–5）

### Day 8 — DQN 原理精读（≤2h）
**参考**
- 论文（Nature 2015）：https://www.nature.com/articles/nature14236
- 解读：https://zhuanlan.zhihu.com/p/26052182
- 王树森 — 第4章

### Day 9 — DQN 实现（≤2h）✅ 有练习文件
**任务**：完成 `exercises/phase2_dqn/dqn.py`，在 CartPole-v1 收敛

**参考**
- CleanRL：https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
- Easy-RL Chapter 6：https://datawhalechina.github.io/easy-rl/#/chapter6/chapter6

### Day 10 — Double DQN & Dueling DQN（≤2h）
**参考**
- Double DQN：https://arxiv.org/abs/1509.06461
- Dueling DQN：https://arxiv.org/abs/1511.06581

### Day 11 — Prioritized Experience Replay（≤2h）
**参考**
- 论文：https://arxiv.org/abs/1511.05952
- SumTree：https://github.com/rlcode/per

### Day 12 — Rainbow 概览（≤2h）
**参考**
- Rainbow：https://arxiv.org/abs/1710.02298
- C51：https://arxiv.org/abs/1707.06887

### Day 13 — Atari 实战（≤2h）
**参考**
- CleanRL Atari DQN：https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py

### Day 14 — 第二阶段总结（≤2h）

---

## 第三阶段：策略梯度（Week 6–8）

### Day 15 — 策略梯度定理推导（≤2h）
**参考**
- Sutton — Chapter 13
- Spinning Up：http://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

### Day 16 — REINFORCE 实现（≤2h）✅ 有练习文件
**任务**：完成 `exercises/phase3_policy_gradient/reinforce.py`

**参考**
- Easy-RL Chapter 9：https://datawhalechina.github.io/easy-rl/#/chapter9/chapter9

### Day 17 — A2C（≤2h）
**参考**
- CleanRL A2C：https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/a2c.py

### Day 18 — 重要性采样（≤2h）
**参考**：Sutton — Chapter 5.5–5.9，王树森 — 第8章

### Day 19 — TRPO 原理（≤2h）
**参考**
- 论文：https://arxiv.org/abs/1502.05477
- Spinning Up：https://spinningup.openai.com/en/latest/algorithms/trpo.html

### Day 20 — GAE（≤2h）
**参考**：论文 https://arxiv.org/abs/1506.02438

### Day 21 — 第三阶段总结（≤2h）

---

## 第四阶段：Actor-Critic 进阶（Week 9–11）

### Day 22 — PPO 原理精读（≤2h）
**参考**
- 论文：https://arxiv.org/abs/1707.06347
- 37个细节：https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

### Day 23 — PPO 实现（≤2h）✅ 有练习文件
**任务**：完成 `exercises/phase4_actor_critic/ppo.py`，在 LunarLander-v2 训练

**核心超参**：lr=3e-4, n_steps=2048, n_epochs=10, clip_eps=0.2, gae_lambda=0.95

**参考**：CleanRL https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

### Day 24 — PPO 调参与消融（≤2h）
**参考**：https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html

### Day 25 — DDPG（≤2h）
**参考**
- 论文：https://arxiv.org/abs/1509.02971
- CleanRL：https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py

### Day 26 — TD3（≤2h）
**参考**
- 论文：https://arxiv.org/abs/1802.09477
- CleanRL：https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py

### Day 27 — SAC 原理（≤2h）
**参考**
- SAC v2：https://arxiv.org/abs/1812.05905

### Day 28 — SAC 实现（≤2h）✅ 有练习文件
**任务**：完成 `exercises/phase4_actor_critic/sac.py`

**参考**：CleanRL https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py

### Day 29 — 算法横向对比（≤2h）

| 算法 | 类型 | 动作空间 | Sample Eff. | 稳定性 |
|------|------|----------|-------------|--------|
| DQN  | off-policy | 离散 | 中 | 中 |
| PPO  | on-policy  | 两者 | 低 | 高 |
| DDPG | off-policy | 连续 | 高 | 低 |
| TD3  | off-policy | 连续 | 高 | 中 |
| SAC  | off-policy | 连续 | 高 | 高 |

### Day 30 — 第四阶段总结（≤2h）

---

## 第五阶段：前沿专题（Week 12–16）

### Day 31–33 — MBRL（≤2h/天）
**参考**：Dreamer v3 https://arxiv.org/abs/2301.04104

### Day 34–36 — MARL（≤2h/天）
**参考**：QMIX https://arxiv.org/abs/1803.11605，MAPPO https://arxiv.org/abs/2103.01955

### Day 37–39 — Offline RL（≤2h/天）
**参考**：CQL https://arxiv.org/abs/2006.04779，D4RL https://github.com/Farama-Foundation/D4RL

### Day 40–42 — 具身智能（≤2h/天）
**参考**
- IsaacLab：https://isaac-sim.github.io/IsaacLab/
- legged_gym：https://github.com/leggedrobotics/legged_gym
- RMA：https://arxiv.org/abs/2107.04034

### Day 43–45 — RLHF（≤2h/天）
**参考**：InstructGPT https://arxiv.org/abs/2203.02155，DPO https://arxiv.org/abs/2305.18290

### Day 46–50 — 自主科研实践（≤2h/天）
**任务**：复现一篇近2年顶会论文，提出改进，撰写实验报告，发布 GitHub 项目

---

## 第六阶段：模仿学习（Week 17–20）

> 目标：掌握从专家数据中学习策略的核心方法，理解与强化学习的融合范式

### Day 51 — 模仿学习概览（≤2h）

**任务**
- [ ] 理解模仿学习（Imitation Learning, IL）的问题定义：给定专家轨迹 $\mathcal{D} = \{(s_t, a_t)\}$，学习策略 $\pi$
- [ ] 梳理三大范式的关系：行为克隆（BC）/ 逆强化学习（IRL）/ 生成对抗模仿学习（GAIL）
- [ ] 理解分布偏移（Distribution Shift）问题：训练分布与执行分布不一致

**参考**
- 综述：Imitation Learning Tutorial (ICML 2018) https://sites.google.com/view/icml2018-imitation-learning/
- 博客：A Survey of Imitation Learning https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#imitation-learning
- 王树森《深度强化学习》— 附录模仿学习章节

---

### Day 52 — 行为克隆（Behavior Cloning, BC）（≤2h）

**任务**
- [ ] 理解 BC 的本质：将模仿学习转化为监督学习，最小化 $\mathcal{L} = \mathbb{E}_{(s,a)\sim\mathcal{D}}[-\log\pi_\theta(a|s)]$
- [ ] 实现 BC：在 CartPole/LunarLander 上用 PPO 生成专家数据，再训练 BC 策略
- [ ] 观察并分析 Covariate Shift（协变量偏移）：BC 策略在遇到训练集外状态时失效

**参考**
- 论文：A Reduction of Imitation Learning (Ross 2010) https://arxiv.org/abs/1011.0686
- 代码参考：https://github.com/HumanCompatibleAI/imitation

**关键代码结构**
```
1. 用训练好的 PPO/SAC 收集专家轨迹 → expert_dataset
2. 构建 Dataset: (obs, action) pairs
3. 用交叉熵/MSE 做监督学习训练策略网络
4. 评估：对比 BC 策略与专家策略的 episode reward
```

---

### Day 53 — DAgger（Dataset Aggregation）（≤2h）

**任务**
- [ ] 理解 DAgger 解决分布偏移的核心思路：
  在执行中不断查询专家，将新状态下的专家动作加入训练集
- [ ] 推导 DAgger 的次线性遗憾界：$\text{regret} \leq O(T^{1/2})$（相比 BC 的 $O(T^2)$）
- [ ] 实现 DAgger 主循环（Interactive IL）
- [ ] 对比 BC 与 DAgger 在需要长程规划任务上的性能差距

**参考**
- 论文：DAgger (Ross 2011) https://arxiv.org/abs/1011.0686
- 解读博客：https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf

**DAgger 伪代码**
```
D ← 专家演示数据集
训练初始策略 π_1 on D
for i = 1, 2, ..., N:
    用 π_i 执行，收集状态序列 {s_t}
    查询专家：获得 {(s_t, a*_t)} 标注
    D ← D ∪ {(s_t, a*_t)}
    在 D 上重新训练 π_{i+1}
```

---

### Day 54 — 逆强化学习（IRL）原理（≤2h）

**任务**
- [ ] 理解 IRL 的目标：从专家行为反推奖励函数 $R^*$，再用 RL 求解最优策略
- [ ] 学习 MaxEnt IRL（最大熵逆强化学习）：
  $\max_R \mathbb{E}_{\tau\sim\pi^*}[R(\tau)] - \log Z(R)$
- [ ] 理解 IRL 与 BC 的核心区别：IRL 学习的是"为什么"而非"怎么做"
- [ ] 了解 IRL 的计算挑战：内循环需要不断求解 RL 问题

**参考**
- 论文 MaxEnt IRL：https://arxiv.org/abs/1507.04888
- 论文 Apprenticeship Learning (Abbeel 2004)：https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf
- 综述：https://thegradient.pub/learning-from-humans-what-is-inverse-reinforcement-learning/

---

### Day 55 — GAIL（生成对抗模仿学习）原理（≤2h）

**任务**
- [ ] 理解 GAIL 将 IRL + RL 统一为 GAN 框架：
  - 判别器 $D$：区分专家轨迹与策略轨迹
  - 生成器 $\pi$（策略）：欺骗判别器
  - 目标：$\min_\pi \max_D \mathbb{E}_{\pi^*}[\log D] + \mathbb{E}_\pi[\log(1-D)]$
- [ ] 理解 GAIL 与 GAN 训练不稳定问题的关联
- [ ] 了解 f-GAIL、AIRL 等变体对奖励可迁移性的改进

**参考**
- 论文 GAIL：https://arxiv.org/abs/1606.03476
- 论文 AIRL：https://arxiv.org/abs/1710.11248
- 代码库：https://github.com/HumanCompatibleAI/imitation

---

### Day 56 — GAIL 代码实现（≤2h）✅ 有练习文件

**任务**
- [ ] 完成 `exercises/phase6_imitation/gail.py`
- [ ] 在 CartPole-v1 / LunarLander-v2 上训练 GAIL
- [ ] 对比 BC、DAgger、GAIL 三种方法的 sample efficiency 与最终性能

**参考**
- 论文：https://arxiv.org/abs/1606.03476
- imitation 库：https://imitation.readthedocs.io/en/latest/algorithms/gail.html
- CleanRL 风格实现参考：https://github.com/vwxyzjn/cleanrl

**关键代码结构**
```
Discriminator (nn.Module)   # 输入 (s, a)，输出真/假概率
Policy (ActorCritic)        # 与 PPO 相同结构

train_loop:
    用当前策略收集轨迹
    更新判别器（BCE loss）
    用判别器输出作为奖励 r = -log(1 - D(s,a))
    用 PPO 更新策略
```

---

### Day 57 — AIRL 与可迁移奖励（≤2h）

**任务**
- [ ] 理解 AIRL（Adversarial Inverse Reinforcement Learning）的设计动机：
  GAIL 学到的奖励与环境动力学耦合，无法迁移
- [ ] 学习 AIRL 的奖励分解：$f(s,a,s') = g(s,a) - \gamma h(s') + h(s)$（势函数整形）
- [ ] 理解 AIRL 奖励在 sim-to-real 中的应用价值

**参考**
- 论文 AIRL：https://arxiv.org/abs/1710.11248
- 解读：https://zhuanlan.zhihu.com/p/574013871

---

### Day 58 — 从人类反馈中学习（RLHF 深入）（≤2h）

**任务**
- [ ] 重新审视 RLHF 三阶段：SFT → Reward Model → PPO
- [ ] 理解 Reward Model 本质上是 IRL 的一种形式
- [ ] 了解 Constitutional AI（CAI）和 RLAIF（AI Feedback）
- [ ] 对比 RLHF 与经典 IRL 的联系与区别

**参考**
- 论文 InstructGPT：https://arxiv.org/abs/2203.02155
- 论文 Constitutional AI：https://arxiv.org/abs/2212.08073
- 博客：https://huggingface.co/blog/rlhf

---

### Day 59 — 离线模仿学习与 Decision Transformer（≤2h）

**任务**
- [ ] 理解 Offline IL：只有静态数据集，无法与环境交互
- [ ] 学习 Decision Transformer：将 RL 问题建模为序列预测
  输入：$(R_{\text{to-go}}, s_t, a_t, \ldots)$，预测下一动作
- [ ] 了解 Trajectory Transformer、Gato 等后续工作
- [ ] 思考：Transformer + IL 对具身智能的意义

**参考**
- 论文 Decision Transformer：https://arxiv.org/abs/2106.01345
- 论文 Trajectory Transformer：https://arxiv.org/abs/2106.02039
- 论文 Gato：https://arxiv.org/abs/2205.06175

---

### Day 60 — 模仿学习专题总结（≤2h）

**任务**
- [ ] 完成方法对比表（见下）
- [ ] 整理适用场景：何时用 BC？何时用 GAIL？何时用 IRL？
- [ ] 思考：在具身智能（机器人操作/运动控制）中，模仿学习的瓶颈是什么？

| 方法 | 是否需要交互 | 专家数据量 | 奖励函数 | 适用场景 |
|------|------------|-----------|---------|----------|
| BC | 否 | 大 | 不需要 | 简单任务、数据充足 |
| DAgger | 是（查询专家）| 中 | 不需要 | 有专家在线的场景 |
| MaxEnt IRL | 是 | 中 | 自动学习 | 需要理解意图 |
| GAIL | 是 | 小 | 自动学习 | 通用连续控制 |
| AIRL | 是 | 小 | 可迁移 | sim-to-real |
| Decision Transformer | 否 | 大 | 不需要 | 离线大数据场景 |

---

## 必读论文清单

| 论文 | 年份 | 链接 |
|------|------|------|
| DQN (Atari) | 2013 | https://arxiv.org/abs/1312.5602 |
| DQN (Nature) | 2015 | https://www.nature.com/articles/nature14236 |
| DDPG | 2015 | https://arxiv.org/abs/1509.02971 |
| TRPO | 2015 | https://arxiv.org/abs/1502.05477 |
| GAE | 2015 | https://arxiv.org/abs/1506.02438 |
| PPO | 2017 | https://arxiv.org/abs/1707.06347 |
| Rainbow | 2017 | https://arxiv.org/abs/1710.02298 |
| SAC | 2018 | https://arxiv.org/abs/1812.05905 |
| TD3 | 2018 | https://arxiv.org/abs/1802.09477 |
| GAIL | 2016 | https://arxiv.org/abs/1606.03476 |
| AIRL | 2018 | https://arxiv.org/abs/1710.11248 |
| DAgger | 2011 | https://arxiv.org/abs/1011.0686 |
| Decision Transformer | 2021 | https://arxiv.org/abs/2106.01345 |
| Dreamer v3 | 2023 | https://arxiv.org/abs/2301.04104 |

---

## 进度追踪

| 阶段 | 起止时间 | 完成度 | 备注 |
|------|----------|--------|------|
| Phase 1：RL 基础 | / | 0% | |
| Phase 2：DQN 族 | / | 0% | |
| Phase 3：策略梯度 | / | 0% | |
| Phase 4：AC 进阶 | / | 0% | |
| Phase 5：前沿专题 | / | 0% | |
| Phase 6：模仿学习 | / | 0% | |

> **核心原则：不理解公式不写代码，不跑通代码不进下一章。**

*创建时间：2026-03-20*


| 论文 | 年份 | 链接 |
|------|------|------|
| DQN (Atari) | 2013 | https://arxiv.org/abs/1312.5602 |
| DQN (Nature) | 2015 | https://www.nature.com/articles/nature14236 |
| DDPG | 2015 | https://arxiv.org/abs/1509.02971 |
| TRPO | 2015 | https://arxiv.org/abs/1502.05477 |
| GAE | 2015 | https://arxiv.org/abs/1506.02438 |
| PPO | 2017 | https://arxiv.org/abs/1707.06347 |
| Rainbow | 2017 | https://arxiv.org/abs/1710.02298 |
| SAC | 2018 | https://arxiv.org/abs/1812.05905 |
| TD3 | 2018 | https://arxiv.org/abs/1802.09477 |
| Dreamer v3 | 2023 | https://arxiv.org/abs/2301.04104 |

---

## 进度追踪

| 阶段 | 起止时间 | 完成度 | 备注 |
|------|----------|--------|------|
| Phase 1：RL 基础 | / | 0% | |
| Phase 2：DQN 族 | / | 0% | |
| Phase 3：策略梯度 | / | 0% | |
| Phase 4：AC 进阶 | / | 0% | |
| Phase 5：前沿专题 | / | 0% | |

> **核心原则：不理解公式不写代码，不跑通代码不进下一章。**

*创建时间：2026-03-20*

