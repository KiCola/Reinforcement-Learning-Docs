"""
【练习】Day 5：Q-Learning 与 SARSA 对比实现
环境：CliffWalking-v0
参考：Sutton《强化学习》第6章
Easy-RL: https://datawhalechina.github.io/easy-rl/#/chapter4/chapter4
标准答案：solutions/phase1_basics/q_learning_sarsa.py
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


def make_env():
    return gym.make("CliffWalking-v0")


def epsilon_greedy(Q, state, epsilon, n_actions):
    """
    ε-greedy 动作选择
    以概率 ε 随机探索，以概率 1-ε 选择 argmax Q(s, a)

    TODO: 完成此函数
    提示：np.random.rand() 返回 [0,1) 的随机数
          np.argmax(Q[state]) 返回最优动作
    """
    pass


def sarsa(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    SARSA：on-policy TD 控制
    更新公式：Q(s,a) <- Q(s,a) + α·[r + γ·Q(s',a') - Q(s,a)]

    on-policy 关键：a' 必须是由当前策略（ε-greedy）实际选出的动作
    """
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        # TODO: 用 epsilon_greedy 选择初始动作 action
        action = None
        ep_reward = 0
        done = False

        while not done:
            # TODO: 执行 action，获得 (next_state, reward, terminated, truncated, _)
            raise NotImplementedError
            done = terminated or truncated

            # TODO: 选择 next_action（SARSA 与 Q-Learning 的本质区别在这里）
            next_action = None

            # TODO: 用贝尔曼方程计算 TD target
            # td_target = r + γ * Q(s', a') * (not done)
            td_target = None

            # TODO: 更新 Q 表
            # Q(s,a) += α * (td_target - Q(s,a))

            state, action = next_state, next_action
            ep_reward += reward

        rewards.append(ep_reward)

    return Q, rewards


def q_learning(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Q-Learning：off-policy TD 控制
    更新公式：Q(s,a) <- Q(s,a) + α·[r + γ·max_a' Q(s',a') - Q(s,a)]

    off-policy 关键：用 max Q(s',a') 更新，与执行哪个动作无关
    """
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            # TODO: 用 epsilon_greedy 选择 action
            action = None

            # TODO: 执行动作
            raise NotImplementedError
            done = terminated or truncated

            # TODO: 计算 TD target（注意和 SARSA 的区别：这里用 max）
            # td_target = r + γ * max_a' Q(s', a') * (not done)
            td_target = None

            # TODO: 更新 Q 表

            state = next_state
            ep_reward += reward

        rewards.append(ep_reward)

    return Q, rewards


def smooth(rewards, window=20):
    return np.convolve(rewards, np.ones(window) / window, mode='valid')


if __name__ == "__main__":
    env = make_env()

    print("训练 SARSA...")
    _, sarsa_rewards = sarsa(env, n_episodes=500)

    print("训练 Q-Learning...")
    _, ql_rewards = q_learning(env, n_episodes=500)

    plt.figure(figsize=(10, 5))
    plt.plot(smooth(sarsa_rewards), label='SARSA (on-policy)',       color='steelblue')
    plt.plot(smooth(ql_rewards),    label='Q-Learning (off-policy)', color='tomato')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (smoothed)')
    plt.title('SARSA vs Q-Learning on CliffWalking-v0')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sarsa_vs_qlearning.png', dpi=150)
    plt.show()
    env.close()

"""
思考题（完成代码后回答）：
1. SARSA 和 Q-Learning 在 CliffWalking 中找到的路径有何不同？为什么？
2. on-policy / off-policy 的区别具体体现在代码的哪一行？
3. 如果 epsilon=0，两者的更新公式会完全一样吗？
"""

