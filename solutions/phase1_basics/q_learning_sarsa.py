"""
Day 5：Q-Learning 与 SARSA 对比实现
环境：CliffWalking-v0
参考：Sutton《强化学习》第6章
"""
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


def make_env():
    return gym.make("CliffWalking-v0")


def epsilon_greedy(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[state]))


def sarsa(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """SARSA：on-policy TD 控制"""
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon, n_actions)
        ep_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = epsilon_greedy(Q, next_state, epsilon, n_actions)

            # SARSA 更新：使用实际执行的 next_action
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state, action = next_state, next_action
            ep_reward += reward

        rewards.append(ep_reward)

    return Q, rewards


def q_learning(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):
    """Q-Learning：off-policy TD 控制"""
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-Learning 更新：使用贪心的 max Q(s', a')
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            ep_reward += reward

        rewards.append(ep_reward)

    return Q, rewards


def smooth(rewards, window=20):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')


if __name__ == "__main__":
    env = make_env()
    N = 500

    print("训练 SARSA...")
    _, sarsa_rewards = sarsa(env, n_episodes=N)

    print("训练 Q-Learning...")
    _, ql_rewards = q_learning(env, n_episodes=N)

    # ---- 绘图对比 ----
    plt.figure(figsize=(10, 5))
    plt.plot(smooth(sarsa_rewards),   label='SARSA (on-policy)',      color='steelblue')
    plt.plot(smooth(ql_rewards),      label='Q-Learning (off-policy)', color='tomato')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (smoothed)')
    plt.title('SARSA vs Q-Learning on CliffWalking-v0')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase1_basics/sarsa_vs_qlearning.png', dpi=150)
    plt.show()
    print("图像已保存至 phase1_basics/sarsa_vs_qlearning.png")
    env.close()

"""
思考题（Day 5 作业）：
1. 在 CliffWalking 中，SARSA 和 Q-Learning 哪个找到的路径更靠近悬崖？为什么？
2. SARSA 是 on-policy，Q-Learning 是 off-policy，具体体现在更新公式的哪一行？
3. 如果 epsilon=0（纯贪心），两者的更新公式会变成一样吗？
"""

