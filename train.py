# 统一训练入口
# 用法：
#   python train.py --algo dqn   --env CartPole-v1
#   python train.py --algo ppo   --env LunarLander-v2
#   python train.py --algo sac   --env Pendulum-v1
#   python train.py --algo reinforce --env CartPole-v1

import argparse
import sys
import os

# 将 exercises/ 加入模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "exercises"))


def main():
    parser = argparse.ArgumentParser(description="Deep RL From Scratch — 统一训练入口")
    parser.add_argument("--algo",  type=str, required=True,
                        choices=["dqn", "reinforce", "ppo", "sac"],
                        help="选择算法")
    parser.add_argument("--env",   type=str, default=None,
                        help="Gymnasium 环境 ID（覆盖默认值）")
    parser.add_argument("--steps", type=int, default=None,
                        help="总训练步数（覆盖默认值）")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    if args.algo == "dqn":
        from phase2_dqn.dqn import train, Config
        cfg = Config(seed=args.seed)
        if args.env:   cfg.env_id = args.env
        if args.steps: cfg.total_timesteps = args.steps
        train(cfg)

    elif args.algo == "reinforce":
        import gymnasium as gym
        from phase3_policy_gradient.reinforce import reinforce
        env_id = args.env or "CartPole-v1"
        env    = gym.make(env_id)
        reinforce(env, n_episodes=args.steps or 800, use_baseline=True)
        env.close()

    elif args.algo == "ppo":
        from phase4_actor_critic.ppo import train_ppo, Config
        cfg = Config(seed=args.seed)
        if args.env:   cfg.env_id = args.env
        if args.steps: cfg.total_timesteps = args.steps
        train_ppo(cfg)

    elif args.algo == "sac":
        from phase4_actor_critic.sac import train_sac, Config
        cfg = Config(seed=args.seed)
        if args.env:   cfg.env_id = args.env
        if args.steps: cfg.total_timesteps = args.steps
        train_sac(cfg)


if __name__ == "__main__":
    main()
