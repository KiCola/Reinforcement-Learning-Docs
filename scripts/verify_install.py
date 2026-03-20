"""
环境安装验证脚本
运行：python scripts/verify_install.py
"""
import sys

PASSED = []
FAILED = []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  [OK]  {name}")
    except Exception as e:
        FAILED.append(name)
        print(f"  [FAIL] {name}: {e}")


print("\n=== Deep RL From Scratch — 环境验证 ===")
print(f"Python {sys.version}\n")

# --- 核心依赖 ---
print("[1] 核心依赖")
check("torch", lambda: __import__("torch"))
check("torch.cuda (GPU)", lambda: (_ for _ in ()).throw(Exception("无 GPU，CPU 训练")) if not __import__("torch").cuda.is_available() else None)
check("numpy", lambda: __import__("numpy"))
check("matplotlib", lambda: __import__("matplotlib"))

# --- RL 环境 ---
print("\n[2] RL 环境")
check("gymnasium", lambda: __import__("gymnasium"))
check("CartPole-v1", lambda: __import__("gymnasium").make("CartPole-v1"))
check("LunarLander-v2", lambda: __import__("gymnasium").make("LunarLander-v2"))
check("Pendulum-v1", lambda: __import__("gymnasium").make("Pendulum-v1"))

# --- 可视化 ---
print("\n[3] 可视化")
check("tensorboard", lambda: __import__("torch.utils.tensorboard", fromlist=["SummaryWriter"]))
check("pandas", lambda: __import__("pandas"))

# --- 可选依赖 ---
print("\n[4] 可选依赖")
check("stable_baselines3", lambda: __import__("stable_baselines3"))
check("mujoco (MuJoCo)", lambda: __import__("gymnasium").make("HalfCheetah-v4"))
check("ale_py (Atari)", lambda: __import__("ale_py"))

# --- 汇总 ---
print(f"\n{'='*40}")
print(f"通过: {len(PASSED)} | 失败: {len(FAILED)}")
if FAILED:
    print(f"失败项: {', '.join(FAILED)}")
    print("提示：可选依赖失败不影响 Phase 1-4 的练习")
else:
    print("全部通过！可以开始学习了。")
print()

