# from pathlib import Path

# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor


# ROOT = Path(__file__).resolve().parent
# MODEL_DIR = ROOT / "models"
# BEST_DIR = MODEL_DIR / "best_model"
# TB_DIR = ROOT / "tb_logs"
# EVAL_DIR = ROOT / "eval_logs"

# MODEL_DIR.mkdir(exist_ok=True)
# BEST_DIR.mkdir(exist_ok=True)
# TB_DIR.mkdir(exist_ok=True)
# EVAL_DIR.mkdir(exist_ok=True)


# def make_env():
#     env = gym.make("Reacher-v5")
#     env = Monitor(env)
#     return env


# def main():
#     train_env = make_env()
#     eval_env = make_env()

#     model = PPO(
#         policy="MlpPolicy",
#         env=train_env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         gamma=0.99,
#         gae_lambda=0.95,
#         ent_coef=0.0,
#         clip_range=0.2,
#         verbose=1,
#         tensorboard_log=str(TB_DIR),
#         policy_kwargs=dict(
#             net_arch=dict(pi=[64, 64], vf=[64, 64])
#         ),
#     )

#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=str(BEST_DIR),
#         log_path=str(EVAL_DIR),
#         eval_freq=5000,
#         deterministic=True,
#         render=False,
#     )

#     total_timesteps = 120_000
#     print(f"开始训练，总步数: {total_timesteps}")
#     model.learn(total_timesteps=total_timesteps, callback=eval_callback)

#     final_model_path = MODEL_DIR / "ppo_reacher_final"
#     model.save(str(final_model_path))
#     print(f"最终模型已保存到: {final_model_path}.zip")

#     mean_reward, std_reward = evaluate_policy(
#         model,
#         eval_env,
#         n_eval_episodes=20,
#         deterministic=True,
#     )
#     print(f"评估结果: mean_reward={mean_reward:.3f} +/- {std_reward:.3f}")

#     result_txt = MODEL_DIR / "result.txt"
#     with open(result_txt, "w", encoding="utf-8") as f:
#         f.write(f"mean_reward={mean_reward:.6f}\n")
#         f.write(f"std_reward={std_reward:.6f}\n")
#         f.write(f"timesteps={total_timesteps}\n")

#     train_env.close()
#     eval_env.close()


# if __name__ == "__main__":
#     main()








############### 第二种 ##################







from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
BEST_DIR = MODEL_DIR / "best_model"
TB_DIR = ROOT / "tb_logs"
EVAL_DIR = ROOT / "eval_logs"

MODEL_DIR.mkdir(exist_ok=True)
BEST_DIR.mkdir(exist_ok=True)
TB_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)


def make_train_env(n_envs: int = 8, seed: int = 42):
    vec_env = make_vec_env("Reacher-v5", n_envs=n_envs, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    return vec_env


def make_eval_env(seed: int = 123):
    vec_env = make_vec_env("Reacher-v5", n_envs=1, seed=seed)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def main():
    train_env = make_train_env(n_envs=8, seed=42)
    eval_env = make_eval_env(seed=123)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(TB_DIR),
        device="auto",   # 能用 CUDA 就会尝试，但这个任务未必比 CPU 快
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(EVAL_DIR),
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )

    total_timesteps = 1_000_000
    print(f"开始长训练，总步数: {total_timesteps}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="ppo_reacher_longrun",
        progress_bar=True,
    )

    final_model_path = MODEL_DIR / "ppo_reacher_final_long"
    vec_stats_path = MODEL_DIR / "vec_normalize.pkl"

    model.save(str(final_model_path))
    train_env.save(str(vec_stats_path))

    print(f"最终模型已保存到: {final_model_path}.zip")
    print(f"归一化统计已保存到: {vec_stats_path}")

    eval_env.training = False
    eval_env.norm_reward = False
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=30,
        deterministic=True,
    )

    print(f"评估结果: mean_reward={mean_reward:.3f} +/- {std_reward:.3f}")

    result_txt = MODEL_DIR / "result_longrun.txt"
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write(f"mean_reward={mean_reward:.6f}\n")
        f.write(f"std_reward={std_reward:.6f}\n")
        f.write(f"timesteps={total_timesteps}\n")
        f.write("algo=PPO\n")
        f.write("n_envs=8\n")
        f.write("vec_normalize=True\n")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()