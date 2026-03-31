# from pathlib import Path

# import gymnasium as gym
# from gymnasium.wrappers import RecordVideo
# from stable_baselines3 import PPO


# ROOT = Path(__file__).resolve().parent
# BEST_MODEL_PATH = ROOT / "models" / "best_model" / "best_model.zip"
# FINAL_MODEL_PATH = ROOT / "models" / "ppo_reacher_final.zip"
# VIDEO_DIR = ROOT / "videos"
# VIDEO_DIR.mkdir(exist_ok=True)


# def load_model():
#     if BEST_MODEL_PATH.exists():
#         print(f"加载 best model: {BEST_MODEL_PATH}")
#         return PPO.load(str(BEST_MODEL_PATH))
#     if FINAL_MODEL_PATH.exists():
#         print(f"加载 final model: {FINAL_MODEL_PATH}")
#         return PPO.load(str(FINAL_MODEL_PATH))
#     raise FileNotFoundError("没有找到模型文件，请先运行 train_reacher.py")


# def play_human(episodes=5):
#     model = load_model()
#     env = gym.make("Reacher-v5", render_mode="human")

#     for ep in range(episodes):
#         obs, info = env.reset(seed=100 + ep)
#         terminated = False
#         truncated = False
#         ep_reward = 0.0

#         while not (terminated or truncated):
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)
#             ep_reward += reward

#         print(f"[human] episode={ep}, reward={ep_reward:.3f}")

#     env.close()


# def record_video(episodes=3):
#     model = load_model()

#     env = gym.make("Reacher-v5", render_mode="rgb_array")
#     env = RecordVideo(
#         env,
#         video_folder=str(VIDEO_DIR),
#         episode_trigger=lambda episode_id: True,
#         name_prefix="reacher_ppo",
#     )

#     for ep in range(episodes):
#         obs, info = env.reset(seed=200 + ep)
#         terminated = False
#         truncated = False
#         ep_reward = 0.0

#         while not (terminated or truncated):
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = env.step(action)
#             ep_reward += reward

#         print(f"[video] episode={ep}, reward={ep_reward:.3f}")

#     env.close()
#     print(f"视频已保存到: {VIDEO_DIR}")


# if __name__ == "__main__":
#     record_video(episodes=3)
#     play_human(episodes=5)











from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ppo_reacher_final_long.zip"
VEC_STATS_PATH = ROOT / "models" / "vec_normalize.pkl"
VIDEO_DIR = ROOT / "videos"
VIDEO_DIR.mkdir(exist_ok=True)


def make_eval_env(render_mode=None):
    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    vec_env = make_vec_env(
        "Reacher-v5",
        n_envs=1,
        env_kwargs=env_kwargs,
    )

    vec_env = VecNormalize.load(str(VEC_STATS_PATH), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"没有找到模型文件：{MODEL_PATH}\n请先运行新的 train_reacher.py"
        )
    return PPO.load(str(MODEL_PATH), device="auto")


def play_human(episodes=10):
    model = load_model()
    env = make_eval_env(render_mode="human")

    obs = env.reset()
    ep_reward = 0.0
    ep_count = 0

    while ep_count < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        ep_reward += float(rewards[0])

        if dones[0]:
            print(f"[human] episode={ep_count}, reward={ep_reward:.3f}")
            ep_reward = 0.0
            ep_count += 1

    env.close()


def record_video(video_length=500):
    model = load_model()

    env = make_eval_env(render_mode="rgb_array")
    env = VecVideoRecorder(
        env,
        video_folder=str(VIDEO_DIR),
        record_video_trigger=lambda step_id: step_id == 0,
        video_length=video_length,
        name_prefix="reacher_ppo_longrun",
    )

    obs = env.reset()

    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

    env.close()
    print(f"视频已保存到: {VIDEO_DIR}")


if __name__ == "__main__":
    record_video(video_length=500)   # 大约 10 个 episode 的长度
    play_human(episodes=10)