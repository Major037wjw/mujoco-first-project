import gymnasium as gym


def main():
    env = gym.make("Reacher-v5", render_mode="rgb_array")

    obs, info = env.reset(seed=42)
    print("环境创建成功")
    print("action_space =", env.action_space)
    print("observation_space =", env.observation_space)
    print("obs.shape =", obs.shape)
    print("初始 info =", info)

    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"step={step:02d}, reward={reward:.4f}, "
            f"terminated={terminated}, truncated={truncated}"
        )

        if terminated or truncated:
            obs, info = env.reset()

    frame = env.render()
    print("渲染帧 shape =", frame.shape)
    print("10 步随机动作累计奖励 =", total_reward)

    env.close()


if __name__ == "__main__":
    main()