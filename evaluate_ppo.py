import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import PPO


gym.register_envs(gymnasium_robotics)

EPISODES = 100
BASE_SEED = 20_000

env = gym.make(
    "FetchReachDense-v4",
    max_episode_steps=50,
)

model = PPO.load(
    "models/ppo_fetch_reach",
    env=env,
    device="cpu",
)

successes = 0
episode_rewards = []
final_distances = []

try:
    for episode in range(1, EPISODES + 1):
        observation, info = env.reset(
            seed=BASE_SEED + episode
        )

        episode_reward = 0.0
        final_distance = None
        final_success = False

        while True:
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = env.step(
                action
            )

            episode_reward += float(reward)

            final_distance = float(
                np.linalg.norm(
                    observation["achieved_goal"]
                    - observation["desired_goal"]
                )
            )

            final_success = bool(
                info.get("is_success", False)
            )

            if terminated or truncated:
                break

        successes += int(final_success)
        episode_rewards.append(episode_reward)
        final_distances.append(final_distance)

        print(
            f"Episode {episode:03d} | "
            f"reward={episode_reward:8.3f} | "
            f"distance={final_distance:.4f}m | "
            f"success={final_success}"
        )

finally:
    env.close()


print("\nPPO 평가 결과")
print(
    f"성공률: {successes}/{EPISODES} "
    f"= {successes / EPISODES:.1%}"
)
print(
    f"평균 누적 보상: "
    f"{np.mean(episode_rewards):.3f}"
)
print(
    f"평균 최종 거리: "
    f"{np.mean(final_distances):.4f}m"
)
print(
    f"최종 거리 표준편차: "
    f"{np.std(final_distances):.4f}m"
)
