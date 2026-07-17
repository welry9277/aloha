import argparse
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from bc_model import BCPolicy


def flatten_observation(observation):
    return np.concatenate(
        [
            observation["observation"],
            observation["achieved_goal"],
            observation["desired_goal"],
        ]
    ).astype(np.float32)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--episodes",
    type=int,
    default=100,
)

parser.add_argument(
    "--render",
    action="store_true",
)

args = parser.parse_args()


gym.register_envs(gymnasium_robotics)

render_mode = "human" if args.render else None

env = gym.make(
    "FetchReachDense-v4",
    render_mode=render_mode,
    max_episode_steps=50,
)


# 모델 체크포인트 불러오기
checkpoint = torch.load(
    "models/bc_fetch_reach.pt",
    map_location="cpu",
)

model = BCPolicy(
    observation_dim=checkpoint["observation_dim"],
    action_dim=checkpoint["action_dim"],
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)
model.eval()

observation_mean = (
    checkpoint["observation_mean"].numpy()
)
observation_std = (
    checkpoint["observation_std"].numpy()
)

successes = 0
episode_rewards = []
final_distances = []

try:
    for episode in range(1, args.episodes + 1):
        observation, info = env.reset(
            seed=20_000 + episode
        )

        episode_reward = 0.0
        final_success = False
        final_distance = None

        while True:
            flat_observation = flatten_observation(
                observation
            )

            normalized_observation = (
                flat_observation - observation_mean
            ) / observation_std

            observation_tensor = torch.tensor(
                normalized_observation,
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():
                action = model(
                    observation_tensor
                ).squeeze(0).numpy()

            action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high,
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

            if args.render:
                time.sleep(0.04)

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


success_rate = successes / args.episodes

print("\nBehavior Cloning 평가 결과")
print(
    f"성공률: {successes}/{args.episodes} "
    f"= {success_rate:.1%}"
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
