from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import PPO


gym.register_envs(gymnasium_robotics)

MODEL_PATH = "models/ppo_fetch_reach" # 아까 ppo 학습시킨 모델
DATA_DIR = Path("demonstrations")
DATA_PATH = DATA_DIR / "ppo_expert_demos.npz"

TARGET_SAMPLES = 20_000
MAX_EPISODE_STEPS = 50

DATA_DIR.mkdir(exist_ok=True)


def flatten_observation(observation):
    """
    Gymnasium의 Dict 관측을 하나의 벡터로 합친다.

    observation:
        로봇 관절 및 그리퍼 상태
    achieved_goal:
        현재 그리퍼 위치
    desired_goal:
        목표 위치
    """
    return np.concatenate(
        [
            observation["observation"],
            observation["achieved_goal"],
            observation["desired_goal"],
        ]
    ).astype(np.float32)


print("PPO 전문가 모델 불러오는 중...")
expert = PPO.load(MODEL_PATH, device="cpu")

env = gym.make(
    "FetchReachDense-v4",
    max_episode_steps=MAX_EPISODE_STEPS,
)

all_observations = []
all_actions = []
all_rewards = []
all_episode_ids = []

attempted_episodes = 0
successful_episodes = 0

print(f"목표: 성공한 시범 데이터 {TARGET_SAMPLES:,}개")

while len(all_actions) < TARGET_SAMPLES:
    attempted_episodes += 1

    observation, info = env.reset(seed=10_000 + attempted_episodes)

    episode_observations = []
    episode_actions = []
    episode_rewards = []
    episode_success = False

    for _ in range(MAX_EPISODE_STEPS):
        # 행동을 실행하기 전 상태를 저장한다.
        flat_observation = flatten_observation(observation)

        action, _ = expert.predict(
            observation,
            deterministic=True,
        )
        action = np.asarray(action, dtype=np.float32)

        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_observations.append(flat_observation)
        episode_actions.append(action.copy())
        episode_rewards.append(float(reward))

        observation = next_observation

        if bool(info.get("is_success", False)):
            episode_success = True
            break

        if terminated or truncated:
            break

    # 성공한 trajectory만 전문가 시범으로 사용한다.
    if episode_success:
        episode_id = successful_episodes
        successful_episodes += 1

        all_observations.extend(episode_observations)
        all_actions.extend(episode_actions)
        all_rewards.extend(episode_rewards)
        all_episode_ids.extend([episode_id] * len(episode_actions))

    if attempted_episodes % 100 == 0:
        print(
            f"시도={attempted_episodes:,} | "
            f"성공={successful_episodes:,} | "
            f"샘플={len(all_actions):,}/{TARGET_SAMPLES:,}"
        )

env.close()

# 목표 개수에 맞춰 자른다.
observations = np.asarray(
    all_observations[:TARGET_SAMPLES],
    dtype=np.float32,
)
actions = np.asarray(
    all_actions[:TARGET_SAMPLES],
    dtype=np.float32,
)
rewards = np.asarray(
    all_rewards[:TARGET_SAMPLES],
    dtype=np.float32,
)
episode_ids = np.asarray(
    all_episode_ids[:TARGET_SAMPLES],
    dtype=np.int32,
)

np.savez_compressed(
    DATA_PATH,
    observations=observations,
    actions=actions,
    rewards=rewards,
    episode_ids=episode_ids,
)

print()
print("시범 데이터 수집 완료")
print(f"저장 위치: {DATA_PATH}")
print(f"관측 데이터 크기: {observations.shape}")
print(f"행동 데이터 크기: {actions.shape}")
print(f"전체 시도 에피소드: {attempted_episodes}")
print(f"성공 에피소드: {successful_episodes}")
