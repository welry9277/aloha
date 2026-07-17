import time

import gymnasium as gym
import gymnasium_robotics

print("환경 등록 중...")
gym.register_envs(gymnasium_robotics)

print("FetchReach 생성 중...")
env = gym.make("FetchReach-v4", render_mode="human")

print("환경 실행 시작!")
observation, info = env.reset(seed=42)

try:
    for _ in range(1_000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(0.02)
finally:
    env.close()
