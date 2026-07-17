from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


# Gymnasium에 로봇 환경 등록
gym.register_envs(gymnasium_robotics)

MODEL_DIR = Path("models")
LOG_DIR = Path("logs")
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def make_env():
    # 학습할 때는 렌더링을 끈다.
    env = gym.make(
        # 7DOF Fetch 로봇팔이 목표 위치에 end-effector를 도달시키는 Task
        # Dense 버전이라 '거리 기반 연속 보상'을 받음
        # Sparse 버전 (FetchReach-v4)이면 목표 도달 전까지 보상이 항상 -1이라 학습이 훨씬 어려움
        "FetchReachDense-v4",
        # 한 에피소드가 최대 몇스텝까지 진행될 수 있는가
        max_episode_steps=50,
    )
    return Monitor(env, filename=str(LOG_DIR / "training"))


env = make_env()

# PPO는 onpolicy라 매 업데이트마다 새로 수집한 데이터만 씀 -> sample efficiency 낮음
model = PPO(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=3e-4,
    # n_steps=1024짜리 배치를 batch size=256, n_epoch=10로 나눠서 학습
    # env를 1024번 실행해서 (s, a, r, ..) 튜플 1024개를 모음 -> 1024개 얘네를 rollout buffer라고 함
    n_steps=1024, # 업데이트 전 몇 step 데이터를 모을건가
    batch_size=256, #1024개를 256개씩 잘라서 미니배치로 학습
    n_epochs=10, # 1024개(4개의 minibatch)를 10번 반복해서 돌면서 학습 -> 총 40번의 gradient update
    gamma=0.98, # 미래 보상에 대한 discount factor
    gae_lambda=0.95,
    verbose=1,
    device="cpu",
    seed=42,
)

print("PPO 학습 시작")

model.learn(
    # on policy라 sample efficiency가 훨씬 떨어져서 200,000 스텝으로 수렴 안될 가능성이 높음
    total_timesteps=200_000,
    progress_bar=False,
)

model.save(MODEL_DIR / "ppo_fetch_reach")
env.close()

print("학습 완료")
print("저장 위치: models/ppo_fetch_reach.zip")
