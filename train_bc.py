from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bc_model import BCPolicy


SEED = 42
BATCH_SIZE = 256
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
VALIDATION_RATIO = 0.2
PATIENCE = 15

DATA_PATH = Path("demonstrations/ppo_expert_demos.npz")
MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_PATH = MODEL_DIR / "bc_fetch_reach.pt"
GRAPH_PATH = RESULT_DIR / "bc_learning_curve.png"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"사용 장치: {device}")


# 1. 데이터 불러오기
data = np.load(DATA_PATH)

observations = data["observations"].astype(np.float32)
actions = data["actions"].astype(np.float32)
episode_ids = data["episode_ids"]

print(f"전체 관측 데이터: {observations.shape}")
print(f"전체 행동 데이터: {actions.shape}")


# 2. 에피소드 단위 분할. unique: 중복 제거하고 자동정렬
unique_episode_ids = np.unique(episode_ids)

rng = np.random.default_rng(SEED)
rng.shuffle(unique_episode_ids)

validation_episode_count = int(
    len(unique_episode_ids) * VALIDATION_RATIO
)

validation_episode_ids = unique_episode_ids[
    :validation_episode_count
]
training_episode_ids = unique_episode_ids[
    validation_episode_count:
]

training_mask = np.isin(
    episode_ids,
    training_episode_ids,
)
validation_mask = np.isin(
    episode_ids,
    validation_episode_ids,
)

x_train = observations[training_mask]
y_train = actions[training_mask]

x_validation = observations[validation_mask]
y_validation = actions[validation_mask]

print(f"학습 에피소드: {len(training_episode_ids)}")
print(f"검증 에피소드: {len(validation_episode_ids)}")
print(f"학습 샘플: {len(x_train)}")
print(f"검증 샘플: {len(x_validation)}")


# 3. 학습 데이터 통계만 사용해 관측값 정규화
observation_mean = x_train.mean(axis=0)
observation_std = x_train.std(axis=0)

# 변화가 거의 없는 특성의 0 나누기 방지
observation_std = np.maximum(
    observation_std,
    1e-6,
)

x_train = (
    x_train - observation_mean
) / observation_std

x_validation = (
    x_validation - observation_mean
) / observation_std


# 4. PyTorch Dataset 생성
training_dataset = TensorDataset(
    torch.tensor(x_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)

validation_dataset = TensorDataset(
    torch.tensor(x_validation, dtype=torch.float32),
    torch.tensor(y_validation, dtype=torch.float32),
)

training_loader = DataLoader(
    training_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# 5. 모델과 손실 함수 생성
model = BCPolicy(
    observation_dim=observations.shape[1],
    action_dim=actions.shape[1],
).to(device)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# 6. 학습
training_losses = []
validation_losses = []

best_validation_loss = float("inf")
epochs_without_improvement = 0

print("\nBehavior Cloning 학습 시작")

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()

    total_training_loss = 0.0
    training_sample_count = 0

    for batch_observations, batch_actions in training_loader:
        batch_observations = batch_observations.to(device)
        batch_actions = batch_actions.to(device)

        predicted_actions = model(batch_observations)
        loss = loss_function(
            predicted_actions,
            batch_actions,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch_observations.size(0)

        total_training_loss += (
            loss.item() * batch_size
        )
        training_sample_count += batch_size

    mean_training_loss = (
        total_training_loss / training_sample_count
    )

    # 검증
    model.eval()

    total_validation_loss = 0.0
    validation_sample_count = 0

    with torch.no_grad():
        for batch_observations, batch_actions in validation_loader:
            batch_observations = batch_observations.to(device)
            batch_actions = batch_actions.to(device)

            predicted_actions = model(batch_observations)
            loss = loss_function(
                predicted_actions,
                batch_actions,
            )

            batch_size = batch_observations.size(0)

            total_validation_loss += (
                loss.item() * batch_size
            )
            validation_sample_count += batch_size

    mean_validation_loss = (
        total_validation_loss / validation_sample_count
    )

    training_losses.append(mean_training_loss)
    validation_losses.append(mean_validation_loss)

    print(
        f"Epoch {epoch:03d} | "
        f"train MSE={mean_training_loss:.6f} | "
        f"validation MSE={mean_validation_loss:.6f}"
    )

    # 검증 오차가 가장 낮은 모델 저장
    if mean_validation_loss < best_validation_loss:
        best_validation_loss = mean_validation_loss
        epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "observation_mean": torch.tensor(
                observation_mean,
                dtype=torch.float32,
            ),
            "observation_std": torch.tensor(
                observation_std,
                dtype=torch.float32,
            ),
            "observation_dim": observations.shape[1],
            "action_dim": actions.shape[1],
            "best_validation_loss": best_validation_loss,
            "epoch": epoch,
        }

        torch.save(checkpoint, MODEL_PATH)

    else:
        epochs_without_improvement += 1

    # 오버피팅 방지를 위한 조기 종료
    if epochs_without_improvement >= PATIENCE:
        print(
            f"\n검증 성능이 {PATIENCE}회 연속 "
            "개선되지 않아 조기 종료합니다."
        )
        break


# 7. 학습 곡선 저장
epochs = np.arange(
    1,
    len(training_losses) + 1,
)

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    training_losses,
    label="Training MSE",
)
plt.plot(
    epochs,
    validation_losses,
    label="Validation MSE",
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Behavior Cloning Learning Curve")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    GRAPH_PATH,
    dpi=200,
)
plt.close()

print("\nBehavior Cloning 학습 완료")
print(f"최적 검증 MSE: {best_validation_loss:.6f}")
print(f"모델 저장 위치: {MODEL_PATH}")
print(f"학습 곡선: {GRAPH_PATH}")
