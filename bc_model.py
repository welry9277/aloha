import torch
from torch import nn


class BCPolicy(nn.Module):
    def __init__(self, observation_dim=16, action_dim=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim),

            # MuJoCo 행동 범위가 -1~1이므로 출력 범위 제한
            nn.Tanh(),
        )

    def forward(self, observation):
        return self.network(observation)
