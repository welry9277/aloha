import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from task_instructions import instruction_from_path, task_from_path


CAMERA_KEYS = (
    "images_overhead_cam",
    "images_wrist_cam_left",
    "images_wrist_cam_right",
)
REQUIRED_KEYS = CAMERA_KEYS + ("states", "actions", "success")


def discover_episode_paths(data_dirs, max_episodes=None):
    paths = []
    for data_dir in data_dirs:
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        paths.extend(root.rglob("episode_*.npz"))
    paths = sorted({path.resolve() for path in paths})
    if max_episodes is not None:
        paths = paths[:max_episodes]
    if not paths:
        raise ValueError(f"No episode_*.npz files found in: {data_dirs}")
    return paths


def inspect_episode(path):
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"{path}: missing keys {missing}")
        length = int(data["actions"].shape[0])
        if data["states"].shape != (length, 14):
            raise ValueError(f"{path}: states shape={data['states'].shape}")
        if data["actions"].shape != (length, 14):
            raise ValueError(f"{path}: actions shape={data['actions'].shape}")
        for key in CAMERA_KEYS:
            if data[key].shape != (length, 224, 224, 3):
                raise ValueError(f"{path}: {key} shape={data[key].shape}")
        if not bool(data["success"].item()):
            raise ValueError(f"{path}: unsuccessful episode in training data")
    return length


def compute_normalization_stats(episode_paths):
    state_sum = np.zeros(14, dtype=np.float64)
    state_sq_sum = np.zeros(14, dtype=np.float64)
    action_sum = np.zeros(14, dtype=np.float64)
    action_sq_sum = np.zeros(14, dtype=np.float64)
    count = 0

    for path in episode_paths:
        with np.load(path, allow_pickle=False) as data:
            states = np.asarray(data["states"], dtype=np.float64)
            actions = np.asarray(data["actions"], dtype=np.float64)
        state_sum += states.sum(axis=0)
        state_sq_sum += np.square(states).sum(axis=0)
        action_sum += actions.sum(axis=0)
        action_sq_sum += np.square(actions).sum(axis=0)
        count += len(states)

    state_mean = state_sum / count
    action_mean = action_sum / count
    state_var = np.maximum(state_sq_sum / count - np.square(state_mean), 1e-8)
    action_var = np.maximum(action_sq_sum / count - np.square(action_mean), 1e-8)
    return {
        "state_mean": state_mean.astype(np.float32),
        "state_std": np.sqrt(state_var).astype(np.float32),
        "action_mean": action_mean.astype(np.float32),
        "action_std": np.sqrt(action_var).astype(np.float32),
        "num_steps": int(count),
    }


def save_normalization_stats(stats, output_path):
    payload = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in stats.items()
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_normalization_stats(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("state_mean", "state_std", "action_mean", "action_std"):
        payload[key] = np.asarray(payload[key], dtype=np.float32)
    return payload


class _EpisodeCache:
    def __init__(self, max_episodes=2):
        self.max_episodes = max_episodes
        self.cache = OrderedDict()

    def get(self, path):
        path = str(path)
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]

        with np.load(path, allow_pickle=False) as data:
            episode = {
                "states": data["states"].astype(np.float32),
                "actions": data["actions"].astype(np.float32),
                **{key: data[key].astype(np.uint8) for key in CAMERA_KEYS},
            }
        self.cache[path] = episode
        while len(self.cache) > self.max_episodes:
            self.cache.popitem(last=False)
        return episode


class AlohaNPZActionChunkDataset(Dataset):
    def __init__(
        self,
        episode_paths,
        chunk_size,
        stats,
        cache_size=2,
    ):
        self.episode_paths = [Path(path) for path in episode_paths]
        self.chunk_size = int(chunk_size)
        self.stats = stats
        self.cache = _EpisodeCache(cache_size)
        self.lengths = [inspect_episode(path) for path in self.episode_paths]
        self.index = [
            (episode_index, timestep)
            for episode_index, length in enumerate(self.lengths)
            for timestep in range(length)
        ]
        self.instructions = [instruction_from_path(path) for path in self.episode_paths]
        self.tasks = [task_from_path(path) for path in self.episode_paths]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        episode_index, timestep = self.index[item]
        path = self.episode_paths[episode_index]
        episode = self.cache.get(path)

        images = np.stack(
            [episode[key][timestep] for key in CAMERA_KEYS], axis=0
        )
        images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32) / 255.0

        state = (episode["states"][timestep] - self.stats["state_mean"]) / self.stats[
            "state_std"
        ]

        action_chunk = np.zeros((self.chunk_size, 14), dtype=np.float32)
        is_pad = np.ones(self.chunk_size, dtype=np.bool_)
        available = min(self.chunk_size, len(episode["actions"]) - timestep)
        raw_actions = episode["actions"][timestep : timestep + available]
        action_chunk[:available] = (
            raw_actions - self.stats["action_mean"]
        ) / self.stats["action_std"]
        is_pad[:available] = False

        return {
            "images": torch.from_numpy(images),
            "state": torch.from_numpy(state.astype(np.float32)),
            "actions": torch.from_numpy(action_chunk),
            "is_pad": torch.from_numpy(is_pad),
            "instruction": self.instructions[episode_index],
            "task": self.tasks[episode_index],
            "episode_path": str(path),
            "timestep": timestep,
        }
