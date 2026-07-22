import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.language_act_dataset import CAMERA_KEYS, discover_episode_paths, inspect_episode
from aloha.task_instructions import TASK_INSTRUCTIONS, task_from_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", nargs="+")
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()

    paths = discover_episode_paths(args.data_dirs, args.max_episodes)
    task_counts = Counter()
    task_steps = Counter()
    errors = []

    for path in paths:
        try:
            length = inspect_episode(path)
            with np.load(path, allow_pickle=False) as data:
                for key in ("states", "actions"):
                    if not np.isfinite(data[key]).all():
                        raise ValueError(f"{key} contains NaN or Inf")
                for key in CAMERA_KEYS:
                    if data[key].dtype != np.uint8:
                        raise ValueError(f"{key} dtype={data[key].dtype}, expected uint8")
            task = task_from_path(path)
            task_counts[task] += 1
            task_steps[task] += length
        except Exception as error:
            errors.append((path, str(error)))

    print(f"episodes={len(paths)}, errors={len(errors)}")
    for task in sorted(task_counts):
        print(
            f"{task}: episodes={task_counts[task]}, steps={task_steps[task]}, "
            f"instruction={TASK_INSTRUCTIONS[task]}"
        )
    for path, error in errors[:20]:
        print(f"ERROR {path}: {error}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
