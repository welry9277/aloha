import argparse
from pathlib import Path


def task_runner(task):
    if task == "seen_lr":
        from expert_tray_push_pick_place import run_episode
    elif task == "left_pick_place":
        from test_left_grasp import run_episode
    elif task == "right_pick_place":
        from test_right_grasp import run_episode
    else:
        raise ValueError(f"Unknown task: {task}")
    return run_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("seen_lr", "left_pick_place", "right_pick_place"),
        default="seen_lr",
        help="Task expert to collect. Default keeps the old L-push + R-place behavior.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("demonstrations"))
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--keep-failures", action="store_true")
    args = parser.parse_args()

    run_episode = task_runner(args.task)
    args.output.mkdir(parents=True, exist_ok=True)
    successes = 0
    attempts = 0
    max_attempts = args.max_attempts or max(args.episodes * 3, args.episodes)

    while successes < args.episodes and attempts < max_attempts:
        seed = args.seed + attempts
        path = args.output / f"episode_{successes:04d}.npz"
        print(
            f"\n=== task={args.task}, attempt {attempts + 1}/{max_attempts}, "
            f"success {successes}/{args.episodes}, seed={seed} ==="
        )
        success = run_episode(
            record_path=path,
            show_viewer=args.viewer,
            seed=seed,
            randomize=args.randomize,
        )
        attempts += 1
        if success:
            successes += 1
            print(f"kept successful episode: {path}")
        else:
            print("episode failed")
            if args.keep_failures:
                failure_dir = args.output / "failures"
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure_path = failure_dir / f"attempt_{attempts - 1:04d}.npz"
                path.replace(failure_path)
                print(f"kept failure for debugging: {failure_path}")
            elif path.exists():
                path.unlink()
                print("discarded failed episode")

    rate = successes / attempts if attempts else 0.0
    print(
        f"attempts={attempts}, collected_successes={successes}, "
        f"success_rate={rate:.1%}"
    )
    if successes < args.episodes:
        raise SystemExit(
            f"Only collected {successes}/{args.episodes} successful episodes"
        )


if __name__ == "__main__":
    main()
