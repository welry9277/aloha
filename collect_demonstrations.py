import argparse
from pathlib import Path

from expert_primitives import run_primitive_episode
from expert_tray_push_pick_place import run_episode as run_full_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("full", "tray_push", "pick_place"), default="full")
    parser.add_argument("--arm", choices=("left", "right"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("demonstrations"))
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--keep-failures", action="store_true")
    args = parser.parse_args()

    if args.task != "full" and args.arm is None:
        parser.error("--arm is required for tray_push and pick_place")
    if args.task == "full" and args.arm is not None:
        parser.error("--arm is only valid for tray_push and pick_place")

    args.output.mkdir(parents=True, exist_ok=True)
    successes = 0
    attempts = 0
    max_attempts = args.max_attempts or max(args.episodes * 3, args.episodes)

    while successes < args.episodes and attempts < max_attempts:
        seed = args.seed + attempts
        path = args.output / f"episode_{successes:04d}.npz"
        print(
            f"\n=== attempt {attempts + 1}/{max_attempts}, "
            f"success {successes}/{args.episodes}, seed={seed}, "
            f"task={args.task}, arm={args.arm} ===",
            flush=True,
        )

        if args.task == "full":
            success = run_full_episode(
                record_path=path,
                show_viewer=args.viewer,
                seed=seed,
                randomize=args.randomize,
            )
        else:
            success = run_primitive_episode(
                task=args.task,
                arm=args.arm,
                record_path=path,
                show_viewer=args.viewer,
                seed=seed,
                randomize=args.randomize,
            )

        attempts += 1
        if success:
            successes += 1
            print(f"kept successful episode: {path}", flush=True)
        else:
            print("episode failed", flush=True)
            if args.keep_failures:
                failure_dir = args.output / "failures"
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure_path = failure_dir / f"attempt_{attempts - 1:04d}.npz"
                path.replace(failure_path)
                print(f"kept failure for debugging: {failure_path}", flush=True)
            elif path.exists():
                path.unlink()
                print("discarded failed episode", flush=True)

    rate = successes / attempts if attempts else 0.0
    print(
        f"attempts={attempts}, collected_successes={successes}, "
        f"success_rate={rate:.1%}",
        flush=True,
    )
    if successes < args.episodes:
        raise SystemExit(f"Only collected {successes}/{args.episodes} successful episodes")


if __name__ == "__main__":
    main()
