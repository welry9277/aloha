"""Collect Left PnP demonstrations from post-Right-push boundary states.

The scripted right tray push is executed but not recorded.  Recording starts
only after the push and retreat have completed, so every saved NPZ is a Left
pick-and-place episode whose initial simulator state is the phase boundary.
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aloha.demonstration_io import DemonstrationRecorder
from aloha.task_env import AlohaTaskEnvironment
from aloha.task_instructions import TASK_INSTRUCTIONS
from evaluation.evaluate_hybrid_transition import scripted_primitive, site_id
from evaluation.evaluate_language_act_suite import geom_labels


TASK_NAME = "left_pick_place_after_right_push"


def setup_unseen_scene(env, seed):
    """Match the randomized scene distribution used by the unseen RL expert."""
    rng = np.random.default_rng(seed)
    tray_y = rng.uniform(0.135, 0.165)
    mirrored_start_x = rng.uniform(-0.060, -0.040)
    mirrored_goal_x = mirrored_start_x + rng.uniform(0.145, 0.165)
    block_x = rng.uniform(-0.035, 0.035)

    tray_start = np.array([-mirrored_start_x, tray_y, 0.018])
    tray_goal = np.array([-mirrored_goal_x, tray_y, 0.018])
    block_start = np.array(
        [-block_x, rng.uniform(-0.140, -0.105), 0.025]
    )

    env.reset(randomize=False)
    env._set_freejoint_pose(env.tray_joint, tray_start)
    env._set_freejoint_pose(env.block_joint, block_start)
    target = site_id(env.model, "target")
    env.model.site_pos[target] = np.array(
        [tray_goal[0], tray_goal[1], 0.012]
    )
    env.data.qvel[:] = 0.0
    env.data.qacc[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    return tray_goal


def collect_attempt(path, seed):
    env = AlohaTaskEnvironment(seed=seed)
    tray_goal = setup_unseen_scene(env, seed)
    labels = geom_labels(env.model)

    push = scripted_primitive(
        env,
        task="tray_push",
        arm="right",
        tray_goal=tray_goal,
        labels=labels,
    )
    if not push["success"]:
        print(
            f"discarding seed={seed}: right push failed "
            f"(tray_error={push['tray_error']:.4f})",
            flush=True,
        )
        return False

    recorder = DemonstrationRecorder(
        env.model,
        env.data,
        path,
        TASK_INSTRUCTIONS[TASK_NAME],
        tray_goal,
    )
    try:
        pick_place = scripted_primitive(
            env,
            task="pick_place",
            arm="left",
            tray_goal=tray_goal,
            labels=labels,
            recorder=recorder,
        )
        success = bool(pick_place["success"] and pick_place["tray_ok"])
        recorder.save(success)
    except BaseException:
        recorder.renderer.close()
        raise

    print(
        f"seed={seed}: push=True left_pnp={pick_place['success']} "
        f"tray_ok={pick_place['tray_ok']} success={success}",
        flush=True,
    )
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Collect post-Right-push Left PnP demonstrations"
    )
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue until --episodes total successful files exist.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.output.glob("episode_*.npz"))
    if existing and not args.resume:
        parser.error(
            f"{args.output} already contains {len(existing)} episodes; "
            "use --resume or choose an empty output directory"
        )
    max_attempts = args.max_attempts or args.episodes * 3
    successes = len(existing)
    attempts = 0

    while successes < args.episodes and attempts < max_attempts:
        seed = args.seed + attempts
        path = args.output / f"episode_{successes:04d}.npz"
        print(
            f"\n=== boundary attempt {attempts + 1}/{max_attempts}, "
            f"success {successes}/{args.episodes}, seed={seed} ===",
            flush=True,
        )
        success = collect_attempt(path, seed)
        attempts += 1
        if success:
            successes += 1
            print(f"kept successful episode: {path}", flush=True)
        elif path.exists():
            if args.keep_failures:
                failure_dir = args.output / "failures"
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure_path = failure_dir / f"attempt_{attempts - 1:04d}.npz"
                path.replace(failure_path)
                print(f"kept failed episode: {failure_path}", flush=True)
            else:
                path.unlink()
                print("discarded failed episode", flush=True)

    rate = successes / attempts if attempts else 0.0
    print(
        f"attempts={attempts}, collected_successes={successes}, "
        f"success_rate={rate:.1%}",
        flush=True,
    )
    if successes < args.episodes:
        raise SystemExit(
            f"Only collected {successes}/{args.episodes} successful episodes"
        )


if __name__ == "__main__":
    main()
