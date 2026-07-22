"""Batch closed-loop evaluation for the ALOHA 2 role-composition project.

The script restores only the initial simulator state from each held-out NPZ.
Demonstration actions and images are never used by the policy during rollout.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aloha.task_env import AlohaTaskEnvironment
from aloha.demonstration_io import actuator_qpos
from evaluation.evaluate_language_act import (
    ACTPolicy,
    clip_action,
    load_stats,
    render_observation,
    restore_episode,
    task_success,
)
from aloha.task_instructions import instruction_from_path, task_from_path


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return [None, None]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def task_result(task, result):
    if task.endswith("tray_push"):
        return result["tray_ok"]
    if task.endswith("pick_place"):
        return result["block_placement_ok"]
    return result["tray_ok"] and result["block_placement_ok"]


def role_assignment(task):
    if task == "seen_lr":
        return "left", "right"
    if task == "unseen_rl":
        return "right", "left"
    if task.startswith("left_"):
        return ("left", None) if task.endswith("tray_push") else (None, "left")
    if task.startswith("right_"):
        return ("right", None) if task.endswith("tray_push") else (None, "right")
    return None, None


def geom_labels(model):
    """Map each geom to arm/object labels using its body ancestry."""
    labels = []
    for geom_id in range(model.ngeom):
        names = []
        body_id = int(model.geom_bodyid[geom_id])
        while body_id > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name:
                names.append(name.lower())
            body_id = int(model.body_parentid[body_id])
        joined = " ".join(names)
        item = set()
        if "left/" in joined or "left_" in joined:
            item.add("left")
        if "right/" in joined or "right_" in joined:
            item.add("right")
        if "tray" in joined:
            item.add("tray")
        if "red_block" in joined or "block" in joined:
            item.add("block")
        labels.append(item)
    return labels


def update_contacts(data, labels, contacts):
    for index in range(data.ncon):
        contact = data.contact[index]
        first = labels[int(contact.geom1)]
        second = labels[int(contact.geom2)]
        for arm in ("left", "right"):
            if (arm in first and "tray" in second) or (arm in second and "tray" in first):
                contacts[f"{arm}_tray"] = True
            if (arm in first and "block" in second) or (arm in second and "block" in first):
                contacts[f"{arm}_block"] = True


def rollout(policy, policy_config, stats, episode_path, device, execute_actions, max_actions):
    with np.load(episode_path, allow_pickle=False) as episode:
        env = AlohaTaskEnvironment(seed=0)
        tray_goal = restore_episode(env, episode)
        image_stride = int(episode["image_stride"])
        horizon = int(max_actions or episode["actions"].shape[0])

    task = task_from_path(episode_path)
    instruction = instruction_from_path(episode_path)
    mover, placer = role_assignment(task)
    chunk_size = int(policy_config["num_queries"])
    renderer = mujoco.Renderer(env.model, height=224, width=224)
    labels = geom_labels(env.model)
    contacts = {
        "left_tray": False, "right_tray": False,
        "left_block": False, "right_block": False,
    }

    initial_obs = env.observation()
    initial_tray = np.asarray(initial_obs["tray_position"], dtype=np.float64).copy()
    initial_block = np.asarray(initial_obs["block_position"], dtype=np.float64).copy()
    initial_tray_error = float(np.linalg.norm(initial_tray[:2] - tray_goal[:2]))
    previous_state = actuator_qpos(env.model, env.data).astype(np.float64)
    tray_phase_motion = np.zeros(2, dtype=np.float64)
    place_phase_motion = np.zeros(2, dtype=np.float64)
    tray_reached = False
    max_block_lift = 0.0
    min_block_tray_xy_after_lift = float("inf")
    stable_place_steps = 0
    max_stable_place_steps = 0
    tray_post_reach_steps = 0
    tray_post_reach_ok_steps = 0
    actions_executed = 0
    started = time.perf_counter()

    try:
        while actions_executed < horizon:
            images = render_observation(renderer, env)
            state = actuator_qpos(env.model, env.data)
            normalized_state = (state - stats["state_mean"]) / stats["state_std"]
            image_tensor = torch.from_numpy(images).unsqueeze(0).to(device)
            state_tensor = torch.from_numpy(normalized_state).unsqueeze(0).to(device)

            with torch.inference_mode():
                normalized_chunk = policy(state_tensor, image_tensor, [instruction])[0].cpu().numpy()
            action_chunk = normalized_chunk * stats["action_std"] + stats["action_mean"]

            for action in action_chunk[: min(execute_actions, horizon - actions_executed)]:
                env.data.ctrl[:] = clip_action(env.model, action)
                for _ in range(image_stride):
                    mujoco.mj_step(env.model, env.data)
                    update_contacts(env.data, labels, contacts)
                actions_executed += 1

                current_state = actuator_qpos(env.model, env.data).astype(np.float64)
                delta = np.array(
                    [
                        np.abs(current_state[:7] - previous_state[:7]).sum(),
                        np.abs(current_state[7:] - previous_state[7:]).sum(),
                    ]
                )
                if tray_reached:
                    place_phase_motion += delta
                else:
                    tray_phase_motion += delta
                previous_state = current_state

                obs = env.observation()
                tray = np.asarray(obs["tray_position"])
                block = np.asarray(obs["block_position"])
                tray_error = float(np.linalg.norm(tray[:2] - tray_goal[:2]))
                tray_reached = tray_reached or tray_error < 0.070
                if tray_reached:
                    tray_post_reach_steps += 1
                    tray_post_reach_ok_steps += int(tray_error < 0.070)
                lift = float(block[2] - initial_block[2])
                max_block_lift = max(max_block_lift, lift)
                if lift >= 0.045:
                    distance = float(np.linalg.norm(block[:2] - tray[:2]))
                    min_block_tray_xy_after_lift = min(min_block_tray_xy_after_lift, distance)

                block_geometry_ok = (
                    abs(block[0] - tray[0]) < 0.11
                    and abs(block[1] - tray[1]) < 0.07
                    and block[2] - tray[2] < 0.08
                )
                current_qpos = actuator_qpos(env.model, env.data)
                gripper_open = placer is not None and current_qpos[6 if placer == "left" else 13] > 0.033
                block_dof = env.model.jnt_dofadr[env.block_joint]
                block_speed = float(np.linalg.norm(env.data.qvel[block_dof : block_dof + 3]))
                if block_geometry_ok and gripper_open and block_speed < 0.050:
                    stable_place_steps += 1
                    max_stable_place_steps = max(max_stable_place_steps, stable_place_steps)
                else:
                    stable_place_steps = 0
    finally:
        renderer.close()

    final = task_success(env, tray_goal)
    final_state = actuator_qpos(env.model, env.data)
    final_tray_error = float(final["tray_error"])
    progress = (
        (initial_tray_error - final_tray_error) / initial_tray_error
        if initial_tray_error > 1e-8
        else 0.0
    )
    side = {"left": 0, "right": 1}
    mover_dominant = None if mover is None else bool(
        tray_phase_motion[side[mover]] > tray_phase_motion[1 - side[mover]]
    )
    placer_dominant = None if placer is None or not tray_reached else bool(
        place_phase_motion[side[placer]] > place_phase_motion[1 - side[placer]]
    )
    placer_gripper = None if placer is None else float(final_state[6 if placer == "left" else 13])
    release_open = None if placer_gripper is None else bool(placer_gripper > 0.033)
    block_placement_ok = bool(final["block_ok"] and release_open and max_stable_place_steps >= 10)
    final["block_placement_ok"] = block_placement_ok

    return {
        "episode": str(episode_path),
        "task": task,
        "success": bool(task_result(task, final)),
        "full_success": bool(final["tray_ok"] and block_placement_ok),
        "tray_ok": bool(final["tray_ok"]),
        "block_ok": bool(final["block_ok"]),
        "grasp_lift_ok": bool(max_block_lift >= 0.045),
        "transport_ok": bool(min_block_tray_xy_after_lift < 0.080),
        "release_open": release_open,
        "stable_place_1s": bool(max_stable_place_steps >= 10),
        "mover_motion_dominant": mover_dominant,
        "placer_motion_dominant": placer_dominant,
        "mover_tray_contact": None if mover is None else contacts[f"{mover}_tray"],
        "wrong_arm_tray_contact": None if mover is None else contacts[f"{'right' if mover == 'left' else 'left'}_tray"],
        "placer_block_contact": None if placer is None else contacts[f"{placer}_block"],
        "wrong_arm_block_contact": None if placer is None else contacts[f"{'right' if placer == 'left' else 'left'}_block"],
        "initial_tray_error": initial_tray_error,
        "final_tray_error": final_tray_error,
        "tray_progress": progress,
        "max_block_lift": max_block_lift,
        "max_stable_place_steps": max_stable_place_steps,
        "tray_maintained_rate": (
            tray_post_reach_ok_steps / tray_post_reach_steps if tray_post_reach_steps else None
        ),
        "min_block_tray_xy_after_lift": (
            None if not np.isfinite(min_block_tray_xy_after_lift) else min_block_tray_xy_after_lift
        ),
        "left_tray_phase_motion": float(tray_phase_motion[0]),
        "right_tray_phase_motion": float(tray_phase_motion[1]),
        "left_place_phase_motion": float(place_phase_motion[0]),
        "right_place_phase_motion": float(place_phase_motion[1]),
        "actions": actions_executed,
        "wall_seconds": time.perf_counter() - started,
    }


def summarize(rows):
    rate_fields = (
        "success", "full_success", "tray_ok", "block_ok", "grasp_lift_ok",
        "transport_ok", "release_open", "stable_place_1s", "mover_motion_dominant", "placer_motion_dominant",
        "mover_tray_contact", "wrong_arm_tray_contact", "placer_block_contact", "wrong_arm_block_contact",
    )
    summary = {"episodes": len(rows)}
    for field in rate_fields:
        values = [bool(row[field]) for row in rows if row[field] is not None]
        successes = sum(values)
        summary[field] = {
            "count": len(values),
            "successes": successes,
            "rate": successes / len(values) if values else None,
            "wilson_95": wilson_interval(successes, len(values)),
        }
    for field in ("final_tray_error", "tray_progress", "max_block_lift", "actions", "wall_seconds"):
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        summary[field] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return summary


def main():
    parser = argparse.ArgumentParser(description="Batch ACT closed-loop evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute-actions", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=220)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    checkpoint_path = args.checkpoint.resolve()
    stats_path = args.stats.resolve() if args.stats else checkpoint_path.parent / "normalization_stats.json"
    stats = load_stats(stats_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy_config = dict(checkpoint["model_config"])
    policy_config["device"] = str(device)
    policy = ACTPolicy(policy_config).to(device)
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.eval()

    chunk_size = int(policy_config["num_queries"])
    if not 1 <= args.execute_actions <= chunk_size:
        parser.error(f"--execute-actions must be between 1 and {chunk_size}")
    paths = sorted(args.episode_dir.glob("episode_*.npz"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        parser.error(f"No episode_*.npz files in {args.episode_dir}")

    rows = []
    for index, path in enumerate(paths, 1):
        row = rollout(policy, policy_config, stats, path, device, args.execute_actions, args.max_actions)
        rows.append(row)
        print(
            f"[{index:03d}/{len(paths):03d}] task={row['task']} success={row['success']} "
            f"tray={row['tray_ok']} block={row['block_ok']} lift={row['grasp_lift_ok']}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary.update(
        checkpoint=str(checkpoint_path), episode_dir=str(args.episode_dir.resolve()),
        execute_actions=args.execute_actions, max_actions=args.max_actions,
    )
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {csv_path}\nsaved: {summary_path}")


if __name__ == "__main__":
    main()
