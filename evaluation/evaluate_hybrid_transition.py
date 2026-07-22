"""Hybrid controller evaluation for the unseen ALOHA 2 role composition.

The evaluator keeps one MuJoCo environment alive while control changes at the
phase boundary.  It supports the two causal diagnostics used in the report:

* scripted right tray push -> ACT left pick-and-place
* ACT right tray push -> scripted left pick-and-place

An expert -> expert oracle is also provided as a sanity check.  Held-out NPZ
files are used only to restore initial simulator state and choose the rollout
horizon metadata; recorded images/actions are never fed to either controller.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aloha.controller import AlohaArmController
from aloha.demonstration_io import actuator_qpos
from aloha.task_env import AlohaTaskEnvironment
from aloha.task_instructions import TASK_INSTRUCTIONS
from evaluation.evaluate_language_act import (
    ACTPolicy,
    clip_action,
    load_stats,
    render_observation,
    restore_episode,
    task_success,
)
from evaluation.evaluate_language_act_suite import (
    geom_labels,
    update_contacts,
    wilson_interval,
)


def site_id(model, name):
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if value == -1:
        raise ValueError(f"Site not found: {name}")
    return value


def finger_midpoint(data, sites):
    return 0.5 * (data.site_xpos[sites[0]] + data.site_xpos[sites[1]])


def midpoint_target(data, controller, current_midpoint, desired_midpoint):
    gripper_position, _ = controller.pose(data)
    return gripper_position + (desired_midpoint - current_midpoint)


def arm_setup(env, arm):
    active = AlohaArmController(env.model, arm)
    idle_name = "right" if arm == "left" else "left"
    idle = AlohaArmController(env.model, idle_name)
    active_initial = env.data.qpos[active.qpos_addresses].copy()
    idle_initial = env.data.qpos[idle.qpos_addresses].copy()
    sites = (
        site_id(env.model, f"{arm}/left_finger"),
        site_id(env.model, f"{arm}/right_finger"),
    )
    finger_joint = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/left_finger"
    )
    finger_qpos = env.model.jnt_qposadr[finger_joint]
    return active, idle, active_initial, idle_initial, sites, finger_qpos


def hold_idle(env, idle, idle_target):
    env.data.ctrl[idle.actuator_ids] = idle_target
    idle.set_gripper(env.data, 0.002)


def block_geometry_ok(tray, block):
    return bool(
        abs(block[0] - tray[0]) < 0.11
        and abs(block[1] - tray[1]) < 0.07
        and block[2] - tray[2] < 0.08
    )


def scripted_primitive(env, task, arm, tray_goal, labels, recorder=None):
    """Run a closed-loop scripted primitive without resetting ``env``."""
    if task not in {"tray_push", "pick_place"}:
        raise ValueError(f"Unsupported scripted task: {task}")
    if arm not in {"left", "right"}:
        raise ValueError(f"Unsupported arm: {arm}")

    active, idle, active_initial, idle_initial, sites, finger_qpos = arm_setup(
        env, arm
    )
    active.set_gripper(env.data, 0.002)
    hold_idle(env, idle, idle_initial)
    side = -1.0 if arm == "left" else 1.0
    observation = env.observation()
    tray_start = np.asarray(observation["tray_position"], dtype=np.float64).copy()
    block_start = np.asarray(observation["block_position"], dtype=np.float64).copy()
    contacts = {
        "left_tray": False,
        "right_tray": False,
        "left_block": False,
        "right_block": False,
    }

    if task == "tray_push":
        contact_offset = side * 0.158
        push_height = 0.025
        push_start = tray_start + np.array([contact_offset, 0.0, push_height])
        push_end = np.asarray(tray_goal) + np.array(
            [contact_offset, 0.0, push_height]
        )
        push_approach = push_start + np.array([0.0, 0.0, 0.100])
        push_retreat = push_end + np.array([side * 0.080, 0.0, 0.100])
        phase = "approach"
    else:
        grasp_midpoint = block_start + np.array([0.0, 0.0, 0.005])
        above_midpoint = grasp_midpoint + np.array([0.0, 0.0, 0.130])
        waypoint_clearance = 0.120
        lift_midpoint = grasp_midpoint + np.array(
            [0.0, 0.0, waypoint_clearance]
        )
        tray_now = np.asarray(observation["tray_position"], dtype=np.float64)
        drop_center = tray_now.copy()
        place_down = drop_center + np.array([0.0, 0.0, 0.060])
        place_above = place_down + np.array(
            [0.0, 0.0, waypoint_clearance]
        )
        block_dof = env.model.jnt_dofadr[env.block_joint]
        grasp_posture = None
        midpoint_to_block = np.array([0.0, 0.0, 0.005])
        carry_progress = 0.0
        carry_start_xy = None
        carry_duration = 1.05
        carry_z = place_above[2] + midpoint_to_block[2]
        place_above_settle_start = None
        phase = "above"

    phase_start = env.data.time
    steps = 0
    max_block_lift = 0.0
    min_block_tray_xy_after_lift = float("inf")
    timed_out = False

    while True:
        phase_elapsed = env.data.time - phase_start
        midpoint = finger_midpoint(env.data, sites)
        hold_idle(env, idle, idle_initial)

        if task == "tray_push":
            desired = {
                "approach": push_approach,
                "contact": push_start,
                "push": push_end,
                "retreat": push_retreat,
            }[phase]
            target = midpoint_target(env.data, active, midpoint, desired)
            active.move_to_position(
                env.data,
                target,
                gain=0.75 if phase == "push" else 0.35,
                max_joint_step=0.075 if phase == "push" else 0.040,
                posture_target=active_initial,
                posture_gain=0.16,
            )
            active.set_gripper(env.data, 0.002)
        else:
            observation = env.observation()
            tray_now = np.asarray(observation["tray_position"], dtype=np.float64)
            block_now = np.asarray(observation["block_position"], dtype=np.float64)
            block_z_to_tray_now = float(block_now[2] - tray_now[2])
            if phase in {"close", "lift", "carry_horizontal", "place_above"}:
                midpoint_to_block = midpoint - block_now
            if phase in {
                "carry_horizontal",
                "place_above",
                "place_down",
                "release",
                "retreat",
            }:
                place_down = drop_center + np.array([0.0, 0.0, 0.060])
                place_above = place_down + np.array(
                    [0.0, 0.0, waypoint_clearance]
                )

            if phase in {"above", "open_above"}:
                desired = above_midpoint
            elif phase in {"descend", "close"}:
                desired = grasp_midpoint
            elif phase == "lift":
                desired = lift_midpoint
            elif phase == "carry_horizontal":
                carry_goal = place_above + midpoint_to_block
                if carry_start_xy is None:
                    carry_start_xy = midpoint[:2].copy()
                z_ready = (
                    midpoint[2] >= carry_z - 0.020
                    and block_z_to_tray_now > 0.105
                )
                z_soft_ready = (
                    phase_elapsed > 0.25
                    and midpoint[2] >= carry_z - 0.040
                    and block_z_to_tray_now > 0.095
                )
                z_warmup_ready = (
                    phase_elapsed > 0.05
                    and midpoint[2] >= carry_z - 0.065
                    and block_z_to_tray_now > 0.085
                )
                if z_ready:
                    carry_scale = 1.0
                elif z_soft_ready:
                    carry_scale = 0.70
                elif z_warmup_ready:
                    carry_scale = 0.35
                else:
                    carry_scale = 0.0
                if carry_scale > 0.0:
                    carry_progress = min(
                        1.0,
                        carry_progress
                        + carry_scale * env.model.opt.timestep / carry_duration,
                    )
                carry_xy = carry_start_xy + carry_progress * (
                    carry_goal[:2] - carry_start_xy
                )
                desired = np.array([carry_xy[0], carry_xy[1], carry_z])
            elif phase in {"place_above", "retreat"}:
                desired = place_above + midpoint_to_block
            else:
                desired = place_down + midpoint_to_block

            target = midpoint_target(env.data, active, midpoint, desired)
            if phase == "retreat":
                gain, max_step, posture_target, posture_gain = (
                    0.40,
                    0.045,
                    None,
                    0.0,
                )
            elif phase == "place_down":
                gain, max_step, posture_target, posture_gain = (
                    0.20,
                    0.020,
                    grasp_posture,
                    0.16,
                )
            elif phase == "lift":
                gain, max_step, posture_target, posture_gain = (
                    0.55,
                    0.065,
                    grasp_posture,
                    0.16,
                )
            elif phase == "place_above":
                gain, max_step, posture_target, posture_gain = (
                    0.50,
                    0.060,
                    grasp_posture,
                    0.04,
                )
            elif phase == "carry_horizontal":
                gain, max_step, posture_target, posture_gain = (
                    0.56,
                    0.060,
                    None,
                    0.0,
                )
            else:
                gain, max_step, posture_target, posture_gain = (
                    0.35,
                    0.040,
                    grasp_posture,
                    0.16,
                )
            active.move_to_position(
                env.data,
                target,
                gain=gain,
                max_joint_step=max_step,
                posture_target=posture_target,
                posture_gain=posture_gain,
            )
            if phase in {"open_above", "descend", "release", "retreat"}:
                active.set_gripper(env.data, 0.037)
            else:
                active.set_gripper(env.data, 0.002)

        if recorder is not None:
            recorder.record_step()
        env.step()
        steps += 1
        update_contacts(env.data, labels, contacts)

        observation = env.observation()
        tray_now = np.asarray(observation["tray_position"], dtype=np.float64)
        block_now = np.asarray(observation["block_position"], dtype=np.float64)
        midpoint = finger_midpoint(env.data, sites)
        position_error = float(np.linalg.norm(midpoint - desired))
        xy_error = float(np.linalg.norm(midpoint[:2] - desired[:2]))
        z_error = float(abs(midpoint[2] - desired[2]))
        tray_error = float(np.linalg.norm(tray_now[:2] - tray_goal[:2]))

        if task == "pick_place":
            block_xy = float(np.linalg.norm(midpoint[:2] - block_now[:2]))
            block_z_offset = float(midpoint[2] - block_now[2])
            block_lift = float(block_now[2] - block_start[2])
            max_block_lift = max(max_block_lift, block_lift)
            block_to_tray = float(np.linalg.norm(block_now[:2] - tray_now[:2]))
            if block_lift >= 0.045:
                min_block_tray_xy_after_lift = min(
                    min_block_tray_xy_after_lift, block_to_tray
                )
            block_speed = float(
                np.linalg.norm(env.data.qvel[block_dof : block_dof + 3])
            )
            actual_finger = float(env.data.qpos[finger_qpos])

        next_phase = None
        if task == "tray_push":
            if phase == "approach" and xy_error < 0.045 and z_error < 0.060:
                next_phase = "contact"
            elif phase == "contact" and position_error < 0.045:
                next_phase = "push"
            elif phase == "push" and tray_error < 0.055:
                next_phase = "retreat"
            elif phase == "retreat" and position_error < 0.065:
                next_phase = "done"
        else:
            if phase == "above" and block_xy < 0.035 and block_z_offset > 0.070:
                grasp_posture = env.data.qpos[active.qpos_addresses].copy()
                next_phase = "open_above"
            elif (
                phase == "open_above"
                and actual_finger > 0.033
                and phase_elapsed > 0.5
            ):
                next_phase = "descend"
            elif (
                phase == "descend"
                and block_xy < 0.025
                and abs(block_z_offset - 0.005) < 0.020
            ):
                next_phase = "close"
            elif phase == "close" and phase_elapsed > 0.45:
                midpoint_to_block = midpoint - block_now
                place_above_settle_start = None
                next_phase = "lift"
            elif phase == "lift" and (
                block_lift >= waypoint_clearance * 0.80
                or (
                    phase_elapsed > 1.3
                    and block_lift >= waypoint_clearance * 0.45
                )
                or (phase_elapsed > 1.7 and block_lift >= 0.045)
            ):
                drop_center = tray_now.copy()
                place_down = drop_center + np.array([0.0, 0.0, 0.060])
                place_above = place_down + np.array(
                    [0.0, 0.0, waypoint_clearance]
                )
                midpoint_to_block = midpoint - block_now
                carry_start_xy = midpoint[:2].copy()
                carry_goal = place_above + midpoint_to_block
                carry_z = max(
                    midpoint[2], carry_goal[2], tray_now[2] + 0.185
                )
                carry_progress = 0.0
                place_above_settle_start = None
                next_phase = "carry_horizontal"
            elif phase == "carry_horizontal":
                carry_done = (
                    carry_progress >= 1.0
                    and block_to_tray < 0.045
                    and block_now[2] - tray_now[2] > 0.105
                    and (position_error < 0.090 or phase_elapsed > 1.6)
                )
                carry_fallback = (
                    phase_elapsed > 2.5
                    and block_to_tray < 0.065
                    and block_now[2] - tray_now[2] > 0.095
                )
                if carry_done or carry_fallback:
                    midpoint_to_block = midpoint - block_now
                    place_above_settle_start = None
                    next_phase = "place_above"
            elif phase == "place_above":
                place_above_ready = (
                    block_to_tray < 0.035
                    and position_error < 0.080
                    and block_now[2] - tray_now[2] > 0.105
                )
                place_above_fallback = (
                    phase_elapsed > 2.5
                    and block_to_tray < 0.055
                    and block_now[2] - tray_now[2] > 0.095
                )
                if place_above_ready or place_above_fallback:
                    if place_above_settle_start is None:
                        place_above_settle_start = env.data.time
                    elif env.data.time - place_above_settle_start >= 0.35:
                        drop_center = tray_now.copy()
                        midpoint_to_block = midpoint - block_now
                        next_phase = "place_down"
                else:
                    place_above_settle_start = None
            elif (
                phase == "place_down"
                and block_to_tray < 0.080
                and (
                    block_now[2] - tray_now[2] < 0.090
                    or phase_elapsed > 2.5
                )
                and (position_error < 0.080 or phase_elapsed > 2.5)
            ):
                next_phase = "release"
            elif (
                phase == "release"
                and phase_elapsed > 2.0
                and actual_finger > 0.033
                and block_speed < 0.050
            ):
                next_phase = "retreat"
            elif phase == "retreat" and midpoint[2] - block_now[2] > 0.040:
                next_phase = "done"

        if next_phase == "done":
            break
        if next_phase is not None:
            phase = next_phase
            phase_start = env.data.time
            phase_elapsed = 0.0

        timeout = 55.0 if phase not in {"close", "release"} else 20.0
        if phase_elapsed > timeout:
            timed_out = True
            break

    final = task_success(env, tray_goal)
    final_observation = env.observation()
    final_tray = np.asarray(final_observation["tray_position"])
    final_block = np.asarray(final_observation["block_position"])
    placement_ok = block_geometry_ok(final_tray, final_block)
    return {
        "success": bool(final["tray_ok"] if task == "tray_push" else placement_ok),
        "tray_ok": bool(final["tray_ok"]),
        "block_placement_ok": placement_ok,
        "placer_block_contact": contacts[f"{arm}_block"] if task == "pick_place" else None,
        "grasp_lift_ok": bool(max_block_lift >= 0.045) if task == "pick_place" else None,
        "transport_ok": (
            bool(min_block_tray_xy_after_lift < 0.080)
            if task == "pick_place"
            else None
        ),
        "tray_error": float(final["tray_error"]),
        "max_block_lift": max_block_lift,
        "actions": steps,
        "timed_out": timed_out,
    }


def act_segment(
    env,
    policy,
    stats,
    device,
    instruction,
    task,
    arm,
    tray_goal,
    image_stride,
    execute_actions,
    max_actions,
    labels,
):
    """Run one ACT primitive from the current simulator state."""
    renderer = mujoco.Renderer(env.model, height=224, width=224)
    contacts = {
        "left_tray": False,
        "right_tray": False,
        "left_block": False,
        "right_block": False,
    }
    initial_block = np.asarray(
        env.observation()["block_position"], dtype=np.float64
    ).copy()
    block_dof = env.model.jnt_dofadr[env.block_joint]
    max_block_lift = 0.0
    min_block_tray_xy_after_lift = float("inf")
    stable_steps = 0
    actions_executed = 0
    try:
        while actions_executed < max_actions:
            images = render_observation(renderer, env)
            state = actuator_qpos(env.model, env.data)
            normalized_state = (state - stats["state_mean"]) / stats["state_std"]
            image_tensor = torch.from_numpy(images).unsqueeze(0).to(device)
            state_tensor = torch.from_numpy(normalized_state).unsqueeze(0).to(device)
            with torch.inference_mode():
                normalized_chunk = policy(
                    state_tensor, image_tensor, [instruction]
                )[0].cpu().numpy()
            action_chunk = (
                normalized_chunk * stats["action_std"] + stats["action_mean"]
            )

            action_count = min(execute_actions, len(action_chunk))
            for action in action_chunk[:action_count]:
                env.data.ctrl[:] = clip_action(env.model, action)
                for _ in range(image_stride):
                    mujoco.mj_step(env.model, env.data)
                    update_contacts(env.data, labels, contacts)
                actions_executed += 1

                observation = env.observation()
                tray = np.asarray(observation["tray_position"])
                block = np.asarray(observation["block_position"])
                tray_error = float(np.linalg.norm(tray[:2] - tray_goal[:2]))

                if task == "tray_push":
                    stable_steps = stable_steps + 1 if tray_error < 0.070 else 0
                else:
                    lift = float(block[2] - initial_block[2])
                    max_block_lift = max(max_block_lift, lift)
                    if lift >= 0.045:
                        block_tray_xy = float(np.linalg.norm(block[:2] - tray[:2]))
                        min_block_tray_xy_after_lift = min(
                            min_block_tray_xy_after_lift, block_tray_xy
                        )
                    current_state = actuator_qpos(env.model, env.data)
                    gripper_open = current_state[6 if arm == "left" else 13] > 0.033
                    block_speed = float(
                        np.linalg.norm(env.data.qvel[block_dof : block_dof + 3])
                    )
                    placed = (
                        block_geometry_ok(tray, block)
                        and gripper_open
                        and block_speed < 0.050
                    )
                    stable_steps = stable_steps + 1 if placed else 0

                if stable_steps >= 10 or actions_executed >= max_actions:
                    break
            if stable_steps >= 10:
                break
    finally:
        renderer.close()

    final = task_success(env, tray_goal)
    placement_ok = bool(stable_steps >= 10) if task == "pick_place" else False
    return {
        "success": bool(stable_steps >= 10),
        "tray_ok": bool(final["tray_ok"]),
        "block_placement_ok": placement_ok,
        "placer_block_contact": contacts[f"{arm}_block"] if task == "pick_place" else None,
        "grasp_lift_ok": bool(max_block_lift >= 0.045) if task == "pick_place" else None,
        "transport_ok": (
            bool(min_block_tray_xy_after_lift < 0.080)
            if task == "pick_place"
            else None
        ),
        "tray_error": float(final["tray_error"]),
        "max_block_lift": max_block_lift,
        "actions": actions_executed,
        "timed_out": bool(stable_steps < 10),
    }


def evaluate_episode(
    mode,
    policy,
    stats,
    device,
    episode_path,
    execute_actions,
    push_max_actions,
    pnp_max_actions,
):
    with np.load(episode_path, allow_pickle=False) as episode:
        env = AlohaTaskEnvironment(seed=0)
        tray_goal = restore_episode(env, episode)
        image_stride = int(episode["image_stride"])

    labels = geom_labels(env.model)
    if mode in {"expert-act", "expert-expert"}:
        phase1 = scripted_primitive(
            env, "tray_push", "right", tray_goal, labels
        )
        phase1_controller = "expert"
    else:
        phase1 = act_segment(
            env,
            policy,
            stats,
            device,
            TASK_INSTRUCTIONS["right_tray_push"],
            "tray_push",
            "right",
            tray_goal,
            image_stride,
            execute_actions,
            push_max_actions,
            labels,
        )
        phase1_controller = "act"

    boundary_observation = env.observation()
    boundary_tray_error = float(
        np.linalg.norm(
            boundary_observation["tray_position"][:2] - tray_goal[:2]
        )
    )
    phase2_started = bool(phase1["success"])
    phase2 = None
    if phase2_started and mode in {"expert-act"}:
        phase2 = act_segment(
            env,
            policy,
            stats,
            device,
            TASK_INSTRUCTIONS["left_pick_place"],
            "pick_place",
            "left",
            tray_goal,
            image_stride,
            execute_actions,
            pnp_max_actions,
            labels,
        )
        phase2_controller = "act"
    elif phase2_started:
        phase2 = scripted_primitive(
            env, "pick_place", "left", tray_goal, labels
        )
        phase2_controller = "expert"
    else:
        phase2_controller = "act" if mode == "expert-act" else "expert"

    final = task_success(env, tray_goal)
    phase2_success = None if phase2 is None else bool(phase2["success"])
    full_success = bool(
        phase1["success"] and phase2_success and final["tray_ok"]
    )
    return {
        "episode": str(episode_path),
        "mode": mode,
        "phase1_controller": phase1_controller,
        "phase2_controller": phase2_controller,
        "phase1_success": bool(phase1["success"]),
        "phase1_tray_error": boundary_tray_error,
        "phase1_controller_steps": int(phase1["actions"]),
        "phase1_timed_out": bool(phase1["timed_out"]),
        "phase2_started": phase2_started,
        "phase2_success": phase2_success,
        "placer_block_contact": (
            None if phase2 is None else phase2["placer_block_contact"]
        ),
        "grasp_lift_ok": None if phase2 is None else phase2["grasp_lift_ok"],
        "transport_ok": None if phase2 is None else phase2["transport_ok"],
        "phase2_controller_steps": 0 if phase2 is None else int(phase2["actions"]),
        "phase2_timed_out": None if phase2 is None else bool(phase2["timed_out"]),
        "final_tray_ok": bool(final["tray_ok"]),
        "final_block_ok": bool(final["block_ok"]),
        "full_success": full_success,
    }


def summarize(rows):
    summary = {"episodes": len(rows)}
    for field in ("phase1_success", "phase2_started", "full_success"):
        values = [bool(row[field]) for row in rows]
        successes = sum(values)
        summary[field] = {
            "count": len(values),
            "successes": successes,
            "rate": successes / len(values) if values else None,
            "wilson_95": wilson_interval(successes, len(values)),
        }
    for field in (
        "phase2_success",
        "placer_block_contact",
        "grasp_lift_ok",
        "transport_ok",
    ):
        values = [bool(row[field]) for row in rows if row[field] is not None]
        successes = sum(values)
        summary[field] = {
            "count": len(values),
            "successes": successes,
            "rate": successes / len(values) if values else None,
            "wilson_95": wilson_interval(successes, len(values)),
        }
    for field in ("phase1_tray_error",):
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        summary[field] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate expert/ACT controller handoffs on unseen RL"
    )
    parser.add_argument(
        "--mode",
        choices=("expert-act", "act-expert", "expert-expert"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute-actions", type=int, default=10)
    parser.add_argument("--push-max-actions", type=int, default=70)
    parser.add_argument("--pnp-max-actions", type=int, default=140)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint else None
    policy = None
    stats = None
    if args.mode != "expert-expert":
        if checkpoint_path is None:
            parser.error("--checkpoint is required when an ACT segment is used")
        stats_path = (
            args.stats.resolve()
            if args.stats
            else checkpoint_path.parent / "normalization_stats.json"
        )
        stats = load_stats(stats_path)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        policy_config = dict(checkpoint["model_config"])
        policy_config["device"] = str(device)
        policy = ACTPolicy(policy_config).to(device)
        policy.load_state_dict(checkpoint["model"], strict=True)
        policy.eval()

        chunk_size = int(policy_config["num_queries"])
        if not 1 <= args.execute_actions <= chunk_size:
            parser.error(
                f"--execute-actions must be between 1 and {chunk_size}"
            )
    paths = sorted(args.episode_dir.glob("episode_*.npz"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        parser.error(f"No episode_*.npz files in {args.episode_dir}")

    rows = []
    for index, path in enumerate(paths, 1):
        row = evaluate_episode(
            args.mode,
            policy,
            stats,
            device,
            path,
            args.execute_actions,
            args.push_max_actions,
            args.pnp_max_actions,
        )
        rows.append(row)
        print(
            f"[{index:03d}/{len(paths):03d}] mode={args.mode} "
            f"phase1={row['phase1_success']} "
            f"phase2={row['phase2_success']} full={row['full_success']}",
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
        mode=args.mode,
        checkpoint=None if checkpoint_path is None else str(checkpoint_path),
        episode_dir=str(args.episode_dir.resolve()),
        execute_actions=args.execute_actions,
        push_max_actions=args.push_max_actions,
        pnp_max_actions=args.pnp_max_actions,
    )
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {csv_path}\nsaved: {summary_path}")


if __name__ == "__main__":
    main()
