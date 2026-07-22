import time
import sys
from contextlib import nullcontext
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aloha.controller import AlohaArmController
from aloha.task_env import AlohaTaskEnvironment
from aloha.demonstration_io import DemonstrationRecorder


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


def _set_scene(env, tray_position, block_position, tray_goal):
    env._set_freejoint_pose(env.tray_joint, tray_position)
    env._set_freejoint_pose(env.block_joint, block_position)
    target = site_id(env.model, "target")
    env.model.site_pos[target] = np.array([tray_goal[0], tray_goal[1], 0.012])
    env.data.qvel[:] = 0.0
    env.data.qacc[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _arm_setup(env, arm):
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


def _hold_idle(env, idle, idle_initial):
    env.data.ctrl[idle.actuator_ids] = idle_initial
    idle.set_gripper(env.data, 0.002)


def run_primitive_episode(
    task,
    arm,
    record_path=None,
    show_viewer=True,
    seed=0,
    randomize=False,
):
    if task not in {"tray_push", "pick_place"}:
        raise ValueError(f"Unsupported primitive task: {task}")
    if arm not in {"left", "right"}:
        raise ValueError(f"Unsupported arm: {arm}")

    env = AlohaTaskEnvironment(seed=seed)
    env.reset(randomize=False)
    rng = np.random.default_rng(seed)
    side = -1.0 if arm == "left" else 1.0

    if task == "tray_push":
        tray_y = rng.uniform(0.135, 0.165) if randomize else 0.15
        start_abs = rng.uniform(0.025, 0.055) if randomize else 0.04
        goal_abs = rng.uniform(0.080, 0.115) if randomize else 0.10
        tray_start = np.array([side * start_abs, tray_y, 0.018])
        tray_goal = np.array([-side * goal_abs, tray_y, 0.018])
        block_start = np.array([0.0, -0.12, 0.025])
        instruction = f"Use the {arm} arm to push the tray to the green target."
    else:
        tray_y = rng.uniform(0.135, 0.165) if randomize else 0.15
        tray_x = side * (rng.uniform(0.080, 0.110) if randomize else 0.10)
        block_x = side * (rng.uniform(0.000, 0.030) if randomize else 0.015)
        block_y = rng.uniform(-0.140, -0.105) if randomize else -0.12
        tray_goal = np.array([tray_x, tray_y, 0.018])
        tray_start = tray_goal.copy()
        block_start = np.array([block_x, block_y, 0.025])
        instruction = f"Use the {arm} arm to pick up the red block and place it in the tray."

    _set_scene(env, tray_start, block_start, tray_goal)
    active, idle, active_initial, idle_initial, sites, finger_qpos = _arm_setup(
        env, arm
    )
    active.set_gripper(env.data, 0.002)
    _hold_idle(env, idle, idle_initial)

    recorder = None
    if record_path is not None:
        recorder = DemonstrationRecorder(
            env.model, env.data, record_path, instruction, tray_goal
        )

    viewer_context = (
        mujoco.viewer.launch_passive(env.model, env.data)
        if show_viewer
        else nullcontext(None)
    )
    done_wait_seconds = 2.0 if show_viewer else 0.0
    result_success = False

    if task == "tray_push":
        contact_offset = side * 0.158
        push_height = 0.025
        push_start = tray_start + np.array([contact_offset, 0.0, push_height])
        push_end = tray_goal + np.array([contact_offset, 0.0, push_height])
        push_approach = push_start + np.array([0.0, 0.0, 0.100])
        push_retreat = push_end + np.array([side * 0.080, 0.0, 0.100])
        phase = "approach"
    else:
        grasp_midpoint = block_start + np.array([0.0, 0.0, 0.005])
        above_midpoint = grasp_midpoint + np.array([0.0, 0.0, 0.130])
        lift_midpoint = grasp_midpoint + np.array([0.0, 0.0, 0.160])
        place_above = tray_goal + np.array([0.0, 0.0, 0.170])
        # Release from above the tray floor.  The long ALOHA fingers can hook
        # the tray rim/floor if their midpoint descends as low as the full-task
        # controller's original placement waypoint.
        place_down = tray_goal + np.array([0.0, 0.0, 0.105])
        block_dof = env.model.jnt_dofadr[env.block_joint]
        block_initial_z = float(block_start[2])
        grasp_posture = None
        phase = "above"

    print(f"primitive task={task}, arm={arm}")
    print("tray start:", tray_start)
    print("tray goal:", tray_goal)
    print("block start:", block_start)

    phase_start = env.data.time
    done_wall_start = None
    last_print = 0.0

    with viewer_context as viewer:
        while viewer is None or viewer.is_running():
            loop_start = time.time()
            phase_elapsed = env.data.time - phase_start
            midpoint = finger_midpoint(env.data, sites)
            _hold_idle(env, idle, idle_initial)

            if task == "tray_push":
                desired = {
                    "approach": push_approach,
                    "contact": push_start,
                    "push": push_end,
                    "retreat": push_retreat,
                    "done": push_retreat,
                }[phase]
                target = midpoint_target(env.data, active, midpoint, desired)
                gain = 0.75 if phase == "push" else 0.35
                max_step = 0.075 if phase == "push" else 0.040
                active.move_to_position(
                    env.data,
                    target,
                    gain=gain,
                    max_joint_step=max_step,
                    posture_target=active_initial,
                    posture_gain=0.16,
                )
                active.set_gripper(env.data, 0.002)
            else:
                if phase in {"above", "open_above"}:
                    desired = above_midpoint
                elif phase in {"descend", "close"}:
                    desired = grasp_midpoint
                elif phase == "lift":
                    desired = lift_midpoint
                elif phase in {"place_above", "retreat", "done"}:
                    desired = place_above
                else:
                    desired = place_down

                target = midpoint_target(env.data, active, midpoint, desired)
                active.move_to_position(
                    env.data,
                    target,
                    gain=0.40 if phase in {"retreat", "done"} else 0.35,
                    max_joint_step=0.045 if phase in {"retreat", "done"} else 0.040,
                    posture_target=None if phase in {"retreat", "done"} else grasp_posture,
                    posture_gain=0.16,
                )
                if phase in {"open_above", "descend", "release", "retreat", "done"}:
                    active.set_gripper(env.data, 0.037)
                else:
                    active.set_gripper(env.data, 0.002)

            if recorder is not None:
                recorder.record_step()
            env.step()
            if viewer is not None:
                viewer.sync()

            obs = env.observation()
            tray_now = obs["tray_position"]
            block_now = obs["block_position"]
            midpoint = finger_midpoint(env.data, sites)
            position_error = float(np.linalg.norm(midpoint - desired))
            xy_error = float(np.linalg.norm(midpoint[:2] - desired[:2]))
            z_error = float(abs(midpoint[2] - desired[2]))
            tray_error = float(np.linalg.norm(tray_now[:2] - tray_goal[:2]))
            actual_finger = float(env.data.qpos[finger_qpos])

            if task == "pick_place":
                block_xy = float(np.linalg.norm(midpoint[:2] - block_now[:2]))
                block_z_offset = float(midpoint[2] - block_now[2])
                block_lift = float(block_now[2] - block_initial_z)
                block_to_tray = float(np.linalg.norm(block_now[:2] - tray_now[:2]))
                block_speed = float(
                    np.linalg.norm(env.data.qvel[block_dof : block_dof + 3])
                )

            if loop_start - last_print >= 0.5:
                message = (
                    f"phase={phase} arm={arm} pos={position_error:.3f} "
                    f"tray_goal={tray_error:.3f} grip={actual_finger:.3f}"
                )
                if task == "pick_place":
                    message += (
                        f" block_xy={block_xy:.3f} z={block_z_offset:.3f} "
                        f"lift={block_lift:.3f} tray_xy={block_to_tray:.3f}"
                    )
                print(message)
                last_print = loop_start

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
                elif phase == "open_above" and actual_finger > 0.033 and phase_elapsed > 0.5:
                    next_phase = "descend"
                elif phase == "descend" and block_xy < 0.025 and abs(block_z_offset - 0.005) < 0.020:
                    next_phase = "close"
                elif phase == "close" and phase_elapsed > 0.8:
                    next_phase = "lift"
                elif phase == "lift" and block_lift >= 0.080:
                    # Match the full-task expert: plan placement from the
                    # tray's measured pose, not only its nominal goal pose.
                    place_above = tray_now + np.array([0.0, 0.0, 0.170])
                    place_down = tray_now + np.array([0.0, 0.0, 0.105])
                    next_phase = "place_above"
                elif (
                    phase == "place_above"
                    and block_to_tray < 0.035
                    and position_error < 0.060
                ):
                    next_phase = "place_down"
                elif phase == "place_down" and block_to_tray < 0.070 and position_error < 0.055:
                    next_phase = "release"
                elif phase == "release" and phase_elapsed > 1.0 and actual_finger > 0.033 and block_speed < 0.050:
                    next_phase = "retreat"
                elif phase == "retreat" and midpoint[2] - block_now[2] > 0.040:
                    next_phase = "done"

            if next_phase is not None:
                print(f"phase transition: {phase} -> {next_phase}")
                phase = next_phase
                phase_start = env.data.time
                if phase == "done":
                    done_wall_start = time.time()

            if phase == "done" and done_wall_start is not None and time.time() - done_wall_start >= done_wait_seconds:
                tray_ok = tray_error < 0.070
                if task == "tray_push":
                    result_success = tray_ok
                    print(f"finished: tray_ok={tray_ok}, success={result_success}")
                else:
                    block_ok = (
                        abs(block_now[0] - tray_now[0]) < 0.11
                        and abs(block_now[1] - tray_now[1]) < 0.07
                        and block_now[2] - tray_now[2] < 0.08
                    )
                    # A placement that shoves the tray out of its target is
                    # not a successful primitive demonstration.
                    result_success = bool(tray_ok and block_ok)
                    print(
                        f"finished: tray_ok={tray_ok}, block_ok={block_ok}, "
                        f"success={result_success}"
                    )
                break

            timeout = 55.0 if phase not in {"close", "release", "done"} else 20.0
            if phase_elapsed > timeout and phase != "done":
                print(
                    f"phase timeout: {phase}, position_error={position_error:.3f}, "
                    f"tray_goal_error={tray_error:.3f}"
                )
                break

            if viewer is not None:
                remaining = env.model.opt.timestep - (time.time() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

    if recorder is not None:
        saved_path = recorder.save(result_success)
        print(f"saved demonstration: {saved_path}")
    return result_success
