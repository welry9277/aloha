import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from aloha_controller import AlohaArmController
from aloha_task_env import AlohaTaskEnvironment
from demonstration_io import DemonstrationRecorder


def site_id(model, name):
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if value == -1:
        raise ValueError(f"Site not found: {name}")
    return value


def finger_midpoint(data, left_site, right_site):
    return 0.5 * (data.site_xpos[left_site] + data.site_xpos[right_site])


def midpoint_target(data, controller, current_midpoint, desired_midpoint):
    gripper_position, _ = controller.pose(data)
    return gripper_position + (desired_midpoint - current_midpoint)


def run_episode(record_path=None, show_viewer=True, seed=0, randomize=False):
    env = AlohaTaskEnvironment(seed=seed)
    env.reset(randomize=False)

    rng = np.random.default_rng(seed)
    if randomize:
        tray_start = np.array(
            [
                rng.uniform(0.085, 0.125),
                rng.uniform(0.135, 0.165),
                0.018,
            ]
        )
        block_start = np.array(
            [
                rng.uniform(0.025, 0.055),
                rng.uniform(-0.140, -0.105),
                0.025,
            ]
        )
    else:
        tray_start = np.array([0.105, 0.15, 0.018])
        block_start = np.array([0.05, -0.12, 0.025])
    tray_goal = tray_start.copy()

    env._set_freejoint_pose(env.tray_joint, tray_start)
    env._set_freejoint_pose(env.block_joint, block_start)
    target_site = site_id(env.model, "target")
    env.model.site_pos[target_site] = np.array(
        [tray_goal[0], tray_goal[1], 0.012]
    )
    env.data.qvel[:] = 0.0
    env.data.qacc[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    observation = env.observation()
    block_start = observation["block_position"].copy()
    block_dof_address = env.model.jnt_dofadr[env.block_joint]

    right = AlohaArmController(env.model, "right")
    right_sites = (
        site_id(env.model, "right/left_finger"),
        site_id(env.model, "right/right_finger"),
    )

    right_finger_joint = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "right/left_finger"
    )
    right_finger_qpos = env.model.jnt_qposadr[right_finger_joint]

    grasp_midpoint = block_start + np.array([0.0, 0.0, 0.005])
    above_midpoint = grasp_midpoint + np.array([0.0, 0.0, 0.13])
    waypoint_clearance = 0.12
    lift_midpoint = grasp_midpoint + np.array([0.0, 0.0, waypoint_clearance])
    close_hold_seconds = 0.45
    carry_duration = 1.05
    place_above_settle_seconds = 0.35
    place_down = tray_goal + np.array([0.0, 0.0, 0.060])
    place_above = place_down + np.array([0.0, 0.0, waypoint_clearance])
    right_mid_to_block = np.array([0.0, 0.0, 0.005])

    print("right-only task: right block place")
    print("tray goal:", tray_goal)
    print("block start:", block_start)
    print("right above waypoint:", above_midpoint)
    print("right place-above waypoint:", place_above)

    right.set_gripper(env.data, 0.002)

    instruction = "Use the right arm to pick up the red block and place it in the tray."
    recorder = None
    if record_path is not None:
        recorder = DemonstrationRecorder(
            env.model,
            env.data,
            record_path,
            instruction,
            tray_goal,
        )

    viewer_context = (
        mujoco.viewer.launch_passive(env.model, env.data)
        if show_viewer
        else nullcontext(None)
    )
    result_success = False
    done_wait_seconds = 2.0 if show_viewer else 0.0

    with viewer_context as viewer:
        phase = "right_above"
        phase_start = env.data.time
        last_print = 0.0
        right_grasp_posture = None
        drop_center = tray_goal.copy()
        place_above_settle_start = None
        carry_progress = 0.0
        carry_start_xy = None
        carry_z = place_above[2] + right_mid_to_block[2]
        finished_reported = False
        done_wall_start = None

        while viewer is None or viewer.is_running():
            loop_start = time.time()
            phase_elapsed = env.data.time - phase_start

            obs = env.observation()
            tray_now = obs["tray_position"]
            block_now = obs["block_position"]
            block_z_to_tray_now = float(block_now[2] - tray_now[2])
            right_mid = finger_midpoint(env.data, *right_sites)

            if phase in {"close", "lift", "carry_horizontal", "place_above"}:
                right_mid_to_block = right_mid - block_now

            if phase in {
                "carry_horizontal",
                "place_above",
                "place_down",
                "release",
                "retreat",
                "done",
            }:
                # Align over the tray first, then keep this XY fixed so
                # place_down is a vertical top-down drop.
                place_down = drop_center + np.array([0.0, 0.0, 0.060])
                place_above = place_down + np.array(
                    [0.0, 0.0, waypoint_clearance]
                )

            if phase in {"right_above", "open_above"}:
                right_desired = above_midpoint
            elif phase in {"descend", "close"}:
                right_desired = grasp_midpoint
            elif phase == "lift":
                right_desired = lift_midpoint
            elif phase == "carry_horizontal":
                carry_goal = place_above + right_mid_to_block
                if carry_start_xy is None:
                    carry_start_xy = right_mid[:2].copy()
                z_ready = (
                    right_mid[2] >= carry_z - 0.020
                    and block_z_to_tray_now > 0.105
                )
                z_soft_ready = (
                    phase_elapsed > 0.25
                    and right_mid[2] >= carry_z - 0.040
                    and block_z_to_tray_now > 0.095
                )
                z_warmup_ready = (
                    phase_elapsed > 0.05
                    and right_mid[2] >= carry_z - 0.065
                    and block_z_to_tray_now > 0.085
                )
                if z_ready:
                    carry_step_scale = 1.0
                elif z_soft_ready:
                    carry_step_scale = 0.70
                elif z_warmup_ready:
                    carry_step_scale = 0.35
                else:
                    carry_step_scale = 0.0
                if carry_step_scale > 0.0:
                    carry_progress = min(
                        1.0,
                        carry_progress
                        + (
                            carry_step_scale
                            * env.model.opt.timestep
                            / carry_duration
                        ),
                    )
                carry_xy = (
                    carry_start_xy
                    + carry_progress * (carry_goal[:2] - carry_start_xy)
                )
                right_desired = np.array([carry_xy[0], carry_xy[1], carry_z])
            elif phase in {"place_above", "retreat", "done"}:
                right_desired = place_above + right_mid_to_block
            else:
                right_desired = place_down + right_mid_to_block

            right_target = midpoint_target(
                env.data, right, right_mid, right_desired
            )
            if phase in {"retreat", "done"}:
                # After release, prioritize a clean vertical escape. The
                # grasp posture regularizer can oppose this large motion.
                right_error = right.move_to_position(
                    env.data,
                    right_target,
                    gain=0.40,
                    max_joint_step=0.045,
                    posture_target=None,
                )
            elif phase == "place_down":
                right_error = right.move_to_position(
                    env.data,
                    right_target,
                    gain=0.20,
                    max_joint_step=0.020,
                    posture_target=right_grasp_posture,
                    posture_gain=0.16,
                )
            elif phase == "place_above":
                right_error = right.move_to_position(
                    env.data,
                    right_target,
                    gain=0.50,
                    max_joint_step=0.060,
                    posture_target=right_grasp_posture,
                    posture_gain=0.04,
                )
            elif phase == "carry_horizontal":
                right_error = right.move_to_position(
                    env.data,
                    right_target,
                    gain=0.56,
                    max_joint_step=0.060,
                    posture_target=None,
                )
            else:
                right_error = right.move_to_position(
                    env.data,
                    right_target,
                    posture_target=right_grasp_posture,
                    posture_gain=0.16,
                )
            if phase in {"open_above", "descend", "release", "retreat", "done"}:
                right.set_gripper(env.data, 0.037)
            else:
                right.set_gripper(env.data, 0.002)

            if recorder is not None:
                recorder.record_step()
            env.step()
            if viewer is not None:
                viewer.sync()

            obs = env.observation()
            tray_now = obs["tray_position"]
            block_now = obs["block_position"]
            right_mid = finger_midpoint(env.data, *right_sites)
            right_error = float(np.linalg.norm(right_mid - right_desired))
            block_xy_error = float(np.linalg.norm(right_mid[:2] - block_now[:2]))
            block_z_offset = float(right_mid[2] - block_now[2])
            block_lift = float(block_now[2] - block_start[2])
            block_speed = float(
                np.linalg.norm(
                    env.data.qvel[
                        block_dof_address : block_dof_address + 3
                    ]
                )
            )
            block_to_tray = float(np.linalg.norm(block_now[:2] - tray_now[:2]))
            block_z_to_tray = float(block_now[2] - tray_now[2])
            actual_right_finger = float(env.data.qpos[right_finger_qpos])

            if loop_start - last_print >= 0.5:
                print(
                    f"phase={phase} | "
                    f"R err={right_error:.3f} grip={actual_right_finger:.3f} | "
                    f"block xy={block_xy_error:.3f} z={block_z_offset:.3f} | "
                    f"lift={block_lift:.3f} speed={block_speed:.3f} | "
                    f"tray_xy={block_to_tray:.3f} z={block_z_to_tray:.3f}"
                )
                last_print = loop_start

            next_phase = None
            if (
                phase == "right_above"
                and block_xy_error < 0.035
                and block_z_offset > 0.070
            ):
                right_grasp_posture = env.data.qpos[
                    right.qpos_addresses
                ].copy()
                next_phase = "open_above"
            elif (
                phase == "open_above"
                and actual_right_finger > 0.033
                and phase_elapsed > 0.5
            ):
                next_phase = "descend"
            elif (
                phase == "descend"
                and block_xy_error < 0.025
                and abs(block_z_offset - 0.005) < 0.020
            ):
                next_phase = "close"
            elif phase == "close" and phase_elapsed > close_hold_seconds:
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
                right_mid_to_block = right_mid - block_now
                carry_start_xy = right_mid[:2].copy()
                carry_goal = place_above + right_mid_to_block
                carry_z = max(
                    right_mid[2],
                    carry_goal[2],
                    tray_now[2] + 0.185,
                )
                carry_progress = 0.0
                place_above_settle_start = None
                next_phase = "carry_horizontal"
            elif phase == "carry_horizontal":
                carry_done = (
                    carry_progress >= 1.0
                    and block_to_tray < 0.045
                    and block_z_to_tray > 0.105
                    and (right_error < 0.090 or phase_elapsed > 1.6)
                )
                carry_fallback = (
                    phase_elapsed > 2.5
                    and block_to_tray < 0.065
                    and block_z_to_tray > 0.095
                )
                if carry_done or carry_fallback:
                    right_mid_to_block = right_mid - block_now
                    place_above_settle_start = None
                    next_phase = "place_above"
            elif phase == "place_above":
                place_above_ready = (
                    block_to_tray < 0.035
                    and right_error < 0.080
                    and block_z_to_tray > 0.105
                )
                place_above_fallback = (
                    phase_elapsed > 2.5
                    and block_to_tray < 0.055
                    and block_z_to_tray > 0.095
                )
                if place_above_ready or place_above_fallback:
                    if place_above_settle_start is None:
                        place_above_settle_start = env.data.time
                    elif (
                        env.data.time - place_above_settle_start
                        >= place_above_settle_seconds
                    ):
                        drop_center = tray_now.copy()
                        right_mid_to_block = right_mid - block_now
                        next_phase = "place_down"
                else:
                    place_above_settle_start = None
            elif (
                phase == "place_down"
                and block_to_tray < 0.080
                and (block_z_to_tray < 0.090 or phase_elapsed > 2.5)
                and (right_error < 0.080 or phase_elapsed > 2.5)
            ):
                next_phase = "release"
            elif (
                phase == "release"
                and phase_elapsed > 2.0
                and actual_right_finger > 0.033
                and block_speed < 0.050
            ):
                next_phase = "retreat"
            elif phase == "retreat" and right_mid[2] - block_now[2] > 0.040:
                next_phase = "done"

            if next_phase is not None:
                print(f"phase transition: {phase} -> {next_phase}")
                phase = next_phase
                phase_start = env.data.time
                phase_elapsed = 0.0
                if phase == "done":
                    done_wall_start = time.time()

            if (
                phase == "done"
                and done_wall_start is not None
                and time.time() - done_wall_start >= done_wait_seconds
                and not finished_reported
            ):
                block_ok = (
                    abs(block_now[0] - tray_now[0]) < 0.11
                    and abs(block_now[1] - tray_now[1]) < 0.07
                    and block_now[2] - tray_now[2] < 0.08
                )
                print(f"finished: block_ok={block_ok}, success={block_ok}")
                result_success = bool(block_ok)
                finished_reported = True
                break

            timeout = 55.0 if phase not in {"close", "release", "done"} else 20.0
            if phase_elapsed > timeout and phase != "done":
                print(
                    f"phase timeout: {phase}, right_error={right_error:.3f}, "
                    f"block_xy={block_xy_error:.3f}, "
                    f"block_z={block_z_offset:.3f}, "
                    f"tray_xy={block_to_tray:.3f}"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize", action="store_true")
    args = parser.parse_args()
    run_episode(
        record_path=args.record,
        show_viewer=not args.no_viewer,
        seed=args.seed,
        randomize=args.randomize,
    )


if __name__ == "__main__":
    main()
