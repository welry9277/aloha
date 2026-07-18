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
                rng.uniform(0.025, 0.055),
                rng.uniform(0.135, 0.165),
                0.018,
            ]
        )
        block_start = np.array(
            [
                rng.uniform(-0.055, -0.025),
                rng.uniform(-0.140, -0.105),
                0.025,
            ]
        )
    else:
        tray_start = np.array([0.05, 0.15, 0.018])
        block_start = np.array([-0.05, -0.12, 0.025])
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

    left = AlohaArmController(env.model, "left")
    left_sites = (
        site_id(env.model, "left/left_finger"),
        site_id(env.model, "left/right_finger"),
    )

    left_finger_joint = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "left/left_finger"
    )
    left_finger_qpos = env.model.jnt_qposadr[left_finger_joint]

    grasp_midpoint = block_start + np.array([0.0, 0.0, 0.005])
    above_midpoint = grasp_midpoint + np.array([0.0, 0.0, 0.13])
    lift_midpoint = grasp_midpoint + np.array([0.0, 0.0, 0.16])
    place_above = tray_goal + np.array([0.0, 0.0, 0.22])
    place_down = tray_goal + np.array([0.0, 0.0, 0.060])
    left_mid_to_block = np.array([0.0, 0.0, 0.005])

    print("left-only task: left block place")
    print("tray goal:", tray_goal)
    print("block start:", block_start)
    print("left above waypoint:", above_midpoint)
    print("left place-above waypoint:", place_above)

    left.set_gripper(env.data, 0.002)

    instruction = "Use the left arm to pick up the red block and place it in the tray."
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
        phase = "left_above"
        phase_start = env.data.time
        last_print = 0.0
        left_grasp_posture = None
        drop_center = tray_goal.copy()
        place_above_settle_start = None
        finished_reported = False
        done_wall_start = None

        while viewer is None or viewer.is_running():
            loop_start = time.time()
            phase_elapsed = env.data.time - phase_start

            obs = env.observation()
            tray_now = obs["tray_position"]
            block_now = obs["block_position"]
            left_mid = finger_midpoint(env.data, *left_sites)

            if phase in {"close", "lift", "place_above"}:
                left_mid_to_block = left_mid - block_now

            if phase == "place_above":
                drop_center = tray_now.copy()
            if phase in {"place_above", "place_down", "release", "retreat", "done"}:
                # Align over the tray first, then keep this XY fixed so
                # place_down is a vertical top-down drop.
                place_above = drop_center + np.array([0.0, 0.0, 0.22])
                place_down = drop_center + np.array([0.0, 0.0, 0.060])

            if phase in {"left_above", "open_above"}:
                left_desired = above_midpoint
            elif phase in {"descend", "close"}:
                left_desired = grasp_midpoint
            elif phase == "lift":
                left_desired = lift_midpoint
            elif phase in {"place_above", "retreat", "done"}:
                left_desired = place_above + left_mid_to_block
            else:
                left_desired = place_down + left_mid_to_block

            left_target = midpoint_target(
                env.data, left, left_mid, left_desired
            )
            if phase in {"retreat", "done"}:
                # After release, prioritize a clean vertical escape. The
                # grasp posture regularizer can oppose this large motion.
                left_error = left.move_to_position(
                    env.data,
                    left_target,
                    gain=0.40,
                    max_joint_step=0.045,
                    posture_target=None,
                )
            elif phase == "place_down":
                left_error = left.move_to_position(
                    env.data,
                    left_target,
                    gain=0.20,
                    max_joint_step=0.020,
                    posture_target=left_grasp_posture,
                    posture_gain=0.16,
                )
            else:
                left_error = left.move_to_position(
                    env.data,
                    left_target,
                    posture_target=left_grasp_posture,
                    posture_gain=0.16,
                )
            if phase in {"open_above", "descend", "release", "retreat", "done"}:
                left.set_gripper(env.data, 0.037)
            else:
                left.set_gripper(env.data, 0.002)

            if recorder is not None:
                recorder.record_step()
            env.step()
            if viewer is not None:
                viewer.sync()

            obs = env.observation()
            tray_now = obs["tray_position"]
            block_now = obs["block_position"]
            left_mid = finger_midpoint(env.data, *left_sites)
            left_error = float(np.linalg.norm(left_mid - left_desired))
            block_xy_error = float(np.linalg.norm(left_mid[:2] - block_now[:2]))
            block_z_offset = float(left_mid[2] - block_now[2])
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
            actual_left_finger = float(env.data.qpos[left_finger_qpos])

            if loop_start - last_print >= 0.5:
                print(
                    f"phase={phase} | "
                    f"L err={left_error:.3f} grip={actual_left_finger:.3f} | "
                    f"block xy={block_xy_error:.3f} z={block_z_offset:.3f} | "
                    f"lift={block_lift:.3f} speed={block_speed:.3f} | "
                    f"tray_xy={block_to_tray:.3f} z={block_z_to_tray:.3f}"
                )
                last_print = loop_start

            next_phase = None
            if (
                phase == "left_above"
                and block_xy_error < 0.035
                and block_z_offset > 0.070
            ):
                left_grasp_posture = env.data.qpos[
                    left.qpos_addresses
                ].copy()
                next_phase = "open_above"
            elif (
                phase == "open_above"
                and actual_left_finger > 0.033
                and phase_elapsed > 0.5
            ):
                next_phase = "descend"
            elif (
                phase == "descend"
                and block_xy_error < 0.025
                and abs(block_z_offset - 0.005) < 0.020
            ):
                next_phase = "close"
            elif phase == "close" and phase_elapsed > 1.5:
                next_phase = "lift"
            elif phase == "lift" and block_lift >= 0.080:
                left_mid_to_block = left_mid - block_now
                place_above_settle_start = None
                next_phase = "place_above"
            elif phase == "place_above":
                place_above_ready = (
                    block_to_tray < 0.045
                    and left_error < 0.080
                    and block_z_to_tray > 0.150
                )
                place_above_fallback = (
                    phase_elapsed > 2.0
                    and block_to_tray < 0.070
                    and block_z_to_tray > 0.140
                )
                if place_above_ready or place_above_fallback:
                    if place_above_settle_start is None:
                        place_above_settle_start = env.data.time
                    elif env.data.time - place_above_settle_start >= 1.0:
                        drop_center = tray_now.copy()
                        left_mid_to_block = left_mid - block_now
                        next_phase = "place_down"
                else:
                    place_above_settle_start = None
            elif (
                phase == "place_down"
                and block_to_tray < 0.080
                and (block_z_to_tray < 0.090 or phase_elapsed > 2.5)
                and (left_error < 0.080 or phase_elapsed > 2.5)
            ):
                next_phase = "release"
            elif (
                phase == "release"
                and phase_elapsed > 2.0
                and actual_left_finger > 0.033
                and block_speed < 0.050
            ):
                next_phase = "retreat"
            elif phase == "retreat" and left_mid[2] - block_now[2] > 0.040:
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
                    f"phase timeout: {phase}, left_error={left_error:.3f}, "
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
