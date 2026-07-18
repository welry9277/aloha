import time

import mujoco
import mujoco.viewer
import numpy as np

from aloha_controller import AlohaArmController
from aloha_task_env import AlohaTaskEnvironment


def site_id(model, name):
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if value == -1:
        raise ValueError(f"Site not found: {name}")
    return value


def finger_midpoint(env, side):
    left_finger_site = site_id(env.model, f"{side}/left_finger")
    right_finger_site = site_id(env.model, f"{side}/right_finger")
    return 0.5 * (
        env.data.site_xpos[left_finger_site]
        + env.data.site_xpos[right_finger_site]
    )


def midpoint_to_gripper_target(env, controller, side, target_midpoint):
    current_midpoint = finger_midpoint(env, side)
    current_gripper_position, _ = controller.pose(env.data)
    return current_gripper_position + (target_midpoint - current_midpoint)


def main():
    env = AlohaTaskEnvironment(seed=0)
    observation = env.reset(randomize=False)
    left = AlohaArmController(env.model, "left")
    right = AlohaArmController(env.model, "right")

    tray_start = observation["tray_position"].copy()
    block_start = observation["block_position"].copy()

    right_finger_joint_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "right/left_finger",
    )
    right_finger_qpos_address = env.model.jnt_qposadr[right_finger_joint_id]

    left_finger_joint_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "left/left_finger",
    )
    left_finger_qpos_address = env.model.jnt_qposadr[left_finger_joint_id]

    grasp_midpoint = block_start + np.array([0.0, 0.0, 0.005])
    above_midpoint = grasp_midpoint + np.array([0.0, 0.0, 0.13])
    lift_midpoint = grasp_midpoint + np.array([0.0, 0.0, 0.16])
    place_above_midpoint = tray_start + np.array([0.0, 0.0, 0.17])
    place_midpoint = tray_start + np.array([0.0, 0.0, 0.075])

    left_hold_midpoint = tray_start + np.array([-0.155, 0.0, 0.052])
    left_above_midpoint = left_hold_midpoint + np.array([-0.020, 0.0, 0.10])

    print("top-down tray insert test")
    print("tray centre:", tray_start)
    print("block centre:", block_start)
    print("right above target:", above_midpoint)
    print("right grasp target:", grasp_midpoint)
    print("left tray hold target:", left_hold_midpoint)

    left.set_gripper(env.data, 0.037)
    right.set_gripper(env.data, 0.002)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        phase_name = "left_above"
        phase_start = time.time()
        last_print = 0.0
        grasp_posture = None
        left_hold_posture = None
        finished_reported = False

        while viewer.is_running():
            loop_start = time.time()
            phase_elapsed = loop_start - phase_start

            if phase_name == "left_above":
                left_target_midpoint = left_above_midpoint
                left_gripper_command = 0.037
                right_target_midpoint = above_midpoint
                right_gripper_command = 0.002
            elif phase_name == "left_hold":
                left_target_midpoint = left_hold_midpoint
                left_gripper_command = 0.002
                right_target_midpoint = above_midpoint
                right_gripper_command = 0.002
            elif phase_name == "right_above":
                left_target_midpoint = left_hold_midpoint
                left_gripper_command = 0.002
                right_target_midpoint = above_midpoint
                right_gripper_command = 0.002
            elif phase_name == "open_above":
                left_target_midpoint = left_hold_midpoint
                left_gripper_command = 0.002
                right_target_midpoint = above_midpoint
                right_gripper_command = 0.037
            elif phase_name == "descend":
                left_target_midpoint = left_hold_midpoint
                left_gripper_command = 0.002
                right_target_midpoint = grasp_midpoint
                right_gripper_command = 0.037
            elif phase_name == "close":
                left_target_midpoint = left_hold_midpoint
                left_gripper_command = 0.002
                right_target_midpoint = grasp_midpoint
                right_gripper_command = 0.002
            elif phase_name == "lift":
                left_target_midpoint = left_hold_midpoint
                left_gripper_command = 0.002
                right_target_midpoint = lift_midpoint
                right_gripper_command = 0.002
            elif phase_name == "place_above":
                left_target_midpoint = left_hold_midpoint
                left_gripper_command = 0.002
                right_target_midpoint = place_above_midpoint
                right_gripper_command = 0.002
            elif phase_name == "place_down":
                left_target_midpoint = left_hold_midpoint
                left_gripper_command = 0.002
                right_target_midpoint = place_midpoint
                right_gripper_command = 0.002
            elif phase_name == "release":
                left_target_midpoint = left_hold_midpoint
                left_gripper_command = 0.002
                right_target_midpoint = place_midpoint
                right_gripper_command = 0.037
            else:
                left_target_midpoint = left_hold_midpoint
                left_gripper_command = 0.002
                right_target_midpoint = place_above_midpoint
                right_gripper_command = 0.037

            left_target_position = midpoint_to_gripper_target(
                env, left, "left", left_target_midpoint
            )
            right_target_position = midpoint_to_gripper_target(
                env, right, "right", right_target_midpoint
            )

            # Keep the naturally reached left-arm posture while moving onto
            # the tray wall. A full 6D pose target caused the arm to stall or
            # drift instead of maintaining contact with the tray.
            left_position_error = left.move_to_position(
                env.data,
                left_target_position,
                posture_target=left_hold_posture,
                posture_gain=0.16,
            )

            right_position_error = right.move_to_position(
                env.data,
                right_target_position,
                posture_target=grasp_posture,
                posture_gain=0.16,
            )

            left.set_gripper(env.data, left_gripper_command)
            right.set_gripper(env.data, right_gripper_command)
            env.step()
            viewer.sync()

            obs = env.observation()
            tray_now = obs["tray_position"]
            block_now = obs["block_position"]
            left_midpoint_now = finger_midpoint(env, "left")
            right_midpoint_now = finger_midpoint(env, "right")
            target_error = float(
                np.linalg.norm(right_midpoint_now - right_target_midpoint)
            )
            block_alignment = float(np.linalg.norm(right_midpoint_now - block_now))
            horizontal_alignment = float(
                np.linalg.norm(right_midpoint_now[:2] - block_now[:2])
            )
            vertical_offset = float(right_midpoint_now[2] - block_now[2])
            left_hold_error = float(
                np.linalg.norm(left_midpoint_now - left_hold_midpoint)
            )
            actual_right_finger = float(env.data.qpos[right_finger_qpos_address])
            actual_left_finger = float(env.data.qpos[left_finger_qpos_address])
            lift_height = float(block_now[2] - block_start[2])
            block_to_tray_xy = float(np.linalg.norm((block_now - tray_now)[:2]))

            if loop_start - last_print >= 0.5:
                print(
                    f"phase={phase_name} | "
                    f"L target={left_position_error:.3f} hold={left_hold_error:.3f} "
                    f"grip={actual_left_finger:.3f} | "
                    f"R target={target_error:.3f} grip={actual_right_finger:.3f} | "
                    f"xy={horizontal_alignment:.3f} m | "
                    f"z offset={vertical_offset:.3f} m | "
                    f"block align={block_alignment:.3f} m | "
                    f"block lift={lift_height:.3f} m | "
                    f"tray_xy={block_to_tray_xy:.3f}"
                )
                last_print = loop_start

            next_phase = None
            if phase_name == "left_above" and left_position_error < 0.060:
                left_hold_posture = env.data.qpos[
                    left.qpos_addresses
                ].copy()
                next_phase = "left_hold"
            elif phase_name == "left_hold" and left_hold_error < 0.080 and phase_elapsed > 0.8:
                next_phase = "right_above"
            elif phase_name == "right_above" and horizontal_alignment < 0.035 and vertical_offset > 0.070:
                grasp_posture = env.data.qpos[right.qpos_addresses].copy()
                next_phase = "open_above"
            elif phase_name == "open_above" and actual_right_finger > 0.033 and phase_elapsed > 0.5:
                next_phase = "descend"
            elif phase_name == "descend" and horizontal_alignment < 0.025 and abs(vertical_offset - 0.005) < 0.020:
                next_phase = "close"
            elif phase_name == "close" and phase_elapsed > 1.5:
                next_phase = "lift"
            elif phase_name == "lift" and lift_height >= 0.080:
                next_phase = "place_above"
            elif phase_name == "place_above" and block_to_tray_xy < 0.070:
                next_phase = "place_down"
            elif phase_name == "place_down" and block_to_tray_xy < 0.070 and right_position_error < 0.045:
                next_phase = "release"
            elif phase_name == "release" and phase_elapsed > 1.2:
                next_phase = "retreat"

            if next_phase is not None:
                print(f"phase transition: {phase_name} -> {next_phase}")
                phase_name = next_phase
                phase_start = loop_start
                phase_elapsed = 0.0

            if phase_name == "retreat" and phase_elapsed > 2.0 and not finished_reported:
                success = env.success()
                print(
                    f"top-down tray insert finished: "
                    f"lift={lift_height:.3f} m, "
                    f"tray_xy={block_to_tray_xy:.3f} m, "
                    f"success={success}"
                )
                print("Close the viewer window to exit.")
                finished_reported = True

            timeout = (
                45.0
                if phase_name in {"left_above", "right_above", "descend", "lift", "place_above"}
                else 20.0
            )
            if phase_elapsed > timeout and phase_name != "retreat":
                print(
                    f"phase timeout: {phase_name}, "
                    f"target_error={target_error:.3f}, "
                    f"xy={horizontal_alignment:.3f}, "
                    f"z_offset={vertical_offset:.3f}, "
                    f"left_hold={left_hold_error:.3f}"
                )
                break

            remaining = env.model.opt.timestep - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
