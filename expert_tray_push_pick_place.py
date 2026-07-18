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
        tray_y = rng.uniform(0.135, 0.165)
        tray_start = np.array(
            [rng.uniform(-0.055, -0.025), tray_y, 0.018]
        )
        tray_goal = np.array(
            [rng.uniform(0.080, 0.115), tray_y, 0.018]
        )
        block_start = np.array(
            [
                rng.uniform(-0.035, 0.035),
                rng.uniform(-0.140, -0.105),
                0.025,
            ]
        )
    else:
        tray_start = np.array([-0.04, 0.15, 0.018])
        tray_goal = np.array([0.10, 0.15, 0.018])
        block_start = np.array([0.0, -0.12, 0.025])

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
    right = AlohaArmController(env.model, "right")
    left_initial_qpos = env.data.qpos[left.qpos_addresses].copy()
    right_initial_qpos = env.data.qpos[right.qpos_addresses].copy()

    left_sites = (
        site_id(env.model, "left/left_finger"),
        site_id(env.model, "left/right_finger"),
    )
    right_sites = (
        site_id(env.model, "right/left_finger"),
        site_id(env.model, "right/right_finger"),
    )

    right_finger_joint = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_JOINT, "right/left_finger"
    )
    right_finger_qpos = env.model.jnt_qposadr[right_finger_joint]

    # Left arm pushes the outside of the tray's left wall along world +X.
    # Aim at the wall centre height. The left wall's outer face is at about
    # tray_x - 0.148 m, so keep the finger midpoint slightly outside it; the
    # finger collision geometry, rather than the site itself, makes contact.
    push_height = 0.025
    wall_contact_offset = -0.158
    push_start = tray_start + np.array(
        [wall_contact_offset, 0.0, push_height]
    )
    push_end = tray_goal + np.array(
        [wall_contact_offset, 0.0, push_height]
    )
    # First align directly above the tray's left wall, then descend to contact.
    # The previous waypoint was shifted another 7 cm outward in X and caused
    # the arm to settle away from the tray.
    push_approach = push_start + np.array([0.0, 0.0, 0.100])
    push_retreat = push_end + np.array([-0.080, 0.0, 0.100])

    grasp_midpoint = block_start + np.array([0.0, 0.0, 0.005])
    above_midpoint = grasp_midpoint + np.array([0.0, 0.0, 0.13])
    lift_midpoint = grasp_midpoint + np.array([0.0, 0.0, 0.16])
    place_above = tray_goal + np.array([0.0, 0.0, 0.17])
    place_down = tray_goal + np.array([0.0, 0.0, 0.075])

    print("bimanual task: left tray push + right block place")
    print("tray start:", tray_start)
    print("tray goal:", tray_goal)
    print("block start:", block_start)
    print("left approach waypoint:", push_approach)
    print("left contact waypoint:", push_start)

    left.set_gripper(env.data, 0.002)
    right.set_gripper(env.data, 0.002)

    instruction = (
        "왼팔로 파란 트레이를 초록색 목표까지 밀고, "
        "오른팔로 빨간 블록을 트레이 안에 넣어라."
    )
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
        phase = "tray_approach"
        phase_start = env.data.time
        last_print = 0.0
        right_grasp_posture = None
        finished_reported = False
        done_wall_start = None

        while viewer is None or viewer.is_running():
            loop_start = time.time()
            phase_elapsed = env.data.time - phase_start

            left_mid = finger_midpoint(env.data, *left_sites)
            right_mid = finger_midpoint(env.data, *right_sites)

            if phase == "tray_approach":
                left_desired = push_approach
            elif phase == "tray_contact":
                left_desired = push_start
            elif phase == "tray_push":
                left_desired = push_end
            else:
                left_desired = push_retreat

            left_target = midpoint_target(
                env.data, left, left_mid, left_desired
            )
            if phase == "tray_push":
                left_gain = 0.75
                left_max_joint_step = 0.075
            else:
                left_gain = 0.35
                left_max_joint_step = 0.040
            left_error = left.move_to_position(
                env.data,
                left_target,
                gain=left_gain,
                max_joint_step=left_max_joint_step,
                posture_target=left_initial_qpos,
                posture_gain=0.16,
            )
            left.set_gripper(env.data, 0.002)

            if phase in {
                "tray_approach",
                "tray_contact",
                "tray_push",
                "tray_retreat",
            }:
                # The task is intentionally sequential at first.
                env.data.ctrl[right.actuator_ids] = right_initial_qpos
                right.set_gripper(env.data, 0.002)
                right_desired = above_midpoint
                right_error = float(np.linalg.norm(right_mid - right_desired))
            else:
                if phase in {"right_above", "open_above"}:
                    right_desired = above_midpoint
                elif phase in {"descend", "close"}:
                    right_desired = grasp_midpoint
                elif phase == "lift":
                    right_desired = lift_midpoint
                elif phase in {"place_above", "retreat", "done"}:
                    right_desired = place_above
                else:
                    right_desired = place_down

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
                elif phase in {"right_above", "open_above", "descend", "close", "lift"}:
                    right_error = right.move_to_position(
                        env.data,
                        right_target,
                        gain=0.45,
                        max_joint_step=0.055,
                        posture_target=right_grasp_posture,
                        posture_gain=0.16,
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
            left_mid = finger_midpoint(env.data, *left_sites)
            right_mid = finger_midpoint(env.data, *right_sites)
            left_error = float(np.linalg.norm(left_mid - left_desired))
            left_xy_error = float(
                np.linalg.norm(left_mid[:2] - left_desired[:2])
            )
            left_z_error = float(abs(left_mid[2] - left_desired[2]))
            right_error = float(np.linalg.norm(right_mid - right_desired))
            tray_goal_error = float(np.linalg.norm(tray_now[:2] - tray_goal[:2]))
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
            actual_right_finger = float(env.data.qpos[right_finger_qpos])

            if loop_start - last_print >= 0.5:
                print(
                    f"phase={phase} | L err={left_error:.3f} "
                    f"xy={left_xy_error:.3f} z={left_z_error:.3f} | "
                    f"tray goal={tray_goal_error:.3f} | "
                    f"R err={right_error:.3f} grip={actual_right_finger:.3f} | "
                    f"block xy={block_xy_error:.3f} z={block_z_offset:.3f} | "
                    f"lift={block_lift:.3f} speed={block_speed:.3f} | "
                    f"tray_xy={block_to_tray:.3f}"
                )
                last_print = loop_start

            next_phase = None
            if (
                phase == "tray_approach"
                and left_xy_error < 0.045
                and left_z_error < 0.060
            ):
                next_phase = "tray_contact"
            elif phase == "tray_contact" and left_error < 0.045:
                next_phase = "tray_push"
            elif phase == "tray_push" and tray_goal_error < 0.055:
                next_phase = "tray_retreat"
                # Place relative to the tray's actual post-push pose.
                place_above = tray_now + np.array([0.0, 0.0, 0.17])
                place_down = tray_now + np.array([0.0, 0.0, 0.075])
            elif phase == "tray_retreat" and left_error < 0.065:
                next_phase = "right_above"
            elif (
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
            elif phase == "close" and phase_elapsed > 1.5:
                next_phase = "lift"
            elif phase == "lift" and block_lift >= 0.080:
                next_phase = "place_above"
            elif phase == "place_above" and block_to_tray < 0.070:
                next_phase = "place_down"
            elif (
                phase == "place_down"
                and block_to_tray < 0.070
                and right_error < 0.055
            ):
                next_phase = "release"
            elif (
                phase == "release"
                and phase_elapsed > 2.0
                and actual_right_finger > 0.033
                and block_speed < 0.050
            ):
                next_phase = "retreat"
            elif (
                phase == "retreat"
                and right_mid[2] - block_now[2] > 0.040
            ):
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
                tray_ok = tray_goal_error < 0.070
                block_ok = (
                    abs(block_now[0] - tray_now[0]) < 0.11
                    and abs(block_now[1] - tray_now[1]) < 0.07
                    and block_now[2] - tray_now[2] < 0.08
                )
                print(
                    f"finished: tray_ok={tray_ok}, block_ok={block_ok}, "
                    f"success={tray_ok and block_ok}"
                )
                result_success = bool(tray_ok and block_ok)
                finished_reported = True
                break

            timeout = 55.0 if phase not in {"close", "release", "done"} else 20.0
            if phase_elapsed > timeout and phase != "done":
                print(
                    f"phase timeout: {phase}, left_error={left_error:.3f}, "
                    f"right_error={right_error:.3f}, "
                    f"tray_goal_error={tray_goal_error:.3f}"
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
