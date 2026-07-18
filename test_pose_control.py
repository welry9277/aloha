import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = (
    Path(__file__).parent
    / "mujoco_menagerie"
    / "aloha"
    / "task_scene.xml"
).resolve()

ARM_JOINTS = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]


def get_arm_indices(model, side):
    actuator_ids = []
    qpos_addresses = []
    dof_addresses = []

    for suffix in ARM_JOINTS:
        name = f"{side}/{suffix}"

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        actuator_ids.append(actuator_id)
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        dof_addresses.append(model.jnt_dofadr[joint_id])

    return (
        np.asarray(actuator_ids),
        np.asarray(qpos_addresses),
        np.asarray(dof_addresses),
    )


def rotation_error(target_rotation, current_rotation):
    """현재 방향에서 목표 방향으로 가는 world-frame 회전 오차."""
    relative = target_rotation @ current_rotation.T

    vector = np.array([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ])

    cosine = np.clip(
        (np.trace(relative) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    angle = np.arccos(cosine)

    if angle < 1e-6:
        return 0.5 * vector

    return angle * vector / (2.0 * np.sin(angle))


def z_rotation(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)

    return np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])


def pose_control(
    model,
    data,
    site_name,
    target_position,
    target_rotation,
    actuator_ids,
    qpos_addresses,
    dof_addresses,
    damping=0.04,
    position_gain=0.35,
    rotation_gain=0.20,
    max_joint_step=0.035,
):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    current_position = data.site_xpos[site_id].copy()
    current_rotation = data.site_xmat[site_id].reshape(3, 3).copy()

    position_error = target_position - current_position
    orientation_error = rotation_error(
        target_rotation,
        current_rotation,
    )

    error = np.concatenate([
        position_gain * position_error,
        rotation_gain * orientation_error,
    ])

    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        site_id,
    )

    jacobian = np.vstack([
        jacobian_position[:, dof_addresses],
        jacobian_rotation[:, dof_addresses],
    ])

    regularizer = (damping ** 2) * np.eye(6)

    joint_delta = (
        jacobian.T
        @ np.linalg.solve(
            jacobian @ jacobian.T + regularizer,
            error,
        )
    )

    joint_delta = np.clip(
        joint_delta,
        -max_joint_step,
        max_joint_step,
    )

    joint_targets = data.qpos[qpos_addresses] + joint_delta

    for index, actuator_id in enumerate(actuator_ids):
        if model.actuator_ctrllimited[actuator_id]:
            low, high = model.actuator_ctrlrange[actuator_id]
            joint_targets[index] = np.clip(
                joint_targets[index],
                low,
                high,
            )

    data.ctrl[actuator_ids] = joint_targets

    return (
        np.linalg.norm(position_error),
        np.linalg.norm(orientation_error),
    )


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    left_indices = get_arm_indices(model, "left")
    right_indices = get_arm_indices(model, "right")

    left_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "left/gripper",
    )
    right_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "right/gripper",
    )

    left_start_position = data.site_xpos[left_site].copy()
    right_start_position = data.site_xpos[right_site].copy()

    left_start_rotation = (
        data.site_xmat[left_site].reshape(3, 3).copy()
    )
    right_start_rotation = (
        data.site_xmat[right_site].reshape(3, 3).copy()
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        last_print = 0.0

        while viewer.is_running():
            loop_start = time.time()
            elapsed = loop_start - start_time

            movement = np.sin(
                2.0 * np.pi * elapsed / 5.0
            )

            left_target_position = left_start_position.copy()
            right_target_position = right_start_position.copy()

            left_target_position[2] += 0.05 * movement
            right_target_position[2] -= 0.05 * movement

            rotation_offset = z_rotation(0.20 * movement)

            left_target_rotation = (
                rotation_offset @ left_start_rotation
            )
            right_target_rotation = (
                rotation_offset.T @ right_start_rotation
            )

            left_position_error, left_rotation_error = pose_control(
                model,
                data,
                "left/gripper",
                left_target_position,
                left_target_rotation,
                *left_indices,
            )

            right_position_error, right_rotation_error = pose_control(
                model,
                data,
                "right/gripper",
                right_target_position,
                right_target_rotation,
                *right_indices,
            )

            mujoco.mj_step(model, data)
            viewer.sync()

            if elapsed - last_print >= 0.5:
                print(
                    f"L pos={left_position_error:.4f}, "
                    f"L rot={left_rotation_error:.4f} | "
                    f"R pos={right_position_error:.4f}, "
                    f"R rot={right_rotation_error:.4f}"
                )
                last_print = elapsed

            remaining = model.opt.timestep - (
                time.time() - loop_start
            )

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
