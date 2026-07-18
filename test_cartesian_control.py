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

ARM_JOINT_SUFFIXES = [
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

    for suffix in ARM_JOINT_SUFFIXES:
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

        if actuator_id == -1 or joint_id == -1:
            raise ValueError(f"Missing actuator or joint: {name}")

        actuator_ids.append(actuator_id)
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        dof_addresses.append(model.jnt_dofadr[joint_id])

    return (
        np.asarray(actuator_ids),
        np.asarray(qpos_addresses),
        np.asarray(dof_addresses),
    )


def get_site_position(model, data, site_name):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    if site_id == -1:
        raise ValueError(f"Missing site: {site_name}")

    return data.site_xpos[site_id].copy()


def cartesian_position_control(
    model,
    data,
    site_name,
    target_position,
    actuator_ids,
    qpos_addresses,
    dof_addresses,
    damping=0.03,
    gain=0.35,
    max_joint_step=0.04,
):
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    current_position = data.site_xpos[site_id]
    position_error = target_position - current_position

    jacobian_position = np.zeros((3, model.nv))
    jacobian_rotation = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        site_id,
    )

    arm_jacobian = jacobian_position[:, dof_addresses]

    # Damped least-squares inverse kinematics
    regularizer = (damping ** 2) * np.eye(3)

    joint_delta = (
        arm_jacobian.T
        @ np.linalg.solve(
            arm_jacobian @ arm_jacobian.T + regularizer,
            gain * position_error,
        )
    )

    joint_delta = np.clip(
        joint_delta,
        -max_joint_step,
        max_joint_step,
    )

    target_joints = data.qpos[qpos_addresses] + joint_delta

    for index, actuator_id in enumerate(actuator_ids):
        if model.actuator_ctrllimited[actuator_id]:
            low, high = model.actuator_ctrlrange[actuator_id]
            target_joints[index] = np.clip(
                target_joints[index],
                low,
                high,
            )

    data.ctrl[actuator_ids] = target_joints

    return np.linalg.norm(position_error)


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    left_indices = get_arm_indices(model, "left")
    right_indices = get_arm_indices(model, "right")

    left_start = get_site_position(
        model,
        data,
        "left/gripper",
    )
    right_start = get_site_position(
        model,
        data,
        "right/gripper",
    )

    print("left start:", left_start)
    print("right start:", right_start)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        last_print = 0.0

        while viewer.is_running():
            loop_start = time.time()
            elapsed = loop_start - start_time

            # 양손을 반대 방향으로 위아래 이동
            z_offset = 0.06 * np.sin(
                2.0 * np.pi * elapsed / 4.0
            )

            left_target = left_start.copy()
            right_target = right_start.copy()

            left_target[2] += z_offset
            right_target[2] -= z_offset

            left_error = cartesian_position_control(
                model,
                data,
                "left/gripper",
                left_target,
                *left_indices,
            )

            right_error = cartesian_position_control(
                model,
                data,
                "right/gripper",
                right_target,
                *right_indices,
            )

            mujoco.mj_step(model, data)
            viewer.sync()

            if elapsed - last_print >= 0.5:
                print(
                    f"left error={left_error:.4f} m | "
                    f"right error={right_error:.4f} m"
                )
                last_print = elapsed

            remaining = model.opt.timestep - (
                time.time() - loop_start
            )
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
