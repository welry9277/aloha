import mujoco
import numpy as np


ARM_JOINTS = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)


def _rotation_error(target_rotation, current_rotation):
    relative = target_rotation @ current_rotation.T
    vector = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cosine)
    if angle < 1e-6:
        return 0.5 * vector
    return angle * vector / (2.0 * np.sin(angle))


class AlohaArmController:
    def __init__(self, model, side):
        if side not in {"left", "right"}:
            raise ValueError(f"Unknown arm side: {side}")

        self.model = model
        self.side = side
        self.site_name = f"{side}/gripper"
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, self.site_name
        )
        self.gripper_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}/gripper"
        )

        actuator_ids = []
        qpos_addresses = []
        dof_addresses = []
        for suffix in ARM_JOINTS:
            name = f"{side}/{suffix}"
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if actuator_id == -1 or joint_id == -1:
                raise ValueError(f"Missing ALOHA joint or actuator: {name}")
            actuator_ids.append(actuator_id)
            qpos_addresses.append(model.jnt_qposadr[joint_id])
            dof_addresses.append(model.jnt_dofadr[joint_id])

        self.actuator_ids = np.asarray(actuator_ids)
        self.qpos_addresses = np.asarray(qpos_addresses)
        self.dof_addresses = np.asarray(dof_addresses)

    def pose(self, data):
        position = data.site_xpos[self.site_id].copy()
        rotation = data.site_xmat[self.site_id].reshape(3, 3).copy()
        return position, rotation

    def set_gripper(self, data, command):
        low, high = self.model.actuator_ctrlrange[self.gripper_id]
        data.ctrl[self.gripper_id] = np.clip(command, low, high)

    def move_to_position(
        self,
        data,
        target_position,
        damping=0.03,
        gain=0.35,
        max_joint_step=0.04,
        posture_target=None,
        posture_gain=0.10,
    ):
        current_position, _ = self.pose(data)
        position_error = np.asarray(target_position) - current_position

        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model,
            data,
            jacobian_position,
            jacobian_rotation,
            self.site_id,
        )
        jacobian = jacobian_position[:, self.dof_addresses]
        regularizer = (damping**2) * np.eye(3)
        jacobian_pinv = jacobian.T @ np.linalg.inv(
            jacobian @ jacobian.T + regularizer
        )
        joint_delta = jacobian_pinv @ (gain * position_error)
        if posture_target is not None:
            posture_error = (
                np.asarray(posture_target)
                - data.qpos[self.qpos_addresses]
            )
            nullspace = np.eye(len(self.dof_addresses)) - (
                jacobian_pinv @ jacobian
            )
            joint_delta += (
                posture_gain * nullspace @ posture_error
            )
        joint_delta = np.clip(joint_delta, -max_joint_step, max_joint_step)
        joint_targets = data.qpos[self.qpos_addresses] + joint_delta

        for index, actuator_id in enumerate(self.actuator_ids):
            if self.model.actuator_ctrllimited[actuator_id]:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                joint_targets[index] = np.clip(joint_targets[index], low, high)
        data.ctrl[self.actuator_ids] = joint_targets
        return float(np.linalg.norm(position_error))

    def move_to_orientation(
        self,
        data,
        target_rotation,
        damping=0.04,
        gain=0.25,
        max_joint_step=0.035,
    ):
        """Rotate the tool without simultaneously constraining its position."""
        _, current_rotation = self.pose(data)
        orientation_error = _rotation_error(
            target_rotation,
            current_rotation,
        )

        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model,
            data,
            jacobian_position,
            jacobian_rotation,
            self.site_id,
        )
        jacobian = jacobian_rotation[:, self.dof_addresses]
        regularizer = (damping**2) * np.eye(3)
        joint_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + regularizer,
            gain * orientation_error,
        )
        joint_delta = np.clip(joint_delta, -max_joint_step, max_joint_step)
        joint_targets = data.qpos[self.qpos_addresses] + joint_delta

        for index, actuator_id in enumerate(self.actuator_ids):
            if self.model.actuator_ctrllimited[actuator_id]:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                joint_targets[index] = np.clip(joint_targets[index], low, high)
        data.ctrl[self.actuator_ids] = joint_targets
        return float(np.linalg.norm(orientation_error))

    def move_to_pose(
        self,
        data,
        target_position,
        target_rotation,
        damping=0.04,
        position_gain=0.35,
        rotation_gain=0.20,
        max_joint_step=0.035,
    ):
        current_position, current_rotation = self.pose(data)
        position_error = np.asarray(target_position) - current_position
        orientation_error = _rotation_error(target_rotation, current_rotation)
        error = np.concatenate(
            [position_gain * position_error, rotation_gain * orientation_error]
        )

        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model,
            data,
            jacobian_position,
            jacobian_rotation,
            self.site_id,
        )
        jacobian = np.vstack(
            [
                jacobian_position[:, self.dof_addresses],
                jacobian_rotation[:, self.dof_addresses],
            ]
        )
        regularizer = (damping**2) * np.eye(6)
        joint_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + regularizer, error
        )
        joint_delta = np.clip(joint_delta, -max_joint_step, max_joint_step)
        joint_targets = data.qpos[self.qpos_addresses] + joint_delta

        for index, actuator_id in enumerate(self.actuator_ids):
            if self.model.actuator_ctrllimited[actuator_id]:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                joint_targets[index] = np.clip(joint_targets[index], low, high)
        data.ctrl[self.actuator_ids] = joint_targets

        return float(np.linalg.norm(position_error)), float(
            np.linalg.norm(orientation_error)
        )
