from pathlib import Path

import mujoco
import numpy as np


DEFAULT_MODEL_PATH = (
    Path(__file__).parent
    / "mujoco_menagerie"
    / "aloha"
    / "task_scene.xml"
).resolve()


class AlohaTaskEnvironment:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, seed=0):
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.tray_joint = self._object_id(mujoco.mjtObj.mjOBJ_JOINT, "tray/freejoint")
        self.block_joint = self._object_id(
            mujoco.mjtObj.mjOBJ_JOINT, "red_block/freejoint"
        )
        self.tray_body = self._object_id(mujoco.mjtObj.mjOBJ_BODY, "tray")
        self.block_body = self._object_id(mujoco.mjtObj.mjOBJ_BODY, "red_block")

    def _object_id(self, object_type, name):
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id == -1:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    def _set_freejoint_pose(self, joint_id, position, yaw=0.0):
        qpos_address = self.model.jnt_qposadr[joint_id]
        half_yaw = yaw / 2.0
        self.data.qpos[qpos_address : qpos_address + 3] = position
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = (
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        )

    def reset(self, randomize=True):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        if randomize:
            tray_position = np.array(
                [
                    self.rng.uniform(-0.08, 0.08),
                    self.rng.uniform(0.13, 0.19),
                    0.018,
                ]
            )
            block_position = np.array(
                [
                    self.rng.uniform(-0.10, 0.10),
                    self.rng.uniform(-0.15, -0.09),
                    0.025,
                ]
            )
            tray_yaw = self.rng.uniform(-0.10, 0.10)
        else:
            # Keyframes created before task objects were added initialize new
            # free joints at zero. Reapply the intended fixed task layout.
            tray_position = np.array([0.0, 0.15, 0.018])
            block_position = np.array([0.0, -0.12, 0.025])
            tray_yaw = 0.0

        self._set_freejoint_pose(self.tray_joint, tray_position, tray_yaw)
        self._set_freejoint_pose(self.block_joint, block_position)

        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def step(self, frame_skip=1):
        for _ in range(frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self.observation()

    def observation(self):
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "tray_position": self.data.xpos[self.tray_body].copy(),
            "block_position": self.data.xpos[self.block_body].copy(),
        }
