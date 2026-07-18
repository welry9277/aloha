from pathlib import Path

import mujoco
import numpy as np


CAMERAS = ("overhead_cam", "wrist_cam_left", "wrist_cam_right")


def actuator_qpos(model, data):
    values = []
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        if joint_id < 0:
            values.append(0.0)
        else:
            values.append(data.qpos[model.jnt_qposadr[joint_id]])
    return np.asarray(values, dtype=np.float32)


class DemonstrationRecorder:
    def __init__(
        self,
        model,
        data,
        output_path,
        instruction,
        tray_goal,
        image_hz=10.0,
        image_size=224,
    ):
        self.model = model
        self.data = data
        self.output_path = Path(output_path)
        self.instruction = instruction
        self.tray_goal = np.asarray(tray_goal, dtype=np.float32)
        self.initial_qpos = data.qpos.copy()
        self.initial_qvel = data.qvel.copy()
        self.initial_ctrl = data.ctrl.copy()
        control_hz = 1.0 / model.opt.timestep
        self.image_stride = max(1, int(round(control_hz / image_hz)))
        self.renderer = mujoco.Renderer(
            model,
            height=image_size,
            width=image_size,
        )
        self.step_index = 0
        self.full_actions = []
        self.sample_indices = []
        self.states = []
        self.actions = []
        self.images = {name: [] for name in CAMERAS}

    def record_step(self):
        self.full_actions.append(self.data.ctrl.astype(np.float32).copy())
        if self.step_index % self.image_stride == 0:
            self.sample_indices.append(self.step_index)
            self.states.append(actuator_qpos(self.model, self.data))
            self.actions.append(self.data.ctrl.astype(np.float32).copy())
            for camera in CAMERAS:
                self.renderer.update_scene(self.data, camera=camera)
                self.images[camera].append(self.renderer.render().copy())
        self.step_index += 1

    def save(self, success):
        self.renderer.close()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "instruction": np.asarray(self.instruction),
            "success": np.asarray(bool(success)),
            "timestep": np.asarray(self.model.opt.timestep),
            "image_stride": np.asarray(self.image_stride),
            "tray_goal": self.tray_goal,
            "initial_qpos": self.initial_qpos,
            "initial_qvel": self.initial_qvel,
            "initial_ctrl": self.initial_ctrl,
            "full_actions": np.asarray(self.full_actions, dtype=np.float32),
            "sample_indices": np.asarray(self.sample_indices, dtype=np.int32),
            "states": np.asarray(self.states, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.float32),
        }
        for camera, frames in self.images.items():
            payload[f"images_{camera}"] = np.asarray(frames, dtype=np.uint8)
        np.savez_compressed(self.output_path, **payload)
        return self.output_path

