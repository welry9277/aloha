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


model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

left = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "left/waist",
)
right = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "right/waist",
)

left_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "left/waist",
)
right_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "right/waist",
)

left_qadr = model.jnt_qposadr[left_joint]
right_qadr = model.jnt_qposadr[right_joint]

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    last_print = 0.0

    while viewer.is_running():
        loop_start = time.time()
        elapsed = loop_start - start

        target = 0.8 * np.sin(2 * np.pi * elapsed / 4.0)

        data.ctrl[left] = target
        data.ctrl[right] = -target

        mujoco.mj_step(model, data)
        viewer.sync()

        if elapsed - last_print > 0.5:
            print(
                f"ctrl={target:+.3f} | "
                f"left={data.qpos[left_qadr]:+.3f} | "
                f"right={data.qpos[right_qadr]:+.3f}"
            )
            last_print = elapsed

        wait = model.opt.timestep - (time.time() - loop_start)
        if wait > 0:
            time.sleep(wait)
