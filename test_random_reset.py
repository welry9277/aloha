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


def freejoint_qpos_address(model, joint_name):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id == -1:
        raise ValueError(f"Joint not found: {joint_name}")

    return model.jnt_qposadr[joint_id]


def set_freejoint_pose(data, qpos_address, position, yaw=0.0):
    """Free joint qpos = [x, y, z, qw, qx, qy, qz]."""
    half_yaw = yaw / 2.0

    quaternion = np.array([
        np.cos(half_yaw),
        0.0,
        0.0,
        np.sin(half_yaw),
    ])

    data.qpos[qpos_address:qpos_address + 3] = position
    data.qpos[qpos_address + 3:qpos_address + 7] = quaternion


def reset_task(model, data, rng):
    # 로봇과 actuator를 공식 초기 자세로 복원
    mujoco.mj_resetDataKeyframe(model, data, 0)

    tray_qadr = freejoint_qpos_address(model, "tray/freejoint")
    block_qadr = freejoint_qpos_address(model, "red_block/freejoint")

    tray_position = np.array([
        rng.uniform(-0.10, 0.10),
        rng.uniform(0.12, 0.20),
        0.018,
    ])

    block_position = np.array([
        rng.uniform(-0.12, 0.12),
        rng.uniform(-0.16, -0.08),
        0.025,
    ])

    tray_yaw = rng.uniform(-0.15, 0.15)

    set_freejoint_pose(
        data,
        tray_qadr,
        tray_position,
        yaw=tray_yaw,
    )

    set_freejoint_pose(
        data,
        block_qadr,
        block_position,
        yaw=0.0,
    )

    # resetDataKeyframe가 velocity를 초기화하지만 명시적으로 한 번 더 처리
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0

    mujoco.mj_forward(model, data)

    return {
        "tray_position": tray_position.copy(),
        "block_position": block_position.copy(),
        "tray_yaw": tray_yaw,
    }


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed=42)

    task_info = reset_task(model, data, rng)
    print(task_info)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_reset = time.time()

        while viewer.is_running():
            loop_start = time.time()

            # 확인을 위해 3초마다 자동 reset
            if loop_start - last_reset >= 3.0:
                task_info = reset_task(model, data, rng)
                print(task_info)
                last_reset = loop_start

            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
