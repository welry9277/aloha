import time
from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = (
    Path(__file__).parent
    / "mujoco_menagerie"
    / "aloha"
    / "task_scene.xml"
).resolve()


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # 로봇 초기 자세와 actuator target 로드
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    left_gripper = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "left/gripper",
    )

    right_gripper = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "right/gripper",
    )

    print(
        "left range:",
        model.actuator_ctrlrange[left_gripper],
    )
    print(
        "right range:",
        model.actuator_ctrlrange[right_gripper],
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        previous_phase = None

        while viewer.is_running():
            loop_start = time.time()
            elapsed = loop_start - start_time

            # 2초마다 열림/닫힘 전환
            phase = int(elapsed // 2) % 2

            if phase == 0:
                gripper_target = 0.037
                state = "OPEN"
            else:
                gripper_target = 0.002
                state = "CLOSED"

            data.ctrl[left_gripper] = gripper_target
            data.ctrl[right_gripper] = gripper_target

            if phase != previous_phase:
                print(
                    f"{state}: target={gripper_target:.3f}"
                )
                previous_phase = phase

            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (
                time.time() - loop_start
            )

            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
