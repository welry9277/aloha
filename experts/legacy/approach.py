import time
import sys
from pathlib import Path

import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aloha.controller import AlohaArmController
from aloha.task_env import AlohaTaskEnvironment


def main():
    env = AlohaTaskEnvironment(seed=7)
    observation = env.reset(randomize=True)
    left = AlohaArmController(env.model, "left")
    right = AlohaArmController(env.model, "right")

    _, left_rotation = left.pose(env.data)
    _, right_rotation = right.pose(env.data)

    tray = observation["tray_position"]
    block = observation["block_position"]

    # First expert milestone: reach safe pre-contact poses without touching objects.
    left_target = tray + np.array([-0.18, 0.0, 0.12])
    right_target = block + np.array([0.0, 0.0, 0.14])
    left.set_gripper(env.data, 0.037)
    right.set_gripper(env.data, 0.037)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        last_print = 0.0
        while viewer.is_running():
            loop_start = time.time()
            left_error = left.move_to_pose(
                env.data, left_target, left_rotation
            )
            right_error = right.move_to_pose(
                env.data, right_target, right_rotation
            )
            env.step()
            viewer.sync()

            if loop_start - last_print >= 0.5:
                print(
                    f"left position error={left_error[0]:.4f} m | "
                    f"right position error={right_error[0]:.4f} m"
                )
                last_print = loop_start

            remaining = env.model.opt.timestep - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
