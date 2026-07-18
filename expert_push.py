import time

import mujoco.viewer
import numpy as np

from aloha_controller import AlohaArmController
from aloha_task_env import AlohaTaskEnvironment


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def interpolate(start, end, progress):
    return start + smoothstep(progress) * (end - start)


def main():
    env = AlohaTaskEnvironment(seed=11)
    observation = env.reset(randomize=True)
    left = AlohaArmController(env.model, "left")
    right = AlohaArmController(env.model, "right")

    left_home, _ = left.pose(env.data)
    right_home, _ = right.pose(env.data)
    tray = observation["tray_position"]
    block = observation["block_position"]

    # Safe poses above the contact locations.
    left_safe = tray + np.array([-0.17, 0.0, 0.13])
    right_safe = block + np.array([0.0, -0.04, 0.13])

    # The left arm braces the tray while the right arm pushes from behind.
    left_brace = tray + np.array([-0.155, 0.0, 0.045])
    right_behind = block + np.array([0.0, -0.04, 0.035])
    right_finish = tray + np.array([0.0, 0.025, 0.035])

    left.set_gripper(env.data, 0.010)
    right.set_gripper(env.data, 0.010)

    phases = (
        ("approach", 5.0),
        ("descend", 4.0),
        ("push", 6.0),
        ("hold", 1.0),
        ("retreat", 3.0),
    )
    phase_starts = np.cumsum([0.0] + [duration for _, duration in phases])

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        start_time = time.time()
        previous_phase = None
        last_print = 0.0

        while viewer.is_running():
            loop_start = time.time()
            elapsed = loop_start - start_time
            if elapsed >= phase_starts[-1]:
                tray_now = env.observation()["tray_position"]
                block_now = env.observation()["block_position"]
                relative = block_now - tray_now
                success = abs(relative[0]) < 0.11 and abs(relative[1]) < 0.075
                print(f"finished: relative block pose={relative}, success={success}")
                break

            phase_index = np.searchsorted(phase_starts[1:], elapsed, side="right")
            phase_name, phase_duration = phases[phase_index]
            phase_time = elapsed - phase_starts[phase_index]
            progress = phase_time / phase_duration

            if phase_name == "approach":
                left_target = interpolate(left_home, left_safe, progress)
                right_target = interpolate(right_home, right_safe, progress)
            elif phase_name == "descend":
                left_target = interpolate(left_safe, left_brace, progress)
                right_target = interpolate(right_safe, right_behind, progress)
            elif phase_name == "push":
                left_target = left_brace
                right_target = interpolate(right_behind, right_finish, progress)
            elif phase_name == "hold":
                left_target = left_brace
                right_target = right_finish
            else:
                left_target = interpolate(left_brace, left_home, progress)
                right_target = interpolate(right_finish, right_home, progress)

            left_error = left.move_to_position(env.data, left_target)
            right_error = right.move_to_position(env.data, right_target)
            env.step()
            viewer.sync()

            if phase_name != previous_phase:
                print(f"phase: {phase_name}")
                previous_phase = phase_name
            if loop_start - last_print >= 0.5:
                print(
                    f"L error={left_error:.3f} m | "
                    f"R error={right_error:.3f} m"
                )
                last_print = loop_start

            remaining = env.model.opt.timestep - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
