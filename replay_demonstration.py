import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from aloha_task_env import AlohaTaskEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    episode = np.load(args.episode, allow_pickle=False)
    tray_goal = episode["tray_goal"]
    env = AlohaTaskEnvironment(seed=0)
    env.reset(randomize=False)
    env.data.qpos[:] = episode["initial_qpos"]
    env.data.qvel[:] = episode["initial_qvel"]
    env.data.ctrl[:] = episode["initial_ctrl"]
    target_site = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "target"
    )
    env.model.site_pos[target_site] = np.array(
        [tray_goal[0], tray_goal[1], 0.012]
    )
    mujoco.mj_forward(env.model, env.data)

    actions = episode["full_actions"]
    viewer_context = (
        mujoco.viewer.launch_passive(env.model, env.data)
        if not args.no_viewer
        else None
    )

    def replay(viewer):
        for action in actions:
            start = time.time()
            env.data.ctrl[:] = action
            mujoco.mj_step(env.model, env.data)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                remaining = env.model.opt.timestep - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)

    if viewer_context is None:
        replay(None)
    else:
        with viewer_context as viewer:
            replay(viewer)

    obs = env.observation()
    tray = obs["tray_position"]
    block = obs["block_position"]
    tray_ok = np.linalg.norm(tray[:2] - tray_goal[:2]) < 0.070
    block_ok = (
        abs(block[0] - tray[0]) < 0.11
        and abs(block[1] - tray[1]) < 0.07
        and block[2] - tray[2] < 0.08
    )
    print(
        f"replay: tray_ok={tray_ok}, block_ok={block_ok}, "
        f"success={bool(tray_ok and block_ok)}"
    )


if __name__ == "__main__":
    main()
