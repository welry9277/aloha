import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aloha.task_env import AlohaTaskEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--sampled-actions",
        action="store_true",
        help=(
            "Replay the 10 Hz sampled actions used to train ACT, holding each "
            "action for image_stride MuJoCo steps. By default, replay the "
            "original full-rate expert actions."
        ),
    )
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

    if args.sampled_actions:
        actions = episode["actions"]
        action_repeat = int(np.asarray(episode["image_stride"]).item())
        mode = "sampled"
    else:
        actions = episode["full_actions"]
        action_repeat = 1
        mode = "full-rate"

    print(
        f"replay mode={mode}, actions={len(actions)}, "
        f"action_repeat={action_repeat}"
    )
    viewer_context = (
        mujoco.viewer.launch_passive(env.model, env.data)
        if not args.no_viewer
        else None
    )

    def replay(viewer):
        for action in actions:
            for _ in range(action_repeat):
                start = time.time()
                env.data.ctrl[:] = action
                mujoco.mj_step(env.model, env.data)
                if viewer is not None:
                    if not viewer.is_running():
                        return
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
