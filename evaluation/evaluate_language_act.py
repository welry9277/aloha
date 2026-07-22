import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aloha.task_env import AlohaTaskEnvironment
from aloha.demonstration_io import CAMERAS, actuator_qpos
from aloha.task_instructions import instruction_from_path

ACT_ROOT = PROJECT_ROOT / "third_party" / "official_act"
if not ACT_ROOT.exists():
    raise FileNotFoundError(f"Vendored official ACT source not found: {ACT_ROOT}")
sys.path.insert(0, str(ACT_ROOT))

from policy import ACTPolicy


def load_stats(path):
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: np.asarray(value, dtype=np.float32) for key, value in payload.items()}


def restore_episode(env, episode):
    env.reset(randomize=False)
    env.data.qpos[:] = episode["initial_qpos"]
    env.data.qvel[:] = episode["initial_qvel"]
    env.data.ctrl[:] = episode["initial_ctrl"]

    tray_goal = np.asarray(episode["tray_goal"], dtype=np.float64)
    target_site = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_SITE, "target"
    )
    if target_site != -1:
        env.model.site_pos[target_site] = np.array(
            [tray_goal[0], tray_goal[1], 0.012]
        )
    mujoco.mj_forward(env.model, env.data)
    return tray_goal


def render_observation(renderer, env):
    frames = []
    for camera in CAMERAS:
        renderer.update_scene(env.data, camera=camera)
        frames.append(renderer.render().copy())
    images = np.stack(frames, axis=0)
    images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32) / 255.0
    return images


def clip_action(model, action):
    action = np.asarray(action, dtype=np.float64).copy()
    if action.shape != (model.nu,):
        raise ValueError(f"Expected action shape ({model.nu},), got {action.shape}")
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    action[limited] = np.clip(
        action[limited],
        model.actuator_ctrlrange[limited, 0],
        model.actuator_ctrlrange[limited, 1],
    )
    return action


def task_success(env, tray_goal):
    observation = env.observation()
    tray = observation["tray_position"]
    block = observation["block_position"]
    tray_error = float(np.linalg.norm(tray[:2] - tray_goal[:2]))
    tray_ok = tray_error < 0.070
    block_ok = (
        abs(block[0] - tray[0]) < 0.11
        and abs(block[1] - tray[1]) < 0.07
        and block[2] - tray[2] < 0.08
    )
    return {
        "tray_ok": bool(tray_ok),
        "block_ok": bool(block_ok),
        "success": bool(tray_ok and block_ok),
        "tray_error": tray_error,
        "tray_position": tray,
        "block_position": block,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop rollout of language-conditioned official ACT."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/language_act_overfit5/best.pt"),
    )
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--max-actions",
        type=int,
        help="Maximum 10 Hz actions. Default uses the demonstration length.",
    )
    parser.add_argument(
        "--execute-actions",
        type=int,
        help=(
            "Actions executed before replanning. Default executes the full "
            "chunk; use 1 to diagnose open-loop compounding."
        ),
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = args.checkpoint.resolve()
    stats_path = (
        args.stats.resolve()
        if args.stats is not None
        else checkpoint_path.parent / "normalization_stats.json"
    )
    stats = load_stats(stats_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy_config = dict(checkpoint["model_config"])
    policy_config["device"] = str(device)
    policy = ACTPolicy(policy_config).to(device)
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.eval()

    episode_path = args.episode.resolve()
    with np.load(episode_path, allow_pickle=False) as episode:
        env = AlohaTaskEnvironment(seed=0)
        tray_goal = restore_episode(env, episode)
        image_stride = int(episode["image_stride"])
        demonstration_length = int(episode["actions"].shape[0])

    instruction = instruction_from_path(episode_path)
    chunk_size = int(policy_config["num_queries"])
    execute_actions = args.execute_actions or chunk_size
    if not 1 <= execute_actions <= chunk_size:
        raise ValueError(
            f"--execute-actions must be in [1, {chunk_size}], got {execute_actions}"
        )
    max_actions = args.max_actions or demonstration_length
    renderer = mujoco.Renderer(env.model, height=224, width=224)
    viewer_context = (
        nullcontext(None)
        if args.no_viewer
        else mujoco.viewer.launch_passive(env.model, env.data)
    )

    print(f"checkpoint={checkpoint_path}")
    print(f"episode={episode_path}")
    print(f"device={device}, instruction={instruction!r}")
    print(
        f"chunk_size={chunk_size}, execute_actions={execute_actions}, "
        f"image_stride={image_stride}, "
        f"max_actions={max_actions}"
    )

    actions_executed = 0
    try:
        with viewer_context as viewer:
            while actions_executed < max_actions:
                if viewer is not None and not viewer.is_running():
                    break

                images = render_observation(renderer, env)
                state = actuator_qpos(env.model, env.data)
                state = (state - stats["state_mean"]) / stats["state_std"]

                image_tensor = torch.from_numpy(images).unsqueeze(0).to(device)
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                with torch.inference_mode():
                    normalized_chunk = policy(
                        state_tensor, image_tensor, [instruction]
                    )[0].cpu().numpy()
                action_chunk = (
                    normalized_chunk * stats["action_std"] + stats["action_mean"]
                )

                remaining = max_actions - actions_executed
                count = min(execute_actions, remaining)
                for action in action_chunk[:count]:
                    env.data.ctrl[:] = clip_action(env.model, action)
                    for _ in range(image_stride):
                        step_start = time.time()
                        mujoco.mj_step(env.model, env.data)
                        if viewer is not None:
                            if not viewer.is_running():
                                break
                            viewer.sync()
                            delay = env.model.opt.timestep - (time.time() - step_start)
                            if delay > 0:
                                time.sleep(delay)
                    actions_executed += 1
                    if actions_executed >= max_actions:
                        break

                result = task_success(env, tray_goal)
                print(
                    f"actions={actions_executed:04d}/{max_actions} | "
                    f"tray_error={result['tray_error']:.3f} | "
                    f"tray_ok={result['tray_ok']} block_ok={result['block_ok']}",
                    flush=True,
                )
    finally:
        renderer.close()

    result = task_success(env, tray_goal)
    print(
        f"finished: tray_ok={result['tray_ok']}, "
        f"block_ok={result['block_ok']}, success={result['success']}"
    )
    print("tray_position:", result["tray_position"])
    print("block_position:", result["block_position"])


if __name__ == "__main__":
    main()
