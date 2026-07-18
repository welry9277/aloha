from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


MODEL_PATH = (
    Path(__file__).parent
    / "mujoco_menagerie"
    / "aloha"
    / "scene.xml"
).resolve()

OUTPUT_DIR = Path(__file__).parent / "camera_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print("Available cameras:")

for camera_id in range(model.ncam):
    camera_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_id,
    )
    print(camera_id, camera_name)

renderer = mujoco.Renderer(
    model,
    height=224,
    width=224,
)

camera_names = [
    "overhead_cam",
    "wrist_cam_left",
    "wrist_cam_right",
]

for camera_name in camera_names:
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()

    print(
        camera_name,
        rgb.shape,
        rgb.dtype,
        f"min={rgb.min()}, max={rgb.max()}",
    )

    output_path = OUTPUT_DIR / f"{camera_name}.png"
    Image.fromarray(rgb).save(output_path)

    print("saved:", output_path.resolve())

renderer.close()
