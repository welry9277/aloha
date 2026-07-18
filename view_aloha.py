import time
from pathlib import Path

import mujoco
import mujoco.viewer


MODEL_PATH = (
    Path(__file__).parent
    / "mujoco_menagerie"
    / "aloha"
    / "scene.xml"
).resolve()


def main():
    print(f"Loading: {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"cameras={model.ncam}")

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
