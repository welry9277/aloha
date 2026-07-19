import re
from pathlib import Path


TASK_INSTRUCTIONS = {
    "left_tray_push": (
        "Use the left arm to push the blue tray to the green target."
    ),
    "right_tray_push": (
        "Use the right arm to push the blue tray to the green target."
    ),
    "left_pick_place": (
        "Use the left arm to pick up the red block and place it "
        "inside the blue tray."
    ),
    "right_pick_place": (
        "Use the right arm to pick up the red block and place it "
        "inside the blue tray."
    ),
    "seen_lr": (
        "First, use the left arm to push the blue tray to the green target. "
        "Then, use the right arm to pick up the red block and place it "
        "inside the blue tray."
    ),
    "unseen_rl": (
        "First, use the right arm to push the blue tray to the green target. "
        "Then, use the left arm to pick up the red block and place it "
        "inside the blue tray."
    ),
}


def task_from_path(path):
    path_string = Path(path).as_posix().lower()

    for task in (
        "left_tray_push",
        "right_tray_push",
        "left_pick_place",
        "right_pick_place",
    ):
        if task in path_string:
            return task

    if "/seen_lr/" in path_string or "demonstrations_lr" in path_string:
        return "seen_lr"

    raise ValueError(f"Cannot infer task from path: {path}")


def instruction_from_path(path):
    return TASK_INSTRUCTIONS[task_from_path(path)]


def tokenize_instruction(instruction):
    return re.findall(r"[a-z0-9']+", instruction.lower())
