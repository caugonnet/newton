"""Headless PNG screenshots of the Newton scenes in the 2026-11-07 record.

Drives newton.viewer.ViewerGL(headless=True) (EGL via pyglet) the way the
example runner does, steps the example for a few frames and saves
viewer.get_frame() with PIL.

  python newton_scene_screenshots.py OUT_DIR [scene ...]

Scenes (bare example names + extra flags, as in REPRODUCE.md section 2);
default is the whole list. Env SHOT_FRAMES (default 60) sets how many frames
to simulate before the shot, SHOT_SIZE "WxH" (default 1280x720).
"""

import importlib
import os
import sys

import pyglet

pyglet.options["headless"] = True

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from PIL import Image  # noqa: E402

import newton  # noqa: E402
import newton.examples  # noqa: E402
import newton.viewer  # noqa: E402

FRAMES = int(os.environ.get("SHOT_FRAMES", 60))
W, H = (int(v) for v in os.environ.get("SHOT_SIZE", "1280x720").split("x"))

SCENES = {
    "franka_vbd_cable_mujoco": ("multiphysics.example_mujoco_franka_vbd_cable_admm_solver", []),
    "franka_vbd_cable_steel": ("multiphysics.example_mujoco_franka_vbd_cable_admm_solver", ["--rigid-solver", "steel"]),
    "mujoco_vbd": ("multiphysics.example_mujoco_vbd_admm_solver", []),
    "kamino_mujoco": ("multiphysics.example_kamino_mujoco_admm_solver", []),
    "admm_contact": ("multiphysics.example_admm_contact_solver", []),
    "steel_arms": ("steel.example_steel_determinism_arms", []),
    "anymal_lox": ("kamino.example_kamino_robot_anymal_d", ["--dynamics-solver", "lox"]),
    # NVIDIA/newton-samples (external checkout, env SHOT_SAMPLES_ROOT); absolute module path
    "franka_pulley": ("examples.multiphysics.example_franka_pulley_mechanical_advantage", []),
}


def shoot(name, out_dir):
    module_path, extra = SCENES[name]
    if module_path.startswith("examples."):
        sys.path.insert(0, os.environ["SHOT_SAMPLES_ROOT"])
        mod = importlib.import_module(module_path)
    else:
        mod = importlib.import_module(f"newton.examples.{module_path}")
    parser = mod.Example.create_parser() if hasattr(mod.Example, "create_parser") else newton.examples.create_parser()
    headless = ["--headless"] if "--headless" in parser.format_help() else []  # newton-samples' runner has no --headless
    args = parser.parse_args(["--viewer", "gl", *headless, "--num-frames", str(FRAMES + 1), *extra])
    viewer = newton.viewer.ViewerGL(headless=True, num_frames=FRAMES + 1, width=W, height=H)
    example = mod.Example(viewer, args)
    for _ in range(FRAMES):
        example.step()
        example.render()  # the examples' render() wraps begin_frame/end_frame
    frame = viewer.get_frame().numpy()
    wp.synchronize()
    path = os.path.join(out_dir, f"{name}.png")
    Image.fromarray(np.ascontiguousarray(frame)).save(path)
    print(f"SHOT {name} {frame.shape[1]}x{frame.shape[0]} -> {path}")
    viewer.close() if hasattr(viewer, "close") else None


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    names = sys.argv[2:] or list(SCENES)
    for name in names:
        shoot(name, out_dir)


if __name__ == "__main__":
    wp.init()
    main()
