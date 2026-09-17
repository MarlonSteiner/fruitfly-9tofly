"""
Render the DeepMind/Janelia flybody model with MuJoCo.

This is the real anatomical fruit fly used in the Nature 2025 locomotion
paper, not a drawing. We render it offline to image frames so the browser
never has to load 140 MB of mesh.

Model: mujoco_menagerie/flybody, Apache-2.0.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

MODEL = Path("vendor/menagerie/flybody/scene.xml")
OUT = Path("web/frames")


def describe(model):
    print(f"bodies   {model.nbody}")
    print(f"joints   {model.njnt}")
    print(f"geoms    {model.ngeom}")
    print(f"meshes   {model.nmesh}")
    print(f"cameras  {model.ncam}")
    print(f"dof      {model.nv}")
    print()

    print("cameras:")
    for i in range(model.ncam):
        print("  ", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i))
    print()

    print("first 30 joints:")
    for i in range(min(30, model.njnt)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"   {i:3d} {name}")


def main():
    if not MODEL.exists():
        sys.exit(f"model not found: {MODEL}")

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    describe(model)

    mujoco.mj_forward(model, data)

    OUT.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=1080, width=1440) as renderer:
        renderer.update_scene(data, camera="hero")
        pixels = renderer.render()

    import imageio.v3 as iio

    iio.imwrite(OUT / "test.png", pixels)
    print(f"\nrendered {pixels.shape} -> {OUT}/test.png")


if __name__ == "__main__":
    main()
