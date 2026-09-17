"""
Pose the fly at its desk and render.

The fly perches on the chair seat with its forelegs (T1) up on the laptop
keyboard. Joint angles are set directly -- this is animation, not physics,
which is exactly what the honesty badge calls out as assumed.
"""

import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

SCENE = Path("scenes/desk.xml")
OUT = Path("renders")

# Where the fly sits: on the chair seat, facing the laptop (+x).
# Thorax at z=0 puts the default-pose feet exactly on the seat top (-0.132).
BODY_POS = (0.0, 0.0, 0.0)

# Joint angles for the seated pose, in radians. Anything not listed stays at
# the model default.
SEATED = {
    # front legs reach up and forward onto the keyboard
    "coxa_T1_left": -0.88, "coxa_T1_right": -0.88,
    "coxa_abduct_T1_left": 0.22, "coxa_abduct_T1_right": -0.22,
    "femur_T1_left": -0.78, "femur_T1_right": -0.78,
    "tibia_T1_left": 0.88, "tibia_T1_right": 0.88,
    "tarsus_T1_left": 0.45, "tarsus_T1_right": 0.45,
    # middle and hind legs keep the model default standing pose
    # head tipped down toward the screen
    "head": -0.22,
    # abdomen curled slightly, a settled posture
    "abdomen": 0.12, "abdomen_2": 0.10, "abdomen_3": 0.08,
}


def set_pose(model, data, pos, angles, quat=(1, 0, 0, 0)):
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = pos
    data.qpos[3:7] = quat
    for name, value in angles.items():
        try:
            data.qpos[model.joint(name).qposadr[0]] = value
        except KeyError:
            print(f"  ! no joint named {name}", file=sys.stderr)
    mujoco.mj_forward(model, data)


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    OUT.mkdir(exist_ok=True)

    set_pose(model, data, BODY_POS, SEATED)

    shots = {
        "shot":  dict(lookat=(0.14, 0.0, -0.06), dist=0.92, azim=-118, elev=-10),
        "close": dict(lookat=(0.08, 0.0, -0.01), dist=0.78, azim=-112, elev=-5),
        "over":  dict(lookat=(0.16, 0.0, -0.08), dist=1.05, azim=-100, elev=-26),
        "side":  dict(lookat=(0.10, 0.0, -0.02), dist=1.05, azim=-84,  elev=-7),
    }
    with mujoco.Renderer(model, height=1080, width=1920) as r:
        for name, s in shots.items():
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = s["lookat"]
            cam.distance = s["dist"]
            cam.azimuth = s["azim"]
            cam.elevation = s["elev"]
            r.update_scene(data, camera=cam)
            iio.imwrite(OUT / f"desk_{name}.png", r.render())
            print(f"wrote {OUT}/desk_{name}.png")


if __name__ == "__main__":
    main()
