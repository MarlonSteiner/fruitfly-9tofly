"""
Render the two ends of the shift: working, and startled off the chair.

Composed full-bleed for a 16:9 hero. The fly sits left of centre so the
bottom-right stays clear for the brain panel and the headline has room.

The startle is keyframed animation, not physics. That is deliberate and it
is what the honesty badge marks grey: the Giant Fiber decides WHEN, the
animation decides WHAT IT LOOKS LIKE.
"""

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

SCENE = Path("scenes/desk.xml")
OUT = Path("web/frames")

W, H = 1920, 1080

WORKING = {
    "coxa_T1_left": -0.88, "coxa_T1_right": -0.88,
    "coxa_abduct_T1_left": 0.22, "coxa_abduct_T1_right": -0.22,
    "femur_T1_left": -0.78, "femur_T1_right": -0.78,
    "tibia_T1_left": 0.88, "tibia_T1_right": 0.88,
    "tarsus_T1_left": 0.45, "tarsus_T1_right": 0.45,
    "head": -0.22,
    "abdomen": 0.12, "abdomen_2": 0.10, "abdomen_3": 0.08,
}

# Escape posture: legs extended in the push-off, wings thrown out, head up,
# abdomen straightened.
STARTLED = {
    "coxa_T1_left": -1.55, "coxa_T1_right": -1.55,
    "coxa_abduct_T1_left": 0.55, "coxa_abduct_T1_right": -0.55,
    "femur_T1_left": -1.60, "femur_T1_right": -1.60,
    "tibia_T1_left": 0.35, "tibia_T1_right": 0.35,
    "coxa_T2_left": -0.45, "coxa_T2_right": -0.45,
    "coxa_T3_left": -0.55, "coxa_T3_right": -0.55,
    "head": 0.30,
    "abdomen": -0.22, "abdomen_2": -0.18, "abdomen_3": -0.12,
    "wing_yaw_left": 0.9, "wing_yaw_right": -0.9,
    "wing_roll_left": -1.1, "wing_roll_right": -1.1,
    "wing_pitch_left": 0.8, "wing_pitch_right": -0.8,
}

CAM = dict(lookat=(0.20, 0.0, -0.055), dist=0.94, azim=-88, elev=-4)


def camera(lookat, dist, azim, elev):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance = dist
    c.azimuth = azim
    c.elevation = elev
    return c


def pose(model, data, angles, pos=(0.0, 0.0, 0.0), pitch=0.0):
    data.qpos[:] = model.qpos0
    data.qpos[0:3] = pos
    # rotate about y (pitch nose-up) as a quaternion
    data.qpos[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
    for name, value in angles.items():
        try:
            data.qpos[model.joint(name).qposadr[0]] = value
        except KeyError:
            pass
    mujoco.mj_forward(model, data)


def blend(a, b, t):
    keys = set(a) | set(b)
    return {k: a.get(k, 0.0) * (1 - t) + b.get(k, 0.0) * t for k in keys}


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    OUT.mkdir(parents=True, exist_ok=True)

    with mujoco.Renderer(model, height=H, width=W) as r:
        cam = camera(**CAM)

        # working
        pose(model, data, WORKING)
        r.update_scene(data, camera=cam)
        iio.imwrite(OUT / "working.jpg", r.render(), quality=92)
        print("wrote working.jpg")

        # startle: 14 frames, up and back down
        n = 14
        for i in range(n):
            t = i / (n - 1)
            lift = np.sin(min(1.0, t * 1.25) * np.pi) * 0.17
            amount = min(1.0, t * 2.2)
            pose(
                model, data,
                blend(WORKING, STARTLED, amount),
                pos=(0.0 - lift * 0.35, 0.0, lift),
                pitch=-lift * 1.9,
            )
            r.update_scene(data, camera=cam)
            iio.imwrite(OUT / f"startle_{i:02d}.jpg", r.render(), quality=88)
        print(f"wrote {n} startle frames")

    total = sum(f.stat().st_size for f in OUT.glob("*.jpg"))
    print(f"total {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
