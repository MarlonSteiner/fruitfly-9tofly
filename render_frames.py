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

W, H = 1920, 1200   # 16:10 survives cover-cropping better than 16:9

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

# Fly centred: a centred subject survives object-fit:cover at any aspect.
CAM = dict(lookat=(0.060, 0.0, -0.045), dist=0.82, azim=-88, elev=-5)


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


CALM  = dict(rgba=(0.30, 0.56, 0.78), emission=0.45, light=(0.62, 0.86, 1.10), glow=(0.40, 0.60, 0.85))
ALERT = dict(rgba=(1.00, 0.62, 0.12), emission=1.00, light=(1.60, 0.95, 0.30), glow=(1.30, 0.70, 0.20))


def set_screen(model, look):
    """Recolour the laptop screen and the light it throws."""
    mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "screen_mat")
    model.mat_rgba[mat, :3] = look["rgba"]
    model.mat_emission[mat] = look["emission"]
    for name, key in (("spill", "light"), ("glow", "glow")):
        lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name)
        if lid >= 0:
            model.light_diffuse[lid] = look[key]


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    OUT.mkdir(parents=True, exist_ok=True)

    with mujoco.Renderer(model, height=H, width=W) as r:
        cam = camera(**CAM)

        # working
        set_screen(model, CALM)
        pose(model, data, WORKING)
        r.update_scene(data, camera=cam)
        iio.imwrite(OUT / "working.jpg", r.render(), quality=92)
        print("wrote working.jpg")

        # Startle: takeoff, airborne, land, settle back to the working pose.
        #
        # The first version ran the pose blend to fully-startled on the final
        # frame and then cut straight back to working.jpg, so every cycle
        # ended in a hard snap. It read as a glitch because it was one.
        n = 22
        for i in range(n):
            t = i / (n - 1)
            # screen flares hard at the incident, then settles back
            flare = min(1.0, t / 0.18) * (1.0 - max(0.0, (t - 0.45) / 0.55) * 0.85)
            set_screen(model, {
                k: tuple(np.array(CALM[k]) * (1 - flare) + np.array(ALERT[k]) * flare)
                if isinstance(CALM[k], tuple) else CALM[k] * (1 - flare) + ALERT[k] * flare
                for k in CALM
            })
            # Arc: leaves the chair, peaks, comes back down by 78% through.
            lift = np.sin(min(1.0, t / 0.78) * np.pi) * 0.105

            # Posture: snaps open on takeoff, holds while airborne, then eases
            # back to the working pose so the last frame matches working.jpg.
            if t < 0.18:
                amount = t / 0.18
            elif t < 0.55:
                amount = 1.0
            else:
                ease = (t - 0.55) / 0.45
                amount = 1.0 - (ease * ease * (3 - 2 * ease))   # smoothstep
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
