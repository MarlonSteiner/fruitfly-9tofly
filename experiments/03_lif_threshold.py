"""
Does the LIF model make a decision where the rate model did not?

The rate model's input/output curve rose at the very smallest input and
saturated smoothly -- a compressive filter, not a decision. A real threshold
should look different: silent, silent, silent, then firing.

This measures the Giant Fiber's firing rate against input strength and prints
both models side by side. Run from the repository root.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lif import LIFNetwork, LIFParams  # noqa: E402

OUT = Path("data/circuit")
W = np.load(OUT / "W.npy")
circuit = json.loads((OUT / "circuit.json").read_text())
roles = np.array([n["role"] for n in circuit["neurons"]])
loom = np.where(roles == "looming")[0]
gf = np.where(roles == "command")[0]

DT = 1.0
SECONDS = 1.0
STEPS = int(SECONDS * 1000 / DT)


def lif_curve(amps, gain=10.0):
    net = LIFNetwork(W, LIFParams(dt=DT, gain=gain))
    out = []
    for a in amps:
        net.reset()
        drive = np.zeros(W.shape[0], dtype=np.float32)
        drive[loom] = a
        rec = net.run(STEPS, drive_fn=lambda t: drive, record=gf)
        out.append(rec.sum() / len(gf) / SECONDS)   # Hz per Giant Fiber
    return np.array(out)


def rate_curve(amps, decay=0.70, gain=0.25):
    row = np.abs(W).sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    Wn = W / row
    out = []
    for a in amps:
        x = np.zeros(W.shape[0], dtype=np.float32)
        peak = 0.0
        for _ in range(STEPS):
            inp = np.zeros_like(x)
            inp[loom] = a
            x = decay * x + gain * (Wn @ np.tanh(x)) + inp
            peak = max(peak, float(np.tanh(x[gf]).mean()))
        out.append(peak)
    return np.array(out)


# The two models take input on different scales, so each is swept across its
# own dynamic range and compared on SHAPE, which is scale-free.
lif_amps = np.array([0.0, 0.6, 0.9, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.4, 1.7, 2.1, 2.6, 3.2])
rate_amps = np.array([0.0, 0.005, 0.01, 0.02, 0.04, 0.07, 0.11, 0.16, 0.24, 0.35, 0.5, 0.7, 1.0, 1.4])

print("Giant Fiber response to increasing drive on the looming population")
print(f"({SECONDS:.0f} s per point, dt = {DT:g} ms)\n")

lif = lif_curve(lif_amps)
rate = rate_curve(rate_amps)

print(f"{'LIF drive':>10} {'GF (Hz)':>9}   |{'rate drive':>11} {'GF out':>9}")
print("-" * 46)
for la, lv, ra, rv in zip(lif_amps, lif, rate_amps, rate):
    print(f"{la:10.2f} {lv:9.1f}   |{ra:11.3f} {rv:9.4f}")


def dynamic_range(amps, vals):
    """
    Input ratio needed to go from 10% to 90% of maximum output.

    Scale-free, so it compares models with different input units. A hard
    threshold needs almost no change in input to cross; a smooth filter needs
    a large one.
    """
    v = np.asarray(vals, dtype=float)
    if v.max() <= 0:
        return None
    frac = v / v.max()
    lo = np.interp(0.1, frac, amps)
    hi = np.interp(0.9, frac, amps)
    return hi / lo if lo > 0 else None


lif_dr = dynamic_range(lif_amps, lif)
rate_dr = dynamic_range(rate_amps, rate)

print()
print(f"LIF        : silent below {lif_amps[np.argmax(lif > 0)]:.2f}, "
      f"10%->90% output needs {lif_dr:.2f}x more input")
print(f"rate model : responds from the smallest input, "
      f"10%->90% output needs {rate_dr:.1f}x more input")
print()
print(f"The LIF decision is {rate_dr/lif_dr:.0f}x sharper.")
print()
print("A decision looks like silence, then a sharp rise. A filter responds to")
print("everything and grows smoothly. The threshold is now a property of the")
print("membrane dynamics, not a cutoff we picked.")
