"""
Looming test, second attempt.

Fixes two bugs from test_looming.py:

1. Spatial structure. Each looming neuron now gets a receptive-field position
   so an expanding object sweeps across the population instead of hitting all
   of them at once. Positions come from soma location, which is a PROXY for
   receptive field, not a measurement of it -- LC dendrites tile the lobula
   retinotopically but the somata sit in a rind. If selectivity appears, the
   proxy is good enough. If it does not, we need real dendrite positions.

2. Stability. Loop gain is now below 1 so activity decays when the stimulus
   stops instead of latching on.

Controls matter more than the main condition. A result is only interesting if
looming beats flat AND receding.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path("data/circuit")

W = np.load(OUT / "W.npy")
circuit = json.loads((OUT / "circuit.json").read_text())
neurons = circuit["neurons"]

roles = np.array([n["role"] for n in neurons])
looming_idx = np.where(roles == "looming")[0]
command_idx = np.where(roles == "command")[0]

# ---- receptive field positions (proxy: soma location) ----
import pandas as pd

ann = pd.read_feather("data/raw/annotations.feather")
soma = {int(r.bodyId): r.somaLocation for r in ann.itertuples() if r.somaLocation is not None}

pos = np.zeros((len(neurons), 2), dtype=np.float32)
for i, n in enumerate(neurons):
    loc = soma.get(n["id"])
    if loc is not None and len(loc) == 3:
        pos[i] = [loc[0], loc[2]]  # x and z, roughly the eye plane

lp = pos[looming_idx]
lo, hi = lp.min(axis=0), lp.max(axis=0)
rf = (lp - lo) / np.maximum(hi - lo, 1e-9)      # normalise to unit square
rf_centre = rf.mean(axis=0)

# ---- dynamics ----
DECAY = 0.7
GAIN = 0.25       # loop gain below 1 so the network settles
STEPS = 140

row_sum = np.abs(W).sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
Wn = W / row_sum


def stimulus(radius_over_time, width=0.12):
    """
    An object of a given angular radius centred in the visual field.
    A looming neuron responds to the expanding EDGE passing over its RF.
    """
    drive = np.zeros((len(radius_over_time), len(looming_idx)), dtype=np.float32)
    dist = np.linalg.norm(rf - rf_centre, axis=1)
    for t, r in enumerate(radius_over_time):
        if r <= 0:
            continue
        drive[t] = np.exp(-((dist - r) ** 2) / (2 * width ** 2))
    return drive


def run(drive):
    x = np.zeros(W.shape[0], dtype=np.float32)
    trace = []
    for t in range(len(drive)):
        inp = np.zeros_like(x)
        inp[looming_idx] = drive[t]
        x = DECAY * x + GAIN * (Wn @ np.tanh(x)) + inp
        trace.append(float(np.tanh(x[command_idx]).mean()))
    return np.array(trace)


t = np.arange(STEPS)
zero = np.zeros(STEPS)

# LOOM: object grows -- edge sweeps outward, accelerating.
loom_r = zero.copy()
loom_r[40:100] = np.linspace(0.0, 0.9, 60) ** 0.6

# RECEDE: same edge, same energy, swept the other way. The key control.
recede_r = zero.copy()
recede_r[40:100] = loom_r[40:100][::-1]

# FLAT: constant size, no motion at all.
flat_r = zero.copy()
flat_r[40:100] = 0.45

conditions = {
    "loom": loom_r,
    "recede": recede_r,
    "flat": flat_r,
}

results = {}
for name, r in conditions.items():
    results[name] = run(stimulus(r))

print(f"neurons {W.shape[0]}  looming {len(looming_idx)}  GF {len(command_idx)}")
print(f"RF spread: x {rf[:,0].min():.2f}-{rf[:,0].max():.2f}  y {rf[:,1].min():.2f}-{rf[:,1].max():.2f}")
print()
print("  t    loom   recede    flat")
for i in range(0, STEPS, 10):
    print(f"{i:3d}  {results['loom'][i]:6.4f}  {results['recede'][i]:6.4f}  {results['flat'][i]:6.4f}")

print()
for name, tr in results.items():
    print(f"peak {name:7s}: {tr.max():.4f}   final: {tr[-1]:.4f}")

print()
print(f"loom vs recede : {results['loom'].max() / max(results['recede'].max(), 1e-9):.2f}x")
print(f"loom vs flat   : {results['loom'].max() / max(results['flat'].max(), 1e-9):.2f}x")
