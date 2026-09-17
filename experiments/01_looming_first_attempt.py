"""
Does the Giant Fiber respond to looming?

Drive the looming detectors (LC4, LPLC2) with an expanding-object signal and
watch DNp01. If the wiring is right, GF activity should rise with the loom and
stay quiet when nothing is approaching.

This is the falsifiable check. If it fails, the circuit is wrong.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path("data/circuit")

W = np.load(OUT / "W.npy")
circuit = json.loads((OUT / "circuit.json").read_text())
neurons = circuit["neurons"]

ids = [n["id"] for n in neurons]
roles = np.array([n["role"] for n in neurons])
looming_idx = np.where(roles == "looming")[0]
command_idx = np.where(roles == "command")[0]

# Raw synapse counts run into the thousands, so normalise each neuron's total
# incoming weight to 1. This keeps relative input strengths but puts the
# network in a sane dynamic range. It is an assumption, not a measurement.
row_sum = np.abs(W).sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
Wn = W / row_sum

DECAY = 0.8       # how much charge a neuron keeps between steps
GAIN = 2.0        # global input scaling
STEPS = 120


def run(drive):
    """drive[t] is the looming signal at timestep t. Returns GF activity."""
    x = np.zeros(W.shape[0], dtype=np.float32)
    trace = []
    for t in range(len(drive)):
        inp = np.zeros_like(x)
        inp[looming_idx] = drive[t]
        x = DECAY * x + GAIN * (Wn @ np.tanh(x)) + inp
        trace.append(float(np.tanh(x[command_idx]).mean()))
    return np.array(trace)


t = np.arange(STEPS)

# A loom: nothing, then an object expanding toward the fly, then gone.
loom = np.zeros(STEPS)
loom[40:80] = np.linspace(0, 1.0, 40) ** 2

# Control: same total energy, but flat. No approach, just ambient light.
flat = np.zeros(STEPS)
flat[40:80] = loom[40:80].mean()

gf_loom = run(loom)
gf_flat = run(flat)

print(f"neurons {W.shape[0]}  looming inputs {len(looming_idx)}  GF {len(command_idx)}")
print()
print("  t   stimulus   GF(loom)  GF(flat)")
for i in range(0, STEPS, 8):
    print(f"{i:3d}   {loom[i]:6.3f}     {gf_loom[i]:7.4f}  {gf_flat[i]:7.4f}")

print()
print(f"peak GF, looming : {gf_loom.max():.4f}")
print(f"peak GF, flat    : {gf_flat.max():.4f}")
print(f"baseline GF      : {gf_loom[:40].max():.4f}")
ratio = gf_loom.max() / max(gf_flat.max(), 1e-9)
print(f"selectivity      : {ratio:.2f}x")
