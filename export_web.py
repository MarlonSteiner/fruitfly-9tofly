"""
Pack the circuit into a compact file the browser can load.

Uses array indices instead of body IDs and pre-normalised weights, so the
page can build its matrix and start running without any preprocessing.
Neuron positions are real soma coordinates from the dataset, projected to 2D.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/circuit")
OUT = Path("web")

circuit = json.loads((SRC / "circuit.json").read_text())
manifest = json.loads((SRC / "manifest.json").read_text())
neurons = circuit["neurons"]

ann = pd.read_feather("data/raw/annotations.feather")
soma = {
    int(r.bodyId): r.somaLocation
    for r in ann.itertuples()
    if r.somaLocation is not None
}

index = {n["id"]: i for i, n in enumerate(neurons)}

# Project soma positions to 2D. x/z is roughly the frontal plane.
pos = np.zeros((len(neurons), 2), dtype=np.float64)
for i, n in enumerate(neurons):
    loc = soma.get(n["id"])
    if loc is not None and len(loc) == 3:
        pos[i] = [float(loc[0]), float(loc[2])]

valid = pos.any(axis=1)
lo, hi = pos[valid].min(axis=0), pos[valid].max(axis=0)
norm = (pos - lo) / np.maximum(hi - lo, 1e-9)
norm[~valid] = 0.5  # neurons without a soma sit in the middle

# Normalise each neuron's total incoming weight to 1, matching the Python
# model. Doing it here keeps the browser code simple.
n_count = len(neurons)
W = np.zeros((n_count, n_count), dtype=np.float32)
for e in circuit["edges"]:
    W[index[e["post"]], index[e["pre"]]] += e["synapses"] * e["sign"]

row_sum = np.abs(W).sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
Wn = W / row_sum

edges = []
rows, cols = np.nonzero(Wn)
for r, c in zip(rows.tolist(), cols.tolist()):
    edges.append([c, r, round(float(Wn[r, c]), 5)])  # [pre, post, weight]

web = {
    "neurons": [
        {
            "t": "?" if (n["type"] is None or pd.isna(n["type"])) else n["type"],
            "r": n["role"],
            "s": n["sign"],
            "m": 1 if n["nt_source"] == "ground_truth" else 0,
            "x": round(float(norm[i, 0]), 4),
            "y": round(float(norm[i, 1]), 4),
        }
        for i, n in enumerate(neurons)
    ],
    "edges": edges,
    "meta": {
        "neurons": len(neurons),
        "edges": len(edges),
        "synapses": manifest["circuit"]["synapses"],
        "measured_nt": manifest["measured"]["neurotransmitters_from_experiment"],
        "dataset": manifest["dataset"],
    },
}

OUT.mkdir(exist_ok=True)
(OUT / "circuit.json").write_text(json.dumps(web, separators=(",", ":"), allow_nan=False))

size = (OUT / "circuit.json").stat().st_size
looming = sum(1 for n in neurons if n["role"] == "looming")
command = sum(1 for n in neurons if n["role"] == "command")

print(f"neurons  {len(neurons)}  (looming {looming}, command {command})")
print(f"edges    {len(edges)}")
print(f"excit    {int((Wn > 0).sum())}   inhib {int((Wn < 0).sum())}")
print(f"wrote    web/circuit.json  {size/1024:.0f} KB")
