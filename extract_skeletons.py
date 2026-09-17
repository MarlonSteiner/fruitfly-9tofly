"""
Pull real neuron skeletons for the brain panel.

The dots in the first version were soma positions -- one point per neuron,
which looks like scatter, not like a brain. These are the actual traced
morphologies: every branch of every dendrite and axon, reconstructed from
electron microscopy.

Source: gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/, public over
HTTPS, no account. Neuroglancer precomputed format:

    uint32   vertex count
    uint32   edge count
    float32  vertices[count][3]
    uint32   edges[count][2]

Output is web/skeletons.json: 2D line segments, normalised, tagged by role so
the panel can light them by activity.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = ("https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
        "skeletons-malecns/skeletons-precomputed")
CACHE = Path("data/skeletons")
OUT = Path("web/skeletons.json")

# How many of each to draw. More looks denser but costs bytes.
SAMPLE = {"DNp01": 2, "LC4": 20, "LPLC2": 20, "JO-B1_a": 10, "JO-B1_c": 6}
INHIBITORY = 16          # strongest GABA/glutamate inputs to the Giant Fiber
MAX_SEGMENTS = 260       # per neuron, after decimation


def fetch(body_id):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{body_id}.bin"
    if not path.exists():
        r = requests.get(f"{BASE}/{body_id}", timeout=60)
        if r.status_code != 200 or len(r.content) < 16:
            return None
        path.write_bytes(r.content)
    return path.read_bytes()


def parse(raw):
    nv, ne = np.frombuffer(raw[:8], dtype="<u4")
    if nv == 0 or ne == 0:
        return None, None
    v = np.frombuffer(raw[8:8 + nv * 12], dtype="<f4").reshape(nv, 3)
    off = 8 + nv * 12
    e = np.frombuffer(raw[off:off + ne * 8], dtype="<u4").reshape(ne, 2)
    return v, e


def segments(v, e, budget=MAX_SEGMENTS):
    """Decimate the edge list down to a drawable number of line segments."""
    if len(e) > budget:
        stride = int(np.ceil(len(e) / budget))
        e = e[::stride]
    a, b = v[e[:, 0]], v[e[:, 1]]
    return np.stack([a, b], axis=1)     # (n, 2, 3)


def pick_neurons():
    ann = pd.read_feather("data/raw/annotations.feather")
    nt = pd.read_feather("data/raw/neurotransmitters.feather")
    w = pd.read_feather("data/raw/weights.feather")

    chosen = []
    for cell_type, n in SAMPLE.items():
        ids = ann[ann["type"] == cell_type]["bodyId"].tolist()[:n]
        role = ("command" if cell_type == "DNp01"
                else "auditory" if cell_type.startswith("JO")
                else "looming")
        chosen += [(int(i), cell_type, role) for i in ids]

    # strongest inhibitory inputs onto the Giant Fiber
    gf_ids = set(ann[ann["type"] == "DNp01"]["bodyId"])
    onto_gf = w[w["body_post"].isin(gf_ids) & (w["weight"] >= 5)]
    ntc = nt.drop_duplicates("cell_type").set_index("cell_type")["consensus_nt"]
    strength = onto_gf.groupby("body_pre")["weight"].sum().sort_values(ascending=False)
    type_of = ann.set_index("bodyId")["type"].to_dict()

    taken = 0
    for body_id, _ in strength.items():
        if taken >= INHIBITORY:
            break
        t = type_of.get(body_id)
        if t is None or pd.isna(t):
            continue
        if ntc.get(t) in ("gaba", "glutamate"):
            chosen.append((int(body_id), t, "inhibitory"))
            taken += 1

    return chosen


def main():
    chosen = pick_neurons()
    print(f"selected {len(chosen)} neurons")

    with ThreadPoolExecutor(max_workers=8) as pool:
        blobs = list(pool.map(lambda c: fetch(c[0]), chosen))

    cells, all_pts = [], []
    for (body_id, cell_type, role), raw in zip(chosen, blobs):
        if raw is None:
            print(f"  ! no skeleton for {body_id} ({cell_type})")
            continue
        v, e = parse(raw)
        if v is None:
            continue
        seg = segments(v, e)
        cells.append({"id": body_id, "type": cell_type, "role": role, "seg": seg})
        all_pts.append(seg.reshape(-1, 3))

    pts = np.concatenate(all_pts)
    lo, hi = pts.min(0), pts.max(0)
    print(f"bbox lo {lo.round(0)}  hi {hi.round(0)}  span {(hi-lo).round(0)}")

    # Frontal view: x is left-right, y is dorsal-ventral. This is the view
    # that reads as a brain, with the optic lobes flanking the centre.
    # Both axes share one scale so the shape is not distorted.
    AX, AY = 0, 1
    span = hi - lo
    scale = max(span[AX], span[AY])

    def norm(a, i):
        return (a[:, :, i] - lo[i]) / scale

    out = []
    for c in cells:
        sx = norm(c["seg"], AX)
        sy = norm(c["seg"], AY)
        flat = np.stack([sx, sy], axis=2).reshape(-1, 4)
        out.append({
            "r": c["role"],
            "t": c["type"],
            "s": [[round(float(x), 4) for x in row] for row in flat],
        })

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
    "extent": [round(float(span[AX] / scale), 4), round(float(span[AY] / scale), 4)],
    "cells": out,
}, separators=(",", ":")))
    total = sum(len(c["s"]) for c in out)
    print(f"{len(out)} neurons, {total} segments, {OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
