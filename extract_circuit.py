"""
Extract the looming-escape circuit from the MaleCNS v1.0 connectome.

The circuit:  LC4 + LPLC2  ->  DNp01 (Giant Fiber)

LC4 and LPLC2 are looming-sensitive visual projection neurons. DNp01 is the
Giant Fiber, the descending command neuron that triggers the escape jump.
This is the best-characterised escape pathway in the fly.

Output is a small JSON file the browser demo loads directly, plus a manifest
recording every assumption we made, so the UI can show what is measured and
what is assumed.

Data: MaleCNS v1.0, CC-BY. FlyEM (HHMI Janelia), Cambridge Connectomics
Group, and Google Research.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/circuit")

# We do not hand-pick the circuit. We seed it with the Giant Fiber and let
# the data decide the rest: every neuron with a significant direct connection
# onto it is recruited. Choosing type names by hand quietly excluded all the
# inhibition, because LC4, LPLC2 and DNp01 all happen to be cholinergic.
SEED_TYPE = "DNp01"          # Giant Fiber, the escape command neuron
LOOMING = ["LC4", "LPLC2"]   # looming detectors, tagged for the UI

# Minimum synapse count for an edge to be kept. Connectomes are full of
# 1-2 synapse connections that are mostly reconstruction noise rather than
# real biology; ~5 is the conventional cutoff.
MIN_SYNAPSES = 5

# Neurotransmitter -> sign. In Drosophila acetylcholine is excitatory,
# GABA is inhibitory, and glutamate is usually inhibitory via the GluCl-alpha
# receptor (though this is context-dependent and the weakest link here).
# Neuromodulators get 0 because they do not act as simple +/- drive.
NT_SIGN = {
    "acetylcholine": +1,
    "gaba": -1,
    "glutamate": -1,
    "dopamine": 0,
    "serotonin": 0,
    "octopamine": 0,
    "histamine": -1,
    "unknown": 0,
}


def recruit(edges, seed_ids):
    """The Giant Fiber plus every neuron that significantly synapses onto it."""
    upstream = edges[
        edges["body_post"].isin(seed_ids) & (edges["weight"] >= MIN_SYNAPSES)
    ]["body_pre"]
    return set(upstream) | set(seed_ids)


def load_neurons(body_ids):
    """Attach a transmitter and a sign to each neuron in the circuit."""
    ann = pd.read_feather(RAW / "annotations.feather")
    nt = pd.read_feather(RAW / "neurotransmitters.feather")

    neurons = ann[ann["bodyId"].isin(body_ids)][
        ["bodyId", "type", "somaSide", "status"]
    ].copy()

    nt_cols = ["body", "consensus_nt", "ground_truth", "predicted_nt_confidence"]
    neurons = neurons.merge(
        nt[nt_cols], left_on="bodyId", right_on="body", how="left"
    ).drop(columns=["body"])

    # Prefer an experimentally measured transmitter over a predicted one.
    neurons["nt"] = neurons["ground_truth"].fillna(neurons["consensus_nt"])
    neurons["nt_source"] = np.where(
        neurons["ground_truth"].notna(), "ground_truth", "predicted"
    )
    neurons["sign"] = neurons["nt"].map(NT_SIGN).fillna(0).astype(int)

    return neurons


def load_edges(edges, body_ids):
    """Keep the edges that live entirely inside the circuit."""
    inside = edges["body_pre"].isin(body_ids) & edges["body_post"].isin(body_ids)
    return edges[inside & (edges["weight"] >= MIN_SYNAPSES)].copy()


def role_of(neuron_type):
    if neuron_type == SEED_TYPE:
        return "command"
    if neuron_type in LOOMING:
        return "looming"
    return "input"


def build_matrix(neurons, edges):
    """
    Turn the edge list into a dense signed matrix W.

    W[i, j] is the signed strength from neuron j to neuron i, so one
    matrix-vector product advances the whole network by a timestep.
    Dense is correct at this scale: a few hundred neurons is well under a
    megabyte. Sparse would only pay off in the tens of thousands.
    """
    ids = neurons["bodyId"].tolist()
    index = {body_id: i for i, body_id in enumerate(ids)}
    sign = dict(zip(neurons["bodyId"], neurons["sign"]))

    n = len(ids)
    W = np.zeros((n, n), dtype=np.float32)

    for pre, post, weight in zip(edges["body_pre"], edges["body_post"], edges["weight"]):
        W[index[post], index[pre]] += weight * sign[pre]

    return W, ids, index


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    all_edges = pd.read_feather(RAW / "weights.feather")
    ann = pd.read_feather(RAW / "annotations.feather")

    seed_ids = set(ann[ann["type"] == SEED_TYPE]["bodyId"])
    body_ids = recruit(all_edges, seed_ids)

    neurons = load_neurons(body_ids)
    edges = load_edges(all_edges, set(neurons["bodyId"]))
    W, ids, index = build_matrix(neurons, edges)

    measured = int((neurons["nt_source"] == "ground_truth").sum())

    circuit = {
        "neurons": [
            {
                "id": int(r.bodyId),
                "type": None if pd.isna(r.type) else r.type,
                "side": None if pd.isna(r.somaSide) else r.somaSide,
                "nt": None if pd.isna(r.nt) else r.nt,
                "nt_source": r.nt_source,
                "sign": int(r.sign),
                "role": role_of(r.type),
            }
            for r in neurons.itertuples()
        ],
        "edges": [
            {
                "pre": int(pre),
                "post": int(post),
                "synapses": int(weight),
                "sign": int(neurons.set_index("bodyId").at[pre, "sign"]),
            }
            for pre, post, weight in zip(
                edges["body_pre"], edges["body_post"], edges["weight"]
            )
        ],
    }

    manifest = {
        "dataset": "MaleCNS v1.0 (male-cns:v1.0)",
        "license": "CC-BY",
        "attribution": [
            "FlyEM, HHMI Janelia Research Campus",
            "Cambridge Connectomics Group",
            "Google Research",
        ],
        "circuit": {
            "definition": (
                f"{SEED_TYPE} (Giant Fiber) plus every neuron with >= "
                f"{MIN_SYNAPSES} synapses onto it"
            ),
            "seed_type": SEED_TYPE,
            "looming_types": LOOMING,
            "neurons": len(neurons),
            "edges": len(edges),
            "synapses": int(edges["weight"].sum()),
        },
        "measured": {
            "connectivity": "synapse counts, directly from the dataset",
            "neurotransmitters_from_experiment": f"{measured}/{len(neurons)}",
        },
        "assumed": {
            "nt_to_sign": NT_SIGN,
            "glutamate_caveat": (
                "Glutamate is treated as inhibitory (GluCl-alpha). This is the "
                "usual case in Drosophila but is context-dependent."
            ),
            "min_synapses": MIN_SYNAPSES,
            "edge_weight_as_strength": (
                "Synapse count is used as a proxy for connection strength. The "
                "connectome does not measure physiological strength."
            ),
            "giant_fiber_caveat": (
                "DNp01 has no ground-truth transmitter and low prediction "
                "confidence (~0.5). It signals largely through electrical gap "
                "junctions, which a chemical-synapse connectome does not "
                "capture. Its downstream influence is therefore "
                "under-represented here."
            ),
        },
        "not_modelled": [
            "synaptic delays",
            "per-neuron time constants",
            "plasticity and learning",
            "neuromodulation",
            "gap junctions",
        ],
    }

    (OUT / "circuit.json").write_text(json.dumps(circuit, allow_nan=False))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    np.save(OUT / "W.npy", W)

    print(f"neurons        {len(neurons)}")
    print(f"edges          {len(edges)} (>= {MIN_SYNAPSES} synapses)")
    print(f"synapses       {int(edges['weight'].sum())}")
    print(f"NT measured    {measured}/{len(neurons)}")
    print(f"matrix         {W.shape}  {W.nbytes / 1024:.0f} KB")
    print(f"excitatory     {int((W > 0).sum())} edges")
    print(f"inhibitory     {int((W < 0).sum())} edges")
    print(f"\nwrote {OUT}/circuit.json, manifest.json, W.npy")


if __name__ == "__main__":
    main()
