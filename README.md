# 9 to Fly

A fruit fly works a night shift. When something spikes, its Giant Fiber fires
and it leaves the chair.

![The fly at its desk](web/frames/working.jpg)

The fly is the real anatomical [flybody](https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody)
model from Google DeepMind and HHMI Janelia, rendered in MuJoCo. The brain is
809 real neurons from [MaleCNS v1.0](https://male-cns.janelia.org/), wired the
way they are in an actual fly. The startle happens when excitation genuinely
outruns inhibition in that circuit.

## Why the Giant Fiber

`DNp01` — the Giant Fiber — is the descending command neuron that triggers the
escape jump. One of the best-characterised neurons in neuroscience, and the
reason you miss when you swat.

The circuit is not hand-picked. It is seeded with DNp01 and then every neuron
with at least 5 synapses onto it is recruited, which gives 809 neurons and
21,086 connections. The two strongest inputs turn out to be **LC4** and
**LPLC2**, the looming detectors — 30% of all synaptic input to the Giant
Fiber, from two cell types out of hundreds. The literature predicts that; the
data agrees without being asked.

## What is measured and what is assumed

This is the part most projects in this genre skip.

| | Source |
|---|---|
| Which neurons connect to which | **Measured.** MaleCNS v1.0 |
| Synapse counts per connection | **Measured.** |
| Excitatory vs inhibitory | **Measured for 377/809 neurons** from experimental ground-truth neurotransmitters; predicted for the rest |
| Connection *strength* | **Assumed.** Synapse count is a proxy; the connectome does not measure physiological strength |
| Time constants, delays, plasticity, neuromodulation | **Not modelled.** Not in the dataset |
| Gap junctions | **Not modelled.** The connectome maps chemical synapses only |
| Firing threshold, decay, gain | **Assumed.** Hand-tuned |
| The fly's movement | **Assumed.** Keyframed animation, not physics |

Two caveats worth stating plainly:

**The Giant Fiber signals largely through electrical gap junctions.** A
chemical-synapse connectome under-represents its real influence. It is also
the one neuron here with no ground-truth transmitter (prediction confidence
≈0.5) — because chemical transmission isn't its main mode.

**Glutamate is treated as inhibitory** via GluClα. Usually right in
*Drosophila*, but context-dependent. It is the weakest link in the sign
assignment.

`data/circuit/manifest.json` records every one of these choices in machine-
readable form, and the demo shows a live badge separating the two columns.

## What we tested and what failed

The circuit **cannot** tell a looming object from a receding one.

`experiments/02_looming_with_receptive_fields.py` drives the looming
population with an expanding edge, then with the same edge reversed, then with
a static one. Peak Giant Fiber response:

```
loom     0.2596
recede   0.2598      loom vs recede: 1.00x
flat     0.2601      loom vs flat:   1.00x
```

No selectivity at all. A ramp test shows the response is compressive, not
thresholded — it rises at the smallest input and saturates smoothly, where a
decision would stay flat and then snap.

Both are real findings, not bugs:

1. **Looming detection is upstream of this slice.** LC4 and LPLC2 are the
   *output* of the computation. It happens in the optic lobe — the 89,403
   `ol_intrinsic` neurons excluded here. A linear sum discards temporal order,
   which is exactly what `loom vs recede = 1.00x` measures.
2. **A rate model cannot make a decision.** The real threshold is spiking
   biophysics this model does not have.

Had we only run the looming condition, Giant Fiber activity would have climbed
from 0 to 0.99 right as the object approached and it would have looked like a
success. The control is the only reason we know it was meaningless.

So the honest claim is narrower than "the fly sees and jumps":

> The connectome does the **integration and gating** — weighing excitation
> against inhibition to decide whether to fire. It does **not** do the
> perception.

Which is why the fly's job here is monitoring. A threshold detector is used as
a threshold detector.

## Pipeline

```
MaleCNS v1.0 (Feather, ~550 MB)
  -> extract_circuit.py   seed DNp01, recruit inputs, sign edges from
                          neurotransmitters, threshold at 5 synapses
  -> export_web.py        pack to 412 KB with real soma coordinates
  -> web/                 browser runs the circuit live; no API key, no account
```

Per frame the browser walks 21,086 edges rather than an 809×809 matrix.

```
flybody (MuJoCo, Apache-2.0)
  -> scenes/desk.xml      desk, chair, laptop proportioned to the fly
  -> render_frames.py     working pose + 14-frame startle
```

## Run it

```bash
make venv     # dependencies
make data     # MaleCNS downloads, ~550 MB, public, no account needed
make model    # flybody model, ~140 MB
make circuit  # -> web/circuit.json
make frames   # -> web/frames/
make web      # http://localhost:8777
```

`web/` is committed, so `make web` alone is enough to see the demo. The rest
is only needed to rebuild from source.

## Status

Working: circuit extraction with data-derived signs, live browser simulation,
MuJoCo renders, startle driven by the real signal.

Next: leaky integrate-and-fire with a real spiking threshold, to move the
decision from the assumed column to the measured one. Then control ablations —
shuffled connectome, random network with matched statistics — so the question
"is the wiring doing anything?" has a number rather than an opinion.

## Credit

MaleCNS v1.0 is CC-BY: FlyEM (HHMI Janelia), the Cambridge Connectomics Group,
and Google Research. flybody and MuJoCo are Apache-2.0, Google DeepMind and
HHMI Janelia. Full detail in [ATTRIBUTION.md](ATTRIBUTION.md).

Code is MIT.
