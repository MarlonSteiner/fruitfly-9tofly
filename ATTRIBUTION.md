# Attribution

The MIT licence in `LICENSE` covers the code in this repository only. The
scientific data and the body model come from other people, under their own
terms, and those terms are listed here.

## MaleCNS v1.0 connectome — CC-BY 4.0

Every neuron, synapse count and neurotransmitter prediction used here comes
from the MaleCNS v1.0 dataset: the complete central nervous system of an adult
male *Drosophila melanogaster*, about 166,700 neurons and 125 million synaptic
connections, released September 2026.

Produced by:

- **FlyEM, HHMI Janelia Research Campus**
- **Cambridge Connectomics Group**, University of Cambridge and the MRC
  Laboratory of Molecular Biology
- **Google Research**

Licensed **CC-BY**. <https://male-cns.janelia.org/>

`web/circuit.json` in this repository is a derivative of that dataset: an
809-neuron subgraph with signs applied. It is redistributed here under CC-BY
with the attribution above.

## flybody — Apache-2.0

The anatomical fruit fly body used for every render is the `flybody` MuJoCo
model, developed by **Google DeepMind** and **HHMI Janelia Research Campus**,
distributed in `mujoco_menagerie` under Apache-2.0.

<https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody>

The model is **not** vendored into this repository. `make model` clones it at
build time, so its licence and provenance stay with upstream.

Note: flybody's separately distributed flight datasets and trained policies
are GPL-3.0-or-later. This project does not use them, which is what keeps the
repository MIT-compatible.

## MuJoCo — Apache-2.0

Physics and rendering by MuJoCo, Google DeepMind. <https://mujoco.org>
