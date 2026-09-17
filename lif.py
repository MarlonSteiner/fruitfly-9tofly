"""
Leaky integrate-and-fire model of the Giant Fiber escape circuit.

The rate model this replaces had no threshold. `tanh` is smooth, so the
smallest input produced some output and the response saturated gradually --
measurably compressive, not decisive (see experiments/03). "Does it fire?"
had to be answered by picking a cutoff by hand, which put the decision in the
assumed column.

A LIF neuron has a real threshold. Charge leaks away between inputs, and
nothing happens at all until arriving current outruns the leak. Below that
point (the rheobase) the cell is silent no matter how long you wait. Above it,
it spikes. The decision comes from the model instead of from a constant we
chose.

Still assumed, and still marked grey in the demo: the time constants, the
threshold voltage, the synaptic gain. The connectome does not contain them.
What changes is that the *shape* of the decision is now a property of the
dynamics rather than a number we picked.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LIFParams:
    dt: float = 1.0           # ms per step (1 kHz)
    tau_m: float = 20.0       # membrane time constant, ms
    tau_s: float = 5.0        # synaptic current decay, ms
    v_th: float = 1.0         # spike threshold (normalised units)
    v_reset: float = 0.0
    refractory: float = 2.0   # ms
    gain: float = 1.0         # synaptic scaling


@dataclass
class LIFNetwork:
    """
    W is the signed connectivity, W[i, j] from j to i. Rows are normalised so
    each neuron's total incoming weight is 1, which keeps the network in a
    sane dynamic range regardless of how many partners a cell has.
    """
    W: np.ndarray
    p: LIFParams = field(default_factory=LIFParams)
    sparse: bool = False

    def __post_init__(self):
        n = self.W.shape[0]
        row = np.abs(self.W).sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        self.Wn = (self.W / row).astype(np.float32)
        if self.sparse:
            # 21k non-zeros out of 654k cells, so the sparse product is ~30x
            # less work. Matters when running hundreds of ablation trials.
            from scipy.sparse import csr_matrix
            self.Wn = csr_matrix(self.Wn)

        self.v = np.zeros(n, dtype=np.float32)
        self.i_syn = np.zeros(n, dtype=np.float32)
        self.spikes = np.zeros(n, dtype=np.float32)
        self.refrac = np.zeros(n, dtype=np.float32)

        self.m_decay = np.exp(-self.p.dt / self.p.tau_m).astype(np.float32)
        self.s_decay = np.exp(-self.p.dt / self.p.tau_s).astype(np.float32)

    def reset(self):
        self.v[:] = 0
        self.i_syn[:] = 0
        self.spikes[:] = 0
        self.refrac[:] = 0

    def step(self, drive=None):
        """Advance one dt. `drive` is external current, per neuron."""
        p = self.p

        # Synaptic current: each presynaptic spike injects its signed weight,
        # then the current decays exponentially.
        self.i_syn *= self.s_decay
        self.i_syn += p.gain * (self.Wn @ self.spikes)

        total = self.i_syn if drive is None else self.i_syn + drive

        # Membrane leaks toward rest, integrating whatever current arrives.
        self.v = self.m_decay * self.v + (1.0 - self.m_decay) * total

        # Refractory cells are clamped and cannot spike.
        held = self.refrac > 0
        self.v[held] = p.v_reset
        self.refrac[held] -= p.dt

        self.spikes = (self.v >= p.v_th).astype(np.float32)
        fired = self.spikes > 0
        self.v[fired] = p.v_reset
        self.refrac[fired] = p.refractory

        return self.spikes

    def run(self, steps, drive_fn=None, record=None):
        """
        Run for `steps`, returning spike counts for the indices in `record`
        (all neurons if None) as an array of shape (steps, len(record)).
        """
        idx = np.arange(self.W.shape[0]) if record is None else np.asarray(record)
        out = np.zeros((steps, len(idx)), dtype=np.float32)
        for t in range(steps):
            drive = None if drive_fn is None else drive_fn(t)
            out[t] = self.step(drive)[idx]
        return out


def rate_hz(spike_trace, dt):
    """Mean firing rate in Hz from a (steps, n) spike record."""
    steps = spike_trace.shape[0]
    return float(spike_trace.sum() / max(spike_trace.shape[1], 1) / (steps * dt / 1000.0))
