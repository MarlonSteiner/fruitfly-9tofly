"""
Is the wiring doing anything?

This is the experiment the viral fly projects do not run. The connectome is
a real graph, but a sufficiently flexible readout can make almost any
recurrent network look purposeful -- someone drove a fly body with a *worm*
connectome and it worked fine. So "we used the real connectome" is not
evidence that the real connectome mattered.

The test: break the wiring in specific ways, keep everything else identical,
and measure whether the circuit still does its job.

Job = telling an incident from ambient noise. The metric is detection
threshold: the lowest drive at which the Giant Fiber fires in at least half of
trials. Lower means more sensitive.

(d-prime was the obvious choice and it is useless here. Ambient produces
exactly zero spikes, so the variance is ~0 and d-prime explodes past 90 for
every variant. Perfect separation measures nothing when the task is that easy.)

Each control isolates one property:

  weights_shuffled  magnitudes permuted among the same connections
                    -> does it matter HOW STRONG each connection is?
  signs_shuffled    which neurons are inhibitory is permuted, Dale's law kept
                    -> does it matter WHICH cells inhibit?
  inputs_rewired    every neuron keeps its in-degree and the same incoming
                    strengths, but from random sources
                    -> does it matter WHO connects to WHOM?
  fully_random      same edge count and weight distribution, placed at random
                    -> does any of the structure matter?

If the real connectome does not beat these, the honest conclusion is that the
wiring is not doing the work, and this README should say so.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lif import LIFNetwork, LIFParams  # noqa: E402

OUT = Path("data/circuit")
circuit = json.loads((OUT / "circuit.json").read_text())
neurons = circuit["neurons"]
N = len(neurons)

index = {n["id"]: i for i, n in enumerate(neurons)}
roles = np.array([n["role"] for n in neurons])
signs = np.array([n["sign"] for n in neurons], dtype=np.float32)
loom = np.where(roles == "looming")[0]
gf = np.where(roles == "command")[0]

E = circuit["edges"]
e_pre = np.array([index[e["pre"]] for e in E])
e_post = np.array([index[e["post"]] for e in E])
e_mag = np.array([e["synapses"] for e in E], dtype=np.float32)

# ---- experiment settings ----
TRIALS = 24
STEPS = 150          # ms per trial
AMBIENT = 0.88
INCIDENT = 0.88 + 0.55
SEEDS = 3            # repeats for each randomised control


def build(pre, post, mag, sgn):
    W = np.zeros((N, N), dtype=np.float32)
    np.add.at(W, (post, pre), mag * sgn[pre])
    return W


def variant(name, rng):
    """Return the W matrix for a named control."""
    if name == "real":
        return build(e_pre, e_post, e_mag, signs)

    if name == "weights_shuffled":
        return build(e_pre, e_post, rng.permutation(e_mag), signs)

    if name == "signs_shuffled":
        # Permute which neurons are inhibitory. Dale's law is preserved: a
        # neuron still has one sign on all of its outputs.
        return build(e_pre, e_post, e_mag, rng.permutation(signs))

    if name == "inputs_rewired":
        # Each neuron keeps its in-degree and incoming strengths; the sources
        # are randomised.
        return build(rng.permutation(e_pre), e_post, e_mag, signs)

    if name == "fully_random":
        return build(rng.integers(0, N, len(e_pre)),
                     rng.integers(0, N, len(e_pre)),
                     rng.permutation(e_mag), signs)

    raise ValueError(name)


def spike_count(net, amp, rng):
    """Giant Fiber spikes in one trial."""
    net.reset()
    total = 0.0
    drive = np.zeros(N, dtype=np.float32)
    for _ in range(STEPS):
        drive[:] = 0
        drive[loom] = amp * (0.75 + rng.random(len(loom)) * 0.5)
        total += net.step(drive)[gf].sum()
    return total


def hit_rate(net, amp, rng, trials=TRIALS):
    """Fraction of trials in which the Giant Fiber fires at all."""
    return float(np.mean([spike_count(net, amp, rng) > 0 for _ in range(trials)]))


SWEEP = np.arange(0.80, 1.61, 0.04)


def detection_threshold(net, rng):
    """
    Lowest drive at which the Giant Fiber fires in at least half of trials.

    This replaces d-prime. With ambient producing exactly zero spikes the
    variance is ~0 and d-prime explodes to meaningless values -- every variant
    separates perfectly, which measures nothing. Detection threshold asks the
    harder and more useful question: how much signal does this network need
    before it responds? A sensitive alarm needs less.
    """
    for amp in SWEEP:
        if hit_rate(net, amp, rng, trials=12) >= 0.5:
            return float(amp)
    return float("nan")


CONTROLS = ["real", "weights_shuffled", "signs_shuffled",
            "inputs_rewired", "fully_random"]

print("Does the specific wiring matter?")
print(f"{STEPS} ms trials. Detection threshold = lowest drive at which the")
print(f"Giant Fiber fires in >=50% of trials. Lower is more sensitive.")
print(f"False alarms measured at ambient drive {AMBIENT}.\n")
print(f"{'network':>18} {'threshold':>10} {'false alarm':>12} {'spikes @1.43':>13}")
print("-" * 57)

results = {}
for name in CONTROLS:
    reps = 1 if name == "real" else SEEDS
    ths, fas, cts = [], [], []
    for s in range(reps):
        rng = np.random.default_rng(1000 + s)
        W = variant(name, rng)
        net = LIFNetwork(W, LIFParams(gain=10.0), sparse=True)
        r = np.random.default_rng(7 + s)
        ths.append(detection_threshold(net, r))
        fas.append(hit_rate(net, AMBIENT, r, trials=16))
        cts.append(np.mean([spike_count(net, INCIDENT, r) for _ in range(8)]))
    th = np.nanmean(ths)
    results[name] = th
    spread = f" ±{np.nanstd(ths):.2f}" if reps > 1 else ""
    shown = "never" if np.isnan(th) else f"{th:.2f}"
    print(f"{name:>18} {shown:>10}{spread:>7} {np.mean(fas):11.0%} {np.mean(cts):13.1f}")

print()
real = results["real"]
controls = {k: v for k, v in results.items() if k != "real"}
best = min((v for v in controls.values() if not np.isnan(v)), default=float("nan"))

print(f"real connectome threshold : {real:.2f}")
print(f"best control threshold    : {best:.2f}")
print()

if np.isnan(best) or real < best * 0.9:
    print("The real wiring detects signal that its randomised versions cannot.")
elif abs(real - best) / real < 0.1:
    print("The real wiring performs about the same as randomised controls on")
    print("this task. Detecting a broad increase in drive does not require")
    print("specific connectivity -- any network that sums inputs and thresholds")
    print("will do it. This is a real limitation, not a tuning problem, and it")
    print("belongs in the README.")
else:
    print("The real wiring is ahead, but not decisively. Treat claims that the")
    print("connectome is doing the work with caution.")

broken = [k for k, v in controls.items() if np.isnan(v)]
if broken:
    print()
    print("Controls that destroyed the circuit entirely: " + ", ".join(broken))
    print("Those properties are load-bearing.")
