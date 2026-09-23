"""Quantum and classical Fisher information of sensor-array probes, computed with squint.

Each of the n qubits picks up its own phase phi_i from the signal. Putting the
n phase gates in one squint Block gives the full n x n Fisher information
matrix over the local phases in one call. For a signal theta that reaches
sensor i with coupling g_i (phi_i = g_i * theta), the chain rule then gives

    F_theta = g^T F g.

Requires squint (Python 3.11+): pip install -e ".[squint]"
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
from squint import Circuit
from squint.backends.tensornetwork.simulator import Simulator
from squint.interface.base import Block, Wire
from squint.interface.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate, x
from squint.math.information_matrices import (
    classical_fisher_information_matrix,
    quantum_fisher_information_matrix,
)
from squint.utils import partition_op

PROBES = ("product", "ghz")


def sensor_circuit(
    n: int, probe: str, flips: Sequence[int] = (), phase: float = 0.1
) -> Circuit:
    """Probe preparation, one RZ per sensor (Block "phases"), then an X-basis readout.

    probe="product" prepares |+>^n; probe="ghz" prepares (|0...0> + |1...1>)/sqrt(2).
    `flips` applies X to the listed qubits after preparation, which turns the GHZ
    state into (|s> + |~s>)/sqrt(2) and reverses the sign of those sensors' phases.
    """
    if probe not in PROBES:
        raise ValueError(f"probe must be one of {PROBES}")
    wires = [Wire(dim=2, idx=i) for i in range(n)]
    circuit = Circuit()
    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))
    if probe == "ghz":
        circuit.add(HGate(wires=(wires[0],)))
        for a, b in pairwise(wires):
            circuit.add(Conditional(ufunc=x, wires=(a, b)))
    else:
        for w in wires:
            circuit.add(HGate(wires=(w,)))
    for i in flips:
        circuit.add(XGate(wires=(wires[i],)))

    phases = Block()
    for w in wires:
        phases.add(RZGate(wires=(w,), phi=phase))
    circuit.add(phases, "phases")

    for w in wires:
        circuit.add(HGate(wires=(w,)))
    return circuit


def fisher_matrices(
    n: int, probe: str, flips: Sequence[int] = (), phase: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """(QFIM, CFIM of the X-basis readout) over the n local phases."""
    params, static = partition_op(sensor_circuit(n, probe, flips, phase), "phases")
    sim = Simulator(static=static, params=params)

    def probabilities(p):
        return jnp.abs(sim.forward(p)) ** 2

    qfim = quantum_fisher_information_matrix(sim.forward, sim.grad, params)
    cfim = classical_fisher_information_matrix(probabilities, jax.jacfwd(probabilities), params)
    return np.asarray(qfim), np.asarray(cfim)


def signal_fisher(fim: np.ndarray, couplings: np.ndarray) -> np.ndarray:
    """F_theta = g^T F g for one coupling vector g of shape (n,) or a batch of shape (k, n)."""
    g = np.atleast_2d(couplings)
    out = np.einsum("ki,ij,kj->k", g, fim, g)
    return out if np.ndim(couplings) > 1 else out[0]
