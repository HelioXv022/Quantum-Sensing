"""Toy model of collective AC sensing with an array of quantum sensors.

Each sensor i sees a local AC signal with amplitude B_i and phase phi_i. In a
GHZ-like collective protocol the local signals add as phasors,

    B_eff = | sum_i B_i exp(i phi_i) |,

and the probe state rotates by Theta(T) = phase_factor * B_eff * T. The trace
distance between the no-signal and signal states is D(T) = |sin(Theta/2)|.

The conventional and quantum-search-sensing (QSS) times are scaling estimates
inspired by Allen et al., arXiv:2501.07625. They are not a simulation of the
QSS oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHASE_FACTOR = 2.0
DELTA_OMEGA = 10.0


@dataclass
class SensorCase:
    """Local signal amplitudes and phases for one array realization."""

    name: str
    amplitudes: np.ndarray
    phases: np.ndarray


def sensor_grid(nx: int, ny: int, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Flattened x, y coordinates of an nx-by-ny rectangular array."""
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    return spacing * x.ravel(), spacing * y.ravel()


def effective_signal(amplitudes: np.ndarray, phases: np.ndarray) -> float:
    """B_eff = |sum_i B_i exp(i phi_i)|."""
    return float(np.abs(np.sum(amplitudes * np.exp(1j * phases))))


def trace_distance(time, b_eff: float, phase_factor: float = PHASE_FACTOR):
    """D(T) = |sin(phase_factor * B_eff * T / 2)|."""
    return np.abs(np.sin(phase_factor * b_eff * np.asarray(time) / 2.0))


def success_probability(distance):
    """Helstrom success probability for equal priors, (1 + D) / 2."""
    return 0.5 * (1.0 + np.asarray(distance))


def detection_time(b_eff: float, d_target: float, phase_factor: float = PHASE_FACTOR) -> float:
    """First time at which D(T) reaches d_target: 2 arcsin(d_target) / (phase_factor * B_eff)."""
    if not 0.0 <= d_target <= 1.0:
        raise ValueError("d_target must lie in [0, 1]")
    if b_eff <= 0.0:
        return np.inf
    return float(2.0 * np.arcsin(d_target) / (phase_factor * b_eff))


def conventional_time(b_eff: float, delta_omega: float = DELTA_OMEGA) -> float:
    """Frequency-scanning scaling, tau_conv ~ delta_omega / B_eff^2 (constants dropped)."""
    return np.inf if b_eff <= 0.0 else float(delta_omega / b_eff**2)


def qss_time(b_eff: float, delta_omega: float = DELTA_OMEGA) -> float:
    """QSS scaling, tau_QSS ~ sqrt(delta_omega) / B_eff^(3/2) (constants and logs dropped)."""
    return np.inf if b_eff <= 0.0 else float(np.sqrt(delta_omega) / b_eff**1.5)


def generate_cases(
    n_sensors: int,
    rng: np.random.Generator,
    b0: float = 1.0,
    phi0: float = 0.0,
    amplitude_std: float = 0.35,
) -> list[SensorCase]:
    """Homogeneous, amplitude-disordered and phase-disordered arrays."""
    amplitudes = b0 * (1.0 + amplitude_std * rng.normal(size=n_sensors))
    return [
        SensorCase("Same B, same phi", np.full(n_sensors, b0), np.full(n_sensors, phi0)),
        SensorCase(
            "Disordered B_i, same phi",
            np.clip(amplitudes, 0.05 * b0, None),
            np.full(n_sensors, phi0),
        ),
        SensorCase(
            "Same B, random phi_i",
            np.full(n_sensors, b0),
            rng.uniform(-np.pi, np.pi, size=n_sensors),
        ),
    ]


def analyze_case(case: SensorCase, d_target: float, b0: float = 1.0) -> dict:
    """Effective signal, coherence fraction and characteristic times for one case."""
    b_eff = effective_signal(case.amplitudes, case.phases)
    return {
        "name": case.name,
        "b_eff": b_eff,
        "coherence_fraction": b_eff / (case.amplitudes.size * b0),
        "tau_detect": detection_time(b_eff, d_target),
        "tau_conv": conventional_time(b_eff),
        "tau_qss": qss_time(b_eff),
    }


def ensemble_effective_signal(
    n_sensors: int,
    n_trials: int,
    rng: np.random.Generator,
    b0: float = 1.0,
    amplitude_std: float = 0.0,
    random_phases: bool = False,
) -> np.ndarray:
    """B_eff for n_trials independent arrays; returns an array of length n_trials.

    With uniformly random phases and equal amplitudes, B_eff is a 2D random walk
    of n_sensors unit steps, so its mean approaches b0 * sqrt(pi * n_sensors) / 2.
    """
    shape = (n_trials, n_sensors)
    amplitudes = b0 * (1.0 + amplitude_std * rng.normal(size=shape))
    phases = rng.uniform(-np.pi, np.pi, size=shape) if random_phases else np.zeros(shape)
    return np.abs(np.sum(amplitudes * np.exp(1j * phases), axis=1))
