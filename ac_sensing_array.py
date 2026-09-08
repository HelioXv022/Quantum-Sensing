"""
AC Quantum Sensing Array Simulation
===================================

Toy numerical model for collective AC quantum sensing in a 2D sensor array.

The model compares three cases:
1. Homogeneous signal:       B_i = B0,       phi_i = phi0
2. Amplitude disorder:       random B_i,     phi_i = phi0
3. Phase disorder:           B_i = B0,       random phi_i

For a GHZ-like collective sensing picture, the local signals are compressed
into an effective complex phasor sum

    B_eff = | sum_i B_i exp(i phi_i) |.

The trace-distance model is

    D(T) = | sin(Theta(T)/2) |,

with the simplified matched-signal phase accumulation

    Theta(T) = phase_factor * B_eff * T.

The conventional and QSS sensing times included below are heuristic scaling
estimates inspired by arXiv:2501.07625. They are NOT a full simulation of the
quantum-search-sensing oracle.

Reference
---------
R. R. Allen, F. Machado, I. L. Chuang, H.-Y. Huang, and S. Choi,
"Quantum Computing Enhanced Sensing", arXiv:2501.07625 (2025).
https://arxiv.org/abs/2501.07625
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NX = 6
NY = 6
N_SENSORS = NX * NY

B0 = 1.0
PHI0 = 0.0

AMPLITUDE_DISORDER_STD = 0.35
RANDOM_SEED = 7

P_TARGET = 0.90
D_TARGET = 2.0 * P_TARGET - 1.0

# Simplified model:
# Theta(T) = PHASE_FACTOR * B_eff * T
PHASE_FACTOR = 2.0

# Used only for heuristic conventional / QSS scaling estimates.
DELTA_OMEGA = 10.0

# Plot / output options
SAVE_FIGURES = False
OUTPUT_DIR = Path("figures")


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SensorCase:
    """Local signal parameters for one sensor-array realization."""

    name: str
    amplitudes: np.ndarray
    phases: np.ndarray


# ---------------------------------------------------------------------------
# Sensor geometry
# ---------------------------------------------------------------------------

def make_sensor_grid(nx: int, ny: int, spacing: float = 1.0):
    """Return flattened x, y coordinates for a rectangular sensor array."""
    x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))
    return spacing * x_grid.ravel(), spacing * y_grid.ravel()


# ---------------------------------------------------------------------------
# Physics / toy-model functions
# ---------------------------------------------------------------------------

def effective_signal(amplitudes: np.ndarray, phases: np.ndarray) -> float:
    """
    Compute the collective effective signal

        B_eff = |sum_i B_i exp(i phi_i)|.
    """
    phasor_sum = np.sum(amplitudes * np.exp(1j * phases))
    return float(np.abs(phasor_sum))


def accumulated_phase(
    time: np.ndarray | float,
    b_eff: float,
    phase_factor: float = PHASE_FACTOR,
):
    """
    Simplified matched-signal accumulated phase

        Theta(T) = phase_factor * B_eff * T.
    """
    return phase_factor * b_eff * time


def trace_distance(
    time: np.ndarray | float,
    b_eff: float,
    phase_factor: float = PHASE_FACTOR,
):
    """
    Trace distance between the no-signal and signal-evolved GHZ-like states:

        D(T) = |sin(Theta(T)/2)|.
    """
    theta = accumulated_phase(time, b_eff, phase_factor)
    return np.abs(np.sin(theta / 2.0))


def success_probability(distance: np.ndarray | float):
    """
    Optimal binary discrimination success probability for equal priors:

        P_succ = 1/2 * (1 + D).
    """
    return 0.5 * (1.0 + distance)


def detection_time(
    b_eff: float,
    d_target: float = D_TARGET,
    phase_factor: float = PHASE_FACTOR,
) -> float:
    """
    First time tau such that D(tau) reaches d_target.

    From

        D(T) = |sin(phase_factor * B_eff * T / 2)|,

    the first crossing satisfies

        tau = 2 arcsin(d_target) / (phase_factor * B_eff).
    """
    if not 0.0 <= d_target <= 1.0:
        raise ValueError("d_target must lie in [0, 1].")

    if b_eff <= 1e-12:
        return np.inf

    return float(
        2.0 * np.arcsin(d_target) / (phase_factor * b_eff)
    )


def conventional_time_scaling(
    b_eff: float,
    delta_omega: float = DELTA_OMEGA,
) -> float:
    """
    Heuristic conventional frequency-scanning scaling:

        tau_conv ~ Delta_omega / B_eff^2.

    Constants and protocol-dependent factors are omitted.
    """
    if b_eff <= 1e-12:
        return np.inf
    return float(delta_omega / b_eff**2)


def qss_time_scaling(
    b_eff: float,
    delta_omega: float = DELTA_OMEGA,
) -> float:
    """
    Heuristic quantum-search-sensing scaling:

        tau_QSS ~ sqrt(Delta_omega) / B_eff^(3/2).

    Constants and logarithmic factors are omitted.
    """
    if b_eff <= 1e-12:
        return np.inf
    return float(np.sqrt(delta_omega) / b_eff**1.5)


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------

def generate_cases(
    n_sensors: int,
    b0: float,
    phi0: float,
    amplitude_disorder_std: float,
    rng: np.random.Generator,
):
    """Generate the three representative sensor-array cases."""

    # Case 1: same B, same phi
    homogeneous = SensorCase(
        name="Same B, same phi",
        amplitudes=np.full(n_sensors, b0),
        phases=np.full(n_sensors, phi0),
    )

    # Case 2: disordered B_i, same phi
    disordered_amplitudes = b0 * (
        1.0 + amplitude_disorder_std * rng.normal(size=n_sensors)
    )
    disordered_amplitudes = np.clip(
        disordered_amplitudes,
        0.05 * b0,
        None,
    )

    amplitude_disorder = SensorCase(
        name="Disordered B_i, same phi",
        amplitudes=disordered_amplitudes,
        phases=np.full(n_sensors, phi0),
    )

    # Case 3: same B, random phi_i
    phase_disorder = SensorCase(
        name="Same B, random phi_i",
        amplitudes=np.full(n_sensors, b0),
        phases=rng.uniform(-np.pi, np.pi, size=n_sensors),
    )

    return [homogeneous, amplitude_disorder, phase_disorder]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_case(case: SensorCase, n_sensors: int, b0: float):
    """Compute derived quantities for one case."""
    b_eff = effective_signal(case.amplitudes, case.phases)

    return {
        "name": case.name,
        "b_eff": b_eff,
        "coherence_fraction": b_eff / (n_sensors * b0),
        "tau_detect": detection_time(b_eff),
        "tau_conv_scaled": conventional_time_scaling(b_eff),
        "tau_qss_scaled": qss_time_scaling(b_eff),
    }


def print_summary(summary):
    """Print the numerical summary."""
    print("\n=== AC Quantum Sensing Array Summary ===")

    for row in summary:
        print(f"\nCase: {row['name']}")
        print(f"  B_eff               = {row['b_eff']:.4f}")
        print(
            f"  B_eff / (N B0)      = "
            f"{row['coherence_fraction']:.4f}"
        )
        print(f"  tau_detect          = {row['tau_detect']:.4f}")
        print(
            f"  tau_conv (scaled)   = "
            f"{row['tau_conv_scaled']:.4f}"
        )
        print(
            f"  tau_QSS (scaled)    = "
            f"{row['tau_qss_scaled']:.4f}"
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def maybe_save(fig, filename: str):
    """Save a figure when SAVE_FIGURES is enabled."""
    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=200,
            bbox_inches="tight",
        )


def plot_sensor_arrays(x, y, cases, summary, nx, ny):
    """
    Visualize the sensor array.

    Marker size   -> local amplitude B_i
    Marker color  -> local phase phi_i

    No arrows are drawn because phi_i is a phase in the complex plane,
    not a physical spatial direction.
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
        constrained_layout=True,
    )

    scatter = None

    for ax, case, row in zip(axes, cases, summary):
        amplitudes = case.amplitudes
        phases = case.phases

        amp_span = np.ptp(amplitudes)
        if amp_span < 1e-12:
            sizes = np.full_like(amplitudes, 140.0)
        else:
            sizes = 80.0 + 260.0 * (
                amplitudes - amplitudes.min()
            ) / amp_span

        scatter = ax.scatter(
            x,
            y,
            c=phases,
            s=sizes,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            edgecolor="black",
            linewidth=0.6,
        )

        ax.set_title(
            f"{case.name}\n"
            f"$B_{{eff}}={row['b_eff']:.2f}$, "
            f"$\\tau_{{detect}}={row['tau_detect']:.2f}$"
        )

        ax.set_aspect("equal")
        ax.set_xticks(range(nx))
        ax.set_yticks(range(ny))
        ax.set_xlim(-0.7, nx - 0.3)
        ax.set_ylim(-0.7, ny - 0.3)
        ax.grid(True, alpha=0.25)

    if scatter is not None:
        colorbar = fig.colorbar(
            scatter,
            ax=axes,
            shrink=0.85,
        )
        colorbar.set_label(r"local phase $\phi_i$")

    fig.suptitle(
        r"Sensor array: marker size $\propto B_i$, "
        r"color $\propto \phi_i$"
    )

    maybe_save(fig, "sensor_arrays.png")
    return fig


def plot_trace_distance(cases, summary):
    """Plot trace distance D(T) for the three cases."""
    finite_taus = [
        row["tau_detect"]
        for row in summary
        if np.isfinite(row["tau_detect"])
    ]

    if not finite_taus:
        return None

    t_max = 1.4 * max(finite_taus)
    times = np.linspace(0.0, t_max, 1000)

    fig, ax = plt.subplots(figsize=(8, 5))

    for case, row in zip(cases, summary):
        distances = trace_distance(
            times,
            row["b_eff"],
        )

        ax.plot(
            times,
            distances,
            label=(
                f"{case.name}, "
                f"$B_{{eff}}={row['b_eff']:.2f}$"
            ),
        )

        if np.isfinite(row["tau_detect"]):
            ax.axvline(
                row["tau_detect"],
                linestyle="--",
                alpha=0.30,
            )

    ax.axhline(
        D_TARGET,
        linestyle=":",
        label=f"target D = {D_TARGET:.2f}",
    )

    ax.set_xlabel("sensing time T")
    ax.set_ylabel(r"trace distance $D(T)$")
    ax.set_title(
        r"Distinguishability: "
        r"$D(T)=|\sin[\Theta(T)/2]|$"
    )
    ax.grid(True, alpha=0.30)
    ax.legend()

    maybe_save(fig, "trace_distance.png")
    return fig


def plot_effective_signal(summary):
    """Compare B_eff for the three cases."""
    names = [row["name"] for row in summary]
    b_eff_values = [row["b_eff"] for row in summary]
    positions = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(positions, b_eff_values)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        names,
        rotation=20,
        ha="right",
    )
    ax.set_ylabel(r"$B_{\rm eff}$")
    ax.set_title(
        r"Effective collective signal "
        r"$B_{\rm eff}=|\sum_i B_i e^{i\phi_i}|$"
    )
    ax.grid(axis="y", alpha=0.30)

    maybe_save(fig, "effective_signal.png")
    return fig


def plot_time_scalings(summary):
    """
    Compare detection time and the two heuristic frequency-search scalings.

    Note:
    tau_detect, tau_conv_scaled, and tau_qss_scaled are not identical physical
    quantities; the latter two are included only to visualize scaling trends.
    """
    names = [row["name"] for row in summary]
    positions = np.arange(len(names))

    tau_detect = np.array(
        [row["tau_detect"] for row in summary]
    )
    tau_conv = np.array(
        [row["tau_conv_scaled"] for row in summary]
    )
    tau_qss = np.array(
        [row["tau_qss_scaled"] for row in summary]
    )

    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        positions - width,
        tau_detect,
        width=width,
        label=r"$\tau_{\rm detect}$",
    )
    ax.bar(
        positions,
        tau_conv,
        width=width,
        label=r"scaled $\tau_{\rm conv}$",
    )
    ax.bar(
        positions + width,
        tau_qss,
        width=width,
        label=r"scaled $\tau_{\rm QSS}$",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        names,
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("time / scaled time")
    ax.set_yscale("log")
    ax.set_title("Detection time and heuristic sensing-time scalings")
    ax.grid(axis="y", alpha=0.30)
    ax.legend()

    maybe_save(fig, "time_scalings.png")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    x, y = make_sensor_grid(NX, NY)

    cases = generate_cases(
        n_sensors=N_SENSORS,
        b0=B0,
        phi0=PHI0,
        amplitude_disorder_std=AMPLITUDE_DISORDER_STD,
        rng=rng,
    )

    summary = [
        analyze_case(case, N_SENSORS, B0)
        for case in cases
    ]

    print_summary(summary)

    plot_sensor_arrays(
        x,
        y,
        cases,
        summary,
        NX,
        NY,
    )
    plot_trace_distance(cases, summary)
    plot_effective_signal(summary)
    plot_time_scalings(summary)

    plt.show()


if __name__ == "__main__":
    main()
