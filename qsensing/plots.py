"""Figures for the sensing-array model. Every function returns a matplotlib Figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .model import SensorCase, trace_distance


def sensor_arrays(x, y, cases: list[SensorCase], summary: list[dict], nx: int, ny: int):
    """Array layout: marker size shows B_i, colour shows phi_i."""
    fig, axes = plt.subplots(1, len(cases), figsize=(15, 4.5), constrained_layout=True)
    scatter = None
    for ax, case, row in zip(axes, cases, summary):
        span = np.ptp(case.amplitudes)
        if span < 1e-12:
            sizes = np.full_like(case.amplitudes, 140.0)
        else:
            sizes = 80.0 + 260.0 * (case.amplitudes - case.amplitudes.min()) / span
        scatter = ax.scatter(
            x, y, c=case.phases, s=sizes, cmap="twilight",
            vmin=-np.pi, vmax=np.pi, edgecolor="black", linewidth=0.6,
        )
        ax.set_title(
            f"{case.name}\n$B_{{eff}}={row['b_eff']:.2f}$, "
            f"$\\tau_{{detect}}={row['tau_detect']:.2f}$"
        )
        ax.set_aspect("equal")
        ax.set_xticks(range(nx))
        ax.set_yticks(range(ny))
        ax.set_xlim(-0.7, nx - 0.3)
        ax.set_ylim(-0.7, ny - 0.3)
        ax.grid(True, alpha=0.25)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, shrink=0.85).set_label(r"local phase $\phi_i$")
    fig.suptitle(r"Sensor array: marker size $\propto B_i$, colour $\propto \phi_i$")
    return fig


def trace_distance_curves(summary: list[dict], d_target: float):
    """D(T) for each case, with the detection threshold."""
    taus = [row["tau_detect"] for row in summary if np.isfinite(row["tau_detect"])]
    times = np.linspace(0.0, 1.4 * max(taus), 1000)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in summary:
        ax.plot(times, trace_distance(times, row["b_eff"]),
                label=f"{row['name']}, $B_{{eff}}={row['b_eff']:.2f}$")
        if np.isfinite(row["tau_detect"]):
            ax.axvline(row["tau_detect"], linestyle="--", alpha=0.3)
    ax.axhline(d_target, linestyle=":", label=f"target D = {d_target:.2f}")
    ax.set_xlabel("sensing time T")
    ax.set_ylabel(r"trace distance $D(T)$")
    ax.set_title(r"Distinguishability $D(T)=|\sin[\Theta(T)/2]|$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def time_scalings(summary: list[dict]):
    """Detection time next to the conventional and QSS scaling estimates."""
    names = [row["name"] for row in summary]
    pos = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(pos - width, [r["tau_detect"] for r in summary], width, label=r"$\tau_{\rm detect}$")
    ax.bar(pos, [r["tau_conv"] for r in summary], width, label=r"scaled $\tau_{\rm conv}$")
    ax.bar(pos + width, [r["tau_qss"] for r in summary], width, label=r"scaled $\tau_{\rm QSS}$")
    ax.set_xticks(pos)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("time / scaled time")
    ax.set_title("Detection time and heuristic sensing-time scalings")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    return fig
