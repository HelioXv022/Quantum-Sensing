"""Run the three-case comparison: python -m qsensing [--out figures/] [--show]."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import plots
from .model import analyze_case, generate_cases, sensor_grid


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(prog="qsensing", description=__doc__)
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--amplitude-std", type=float, default=0.35)
    parser.add_argument("--target-probability", type=float, default=0.90,
                        help="success probability to reach; sets D_target = 2P - 1")
    parser.add_argument("--out", type=Path, help="directory to save figures in")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args(argv)

    if not args.show:
        matplotlib.use("Agg")

    d_target = 2.0 * args.target_probability - 1.0
    rng = np.random.default_rng(args.seed)
    x, y = sensor_grid(args.nx, args.ny)
    cases = generate_cases(args.nx * args.ny, rng, amplitude_std=args.amplitude_std)
    summary = [analyze_case(case, d_target) for case in cases]

    print(f"{'case':<26}{'B_eff':>9}{'B_eff/NB0':>11}{'tau_detect':>12}{'tau_conv':>10}{'tau_QSS':>10}")
    for row in summary:
        print(f"{row['name']:<26}{row['b_eff']:>9.3f}{row['coherence_fraction']:>11.3f}"
              f"{row['tau_detect']:>12.4f}{row['tau_conv']:>10.4f}{row['tau_qss']:>10.4f}")

    figures = {
        "sensor_arrays.png": plots.sensor_arrays(x, y, cases, summary, args.nx, args.ny),
        "trace_distance.png": plots.trace_distance_curves(summary, d_target),
        "time_scalings.png": plots.time_scalings(summary),
    }
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(args.out / name, dpi=200, bbox_inches="tight")
        print(f"Saved {len(figures)} figures to {args.out}/")
    if args.show:
        plt.show()
    return summary


if __name__ == "__main__":
    main()
