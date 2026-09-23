"""How B_eff grows with array size: python -m qsensing.scaling [--out docs/]."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from .model import ensemble_effective_signal


def run(sizes: np.ndarray, n_trials: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    amp = [ensemble_effective_signal(n, n_trials, rng, amplitude_std=0.35) for n in sizes]
    phase = [ensemble_effective_signal(n, n_trials, rng, random_phases=True) for n in sizes]
    return {
        "sizes": sizes,
        "coherent": sizes.astype(float),
        "amplitude_mean": np.array([a.mean() for a in amp]),
        "phase_mean": np.array([p.mean() for p in phase]),
        "phase_p10": np.array([np.percentile(p, 10) for p in phase]),
        "phase_p90": np.array([np.percentile(p, 90) for p in phase]),
    }


def plot(result: dict[str, np.ndarray]):
    import matplotlib.pyplot as plt

    n = result["sizes"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(n, result["coherent"], "o-", label="same phase: $B_{eff} = N B$")
    ax.loglog(n, result["amplitude_mean"], "s--", label="amplitude disorder (mean)")
    ax.loglog(n, result["phase_mean"], "^-", label="random phases (mean)")
    ax.fill_between(n, result["phase_p10"], result["phase_p90"], alpha=0.2, color="C2",
                    label="random phases, 10-90%")
    ax.loglog(n, np.sqrt(np.pi * n) / 2, "k:", label=r"$\sqrt{\pi N}/2$")
    ax.set_xlabel("number of sensors $N$")
    ax.set_ylabel(r"$B_{eff}$ (units of $B$)")
    ax.set_title("Collective signal vs array size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="qsensing.scaling", description=__doc__)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    matplotlib.use("Agg")

    sizes = 2 ** np.arange(2, 11)
    result = run(sizes, args.trials, args.seed)
    slope = np.polyfit(np.log(sizes), np.log(result["phase_mean"]), 1)[0]
    print(f"{'N':>6}{'coherent':>10}{'amp. dis.':>11}{'rand. phase':>13}")
    for i, n in enumerate(sizes):
        print(f"{n:>6}{result['coherent'][i]:>10.1f}{result['amplitude_mean'][i]:>11.1f}"
              f"{result['phase_mean'][i]:>13.2f}")
    print(f"log-log slope for random phases: {slope:.3f} (random walk: 0.5)")

    args.out.mkdir(parents=True, exist_ok=True)
    plot(result).savefig(args.out / "scaling.png", dpi=150)
    print(f"Saved {args.out / 'scaling.png'}")


if __name__ == "__main__":
    main()
