"""Fisher information of GHZ and product probes under spatial phase disorder.

    python -m qsensing.disorder --out docs/

Sensor i sees the AC signal with phase phi_i, so after demodulation its qubit
picks up g_i * theta with g_i = cos(phi_i - alpha), where alpha is the common
reference phase of the control sequence. The per-sensor QFIM comes from squint
(qsensing.fisher); the disorder average is then just the quadratic form g^T F g.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

from .fisher import fisher_matrices, signal_fisher

STRATEGIES = {
    "ghz_known": r"GHZ, known phases ($X$ on sensors with $g_i < 0$)",
    "ghz_best_ref": "GHZ, random phases, best common reference",
    "ghz_random": "GHZ, random phases",
    "product_random": "product state, random phases",
}


def disorder_average(qfim_product, qfim_ghz, phases: np.ndarray) -> dict[str, np.ndarray]:
    """F_theta for each disorder draw (rows of `phases`) and each strategy."""
    g = np.cos(phases)
    alpha = np.angle(np.exp(1j * phases).sum(axis=1, keepdims=True))
    g_best = np.cos(phases - alpha)
    # X on sensor i maps Z_i -> -Z_i, so the QFIM becomes D F D with D = diag(sign g);
    # applying the signs to g instead is equivalent.
    return {
        "ghz_known": signal_fisher(qfim_ghz, np.abs(g)),
        "ghz_best_ref": signal_fisher(qfim_ghz, g_best),
        "ghz_random": signal_fisher(qfim_ghz, g),
        "product_random": signal_fisher(qfim_product, g),
    }


def predictions(n: np.ndarray) -> dict[str, np.ndarray]:
    """Disorder averages for phi_i uniform on [-pi, pi)."""
    return {
        "ghz_known": (2 / np.pi) ** 2 * n**2 + (0.5 - 4 / np.pi**2) * n,
        "ghz_best_ref": n.astype(float),
        "ghz_random": n / 2,
        "product_random": n / 2,
    }


def run(sizes, n_draws: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = {key: [] for key in STRATEGIES}
    for n in sizes:
        qfim_product, _ = fisher_matrices(n, "product")
        qfim_ghz, _ = fisher_matrices(n, "ghz")
        values = disorder_average(qfim_product, qfim_ghz, rng.uniform(-np.pi, np.pi, (n_draws, n)))
        for key in STRATEGIES:
            means[key].append(values[key].mean())
    return {"sizes": np.asarray(sizes), **{k: np.asarray(v) for k, v in means.items()}}


COLORS = {  # fixed categorical order; markers and direct labels carry identity too
    "ghz_known": ("#2a78d6", "o"),
    "ghz_best_ref": ("#eb6834", "^"),
    "ghz_random": ("#1baf7a", "s"),
    "product_random": ("#eda100", "D"),
}
DIRECT_LABELS = {
    "ghz_known": r"GHZ, known phases: $\approx 0.41\,N^2$",
    "ghz_best_ref": r"GHZ, best reference: $N$",
    "ghz_random": r"GHZ or product: $N/2$",
}


def plot(result):
    import matplotlib.pyplot as plt

    n = result["sizes"]
    theory = predictions(n)
    ink, muted = "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(n, n**2, color=muted, lw=1, ls="--")
    ax.annotate(r"$N^2$ (no disorder)", (n[-1], n[-1] ** 2), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=muted)
    for key, label in STRATEGIES.items():
        color, marker = COLORS[key]
        ax.loglog(n, theory[key], color=color, lw=1.5)
        size = 10 if key == "ghz_random" else 7  # keeps both coinciding N/2 series visible
        ax.loglog(n, result[key], marker=marker, ms=size, ls="none", color=color,
                  markeredgecolor="white", markeredgewidth=1, label=label)
        if key in DIRECT_LABELS:
            ax.annotate(DIRECT_LABELS[key], (n[-1], theory[key][-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=ink)
    ax.set_xlabel("number of sensors $N$", color=ink)
    ax.set_ylabel(r"mean QFI for the signal, $\langle F_\theta \rangle$", color=ink)
    ax.set_title("GHZ advantage under spatial phase disorder", color=ink, loc="left")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(n)
    ax.set_xticklabels([str(k) for k in n])
    ax.set_xlim(n[0] * 0.9, n[-1] * 2.1)
    ax.tick_params(colors=muted, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", color="#e6e5e0", lw=0.8)
    ax.legend(fontsize=8, frameon=False, loc="upper left",
              title="markers: squint · lines: analytic", title_fontsize=8)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="qsensing.disorder", description=__doc__)
    parser.add_argument("--max-n", type=int, default=12)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args(argv)
    matplotlib.use("Agg")

    sizes = np.arange(2, args.max_n + 1)
    result = run(sizes, args.draws, args.seed)
    theory = predictions(sizes)
    print(f"{'N':>3}" + "".join(f"{k:>16}" for k in STRATEGIES) + f"{'N^2':>8}")
    for i, n in enumerate(sizes):
        print(f"{n:>3}" + "".join(f"{result[k][i]:>9.2f} ({theory[k][i]:>4.1f})" for k in STRATEGIES)
              + f"{n**2:>8}")
    args.out.mkdir(parents=True, exist_ok=True)
    plot(result).savefig(args.out / "ghz_disorder.png", dpi=150)
    print(f"Saved {args.out / 'ghz_disorder.png'} (values in parentheses: analytic averages)")


if __name__ == "__main__":
    main()
