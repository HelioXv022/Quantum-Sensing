# AC Quantum Sensing Array Simulation

[![CI](https://github.com/HelioXv022/Quantum-Sensing/actions/workflows/ci.yml/badge.svg)](https://github.com/HelioXv022/Quantum-Sensing/actions/workflows/ci.yml)

A small Python model of collective AC field sensing with an array of quantum sensors. It shows how spatial phase disorder removes the coherent enhancement of a GHZ-like probe: the collective signal drops from growing like $N$ to growing like $\sqrt{N}$.

## Model

Sensor $i$ sees a local signal with amplitude $B_i$ and phase $\phi_i$. In a collective protocol the local signals add as phasors,

$$
B_{\rm eff} = \Big|\sum_i B_i e^{i\phi_i}\Big| ,
$$

and the probe accumulates a phase $\Theta(T) = 2 B_{\rm eff} T$. The trace distance between the no-signal and signal states is

$$
D(T) = \left|\sin\frac{\Theta(T)}{2}\right| ,
$$

so the time needed to reach a fixed distinguishability scales as $1/B_{\rm eff}$.

Three arrays are compared:

| Case | Amplitudes | Phases | Typical $B_{\rm eff}$ |
| --- | --- | --- | --- |
| Homogeneous | $B_i = B$ | $\phi_i = \phi$ | $NB$ |
| Amplitude disorder | random $B_i$ | $\phi_i = \phi$ | $\approx NB$ |
| Phase disorder | $B_i = B$ | random $\phi_i$ | $\approx \tfrac{\sqrt{\pi N}}{2}B$ |

The code also plots two heuristic sensing-time scalings for an unknown frequency in a band $\Delta\omega$: conventional scanning, $\tau_{\rm conv} \sim \Delta\omega / B_{\rm eff}^2$, and quantum search sensing, $\tau_{\rm QSS} \sim \sqrt{\Delta\omega} / B_{\rm eff}^{3/2}$, following Allen et al. (below). These are scaling estimates with constants dropped, not a simulation of the QSS oracle.

## Result

![B_eff versus number of sensors](docs/scaling.png)

Averaged over 2000 random arrays per size, the random-phase signal follows the random-walk mean $\sqrt{\pi N}/2$ with a fitted exponent of 0.495. Amplitude disorder with a 35% spread leaves $B_{\rm eff}$ within a fraction of a percent of $NB$ on average. At $N = 1024$ the phase-disordered array needs about 36 times longer than the coherent one to reach the same trace distance.

## Usage

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

python -m qsensing --out figures/          # three-case comparison on a 6x6 array
python -m qsensing.scaling --out figures/  # B_eff vs N, as in the figure above
pytest                                     # tests
```

`python -m qsensing --help` lists the options (array size, seed, amplitude spread, target success probability).

## Layout

```text
qsensing/
  model.py      effective signal, trace distance, detection time, case generation
  plots.py      figures for the three-case comparison
  scaling.py    ensemble study of B_eff vs N
  __main__.py   command-line entry point
tests/          pytest suite
```

## Reference

R. R. Allen, F. Machado, I. L. Chuang, H.-Y. Huang and S. Choi, *Quantum Computing Enhanced Sensing*, [arXiv:2501.07625](https://arxiv.org/abs/2501.07625) (2025).

## License

MIT
